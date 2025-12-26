import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import wandb

@hydra.main(version_base=None, config_path="configs", config_name="dpo_arc")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # 1. 初始化 WandB
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.training.run_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # 2. 加载模型和 Tokenizer
    # DPO 需要一个参考模型 (ref_model)，通常如果不指定，DPOTrainer 会自动复制一份 model 作为 ref_model
    # 为了节省显存，可以使用 PEFT，这样 ref_model 可以是基础模型，model 是加载了 LoRA 的模型
    
    print(f"Loading model from {cfg.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name_or_path, 
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.training.bf16 else torch.float16,
        device_map="auto"
    )

    # 3. 配置 PEFT (LoRA)
    peft_config = None
    if cfg.model.use_peft:
        target_modules = OmegaConf.to_container(cfg.model.target_modules, resolve=True)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            target_modules=target_modules
        )
        print("LoRA enabled for DPO.")

    # 4. 加载数据
    # DPO 数据需要包含三列: prompt, chosen, rejected
    if os.path.exists(cfg.data.train_file):
        if cfg.data.train_file.endswith(".arrow"):
            print(f"Loading Arrow dataset from {cfg.data.train_file}")
            dataset = load_dataset("arrow", data_files={"train": cfg.data.train_file}, split="train")
        else:
            print(f"Loading JSON dataset from {cfg.data.train_file}")
            dataset = load_dataset("json", data_files=cfg.data.train_file, split="train")
    else:
        print(f"Warning: Data file {cfg.data.train_file} not found. Using dummy data for demonstration.")
        dummy_data = [
            {
                "prompt": "User: How are you?",
                "chosen": "Assistant: I am doing well, thank you!",
                "rejected": "Assistant: I don't know."
            },
            {
                "prompt": "User: What is 1+1?",
                "chosen": "Assistant: The answer is 2.",
                "rejected": "Assistant: It is 3."
            }
        ]
        from datasets import Dataset
        dataset = Dataset.from_list(dummy_data)

    # 5. 训练参数
    output_dir = HydraConfig.get().runtime.output_dir
    print(f"Output directory set to: {output_dir}")

    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        report_to=cfg.training.report_to,
        run_name=cfg.training.run_name,
        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        beta=cfg.data.beta,
        max_length=cfg.data.max_seq_length,
        max_prompt_length=512,
    )

    # 6. 初始化 DPOTrainer
    # ARC 数据集没有 chosen/rejected 字段，需要构造 DPO 格式
    # 这里我们使用一个简单的策略：
    # prompt: User: {question}\nChoices:\n{options}\nAnswer:\nAssistant:
    # chosen: {answerKey}
    # rejected: {random_wrong_answer}
    
    if "question" in dataset.column_names and "choices" in dataset.column_names:
        print("Detected ARC dataset. Converting to DPO format...")
        import random
        
        def format_dpo(example):
            question = example['question']
            choices = example['choices']
            answer_key = example['answerKey']
            
            # 构建 prompt
            options = []
            label_to_text = {}
            for label, text in zip(choices['label'], choices['text']):
                options.append(f"{label}. {text}")
                label_to_text[label] = text
            options_str = "\n".join(options)
            
            prompt = f"User: {question}\nChoices:\n{options_str}\nAnswer:\nAssistant: "
            
            # 构建 chosen
            chosen = answer_key
            
            # 构建 rejected (随机选一个错误的选项)
            all_labels = choices['label']
            wrong_labels = [l for l in all_labels if l != answer_key]
            if not wrong_labels:
                # 极少数情况可能只有一个选项且是正确的，或者数据异常
                rejected = "Unknown" 
            else:
                rejected = random.choice(wrong_labels)
            
            return {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            }
        
        # 移除原始列，只保留 DPO 需要的列
        original_columns = dataset.column_names
        dataset = dataset.map(format_dpo, remove_columns=original_columns)
        print(f"Converted dataset size: {len(dataset)}")
        print("Sample data:", dataset[0])

    trainer = DPOTrainer(
        model=model,
        ref_model=None, # 如果使用 PEFT，ref_model 可以为 None，Trainer 会自动处理
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 7. 开始训练
    print("Starting DPO training...")
    trainer.train()
    
    # 8. 保存模型
    final_dir = os.path.join(output_dir, "final_model")
    print(f"Saving model to {final_dir}")
    trainer.save_model(final_dir)

if __name__ == "__main__":
    main()
