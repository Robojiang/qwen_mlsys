import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import wandb

@hydra.main(version_base=None, config_path="configs", config_name="sft_arc")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # 1. 初始化 WandB
    # 建议在环境变量中设置 WANDB_API_KEY，或者在终端运行 wandb login
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.training.run_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    # 2. 加载模型和 Tokenizer
    print(f"Loading model from {cfg.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name_or_path, 
        trust_remote_code=True,
        padding_side="right" # SFT 通常用 right padding
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.training.bf16 else torch.float16,
        device_map="auto"
    )

    # 3. 配置 PEFT (LoRA)
    peft_config = None
    if cfg.model.use_peft:
        # 将 OmegaConf 的 ListConfig 转换为普通 list，否则 PEFT 保存时会报错
        target_modules = OmegaConf.to_container(cfg.model.target_modules, resolve=True)
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=cfg.model.lora_r,
            lora_alpha=cfg.model.lora_alpha,
            lora_dropout=cfg.model.lora_dropout,
            target_modules=target_modules
        )
        print("LoRA enabled.")

    # 4. 加载数据
    data_file = cfg.data.train_file
    if os.path.exists(data_file):
        if data_file.endswith(".arrow"):
            print(f"Loading Arrow dataset from {data_file}")
            dataset = load_dataset("arrow", data_files={"train": data_file}, split="train")
        else:
            print(f"Loading JSON dataset from {data_file}")
            dataset = load_dataset("json", data_files=data_file, split="train")
    else:
        print(f"Warning: Data file {data_file} not found. Using dummy data for demonstration.")
        # 创建 dummy 数据
        dummy_data = [
            {"text": "User: Hello\nAssistant: Hi there!"},
            {"text": "User: What is AI?\nAssistant: AI stands for Artificial Intelligence."}
        ]
        from datasets import Dataset
        dataset = Dataset.from_list(dummy_data)

    # 5. 训练参数
    # 使用 Hydra 运行时生成的输出目录，确保日志和模型都在同一位置
    output_dir = HydraConfig.get().runtime.output_dir
    print(f"Output directory set to: {output_dir}")

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.get("save_strategy", "steps"), # 支持从配置读取，默认 steps
        save_steps=cfg.training.get("save_steps", 500),
        save_total_limit=cfg.training.get("save_total_limit", None),
        num_train_epochs=cfg.training.num_train_epochs,
        report_to=cfg.training.report_to,
        run_name=cfg.training.run_name,
        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,
        gradient_checkpointing=True, # 节省显存
        remove_unused_columns=False,
        max_length=cfg.data.max_seq_length, # max_seq_length -> max_length
        dataset_text_field=cfg.data.dataset_text_field,
    )

    # 定义格式化函数 (针对 ARC 数据集)
    formatting_func = None
    if "question" in dataset.column_names and "choices" in dataset.column_names:
        print("Detected ARC-style dataset. Using custom formatting function.")
        def arc_formatting_func(example):
            # 判断输入是单个样本还是 batch
            is_batch = isinstance(example['question'], list)
            
            if is_batch:
                output_texts = []
                for i in range(len(example['question'])):
                    question = example['question'][i]
                    choices = example['choices'][i]
                    answer_key = example['answerKey'][i]
                    
                    options = []
                    for label, text in zip(choices['label'], choices['text']):
                        options.append(f"{label}. {text}")
                    options_str = "\n".join(options)
                    
                    text = f"User: {question}\nChoices:\n{options_str}\nAnswer:\nAssistant: {answer_key}"
                    output_texts.append(text)
                return output_texts
            else:
                # 处理单个样本
                question = example['question']
                choices = example['choices']
                answer_key = example['answerKey']
                
                options = []
                for label, text in zip(choices['label'], choices['text']):
                    options.append(f"{label}. {text}")
                options_str = "\n".join(options)
                
                text = f"User: {question}\nChoices:\n{options_str}\nAnswer:\nAssistant: {answer_key}"
                return text # 返回单个字符串，而不是列表
   
        formatting_func = arc_formatting_func

    # 6. 初始化 Trainer
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "processing_class": tokenizer,
        "args": training_args,
        "peft_config": peft_config,
    }

    if formatting_func:
        trainer_kwargs["formatting_func"] = formatting_func
    
    trainer = SFTTrainer(**trainer_kwargs)

    # 7. 开始训练
    print("Starting training...")
    trainer.train()
    
    final_dir = os.path.join(output_dir, "final_model")
    # 8. 保存模型
    print(f"Saving model to {final_dir}")
    trainer.save_model(final_dir)

if __name__ == "__main__":
    main()
