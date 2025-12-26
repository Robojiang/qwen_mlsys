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

@hydra.main(version_base=None, config_path="configs", config_name="dpo_winogrande")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.training.run_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    print(f"Loading model from {cfg.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name_or_path, 
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.training.bf16 else torch.float16,
        device_map="auto"
    )

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

    if os.path.exists(cfg.data.train_file):
        if cfg.data.train_file.endswith(".arrow"):
            print(f"Loading Arrow dataset from {cfg.data.train_file}")
            dataset = load_dataset("arrow", data_files={"train": cfg.data.train_file}, split="train")
        else:
            print(f"Loading JSON dataset from {cfg.data.train_file}")
            dataset = load_dataset("json", data_files=cfg.data.train_file, split="train")
    else:
        raise FileNotFoundError(f"Data file {cfg.data.train_file} not found.")

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

    # Winogrande DPO Formatting
    # Winogrande: sentence, option1, option2, answer
    # Prompt: Context: {sentence}\nAnswer:
    # Chosen: {correct_option}
    # Rejected: {wrong_option}
    
    if "sentence" in dataset.column_names and "option1" in dataset.column_names:
        print("Detected Winogrande dataset. Converting to DPO format...")
        
        def format_dpo(example):
            sentence = example['sentence']
            option1 = example['option1']
            option2 = example['option2']
            answer = example['answer'] # "1" or "2"
            
            prompt = f"Context: {sentence}\nAnswer:"
            
            if answer == "1":
                chosen = option1
                rejected = option2
            else:
                chosen = option2
                rejected = option1
            
            return {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            }
        
        original_columns = dataset.column_names
        dataset = dataset.map(format_dpo, remove_columns=original_columns)
        print(f"Converted dataset size: {len(dataset)}")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting DPO training...")
    trainer.train()
    
    final_dir = os.path.join(output_dir, "final_model")
    print(f"Saving model to {final_dir}")
    trainer.save_model(final_dir)

if __name__ == "__main__":
    main()
