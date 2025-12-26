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

@hydra.main(version_base=None, config_path="configs", config_name="sft_piqa")
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
        print("LoRA enabled.")

    data_file = cfg.data.train_file
    if os.path.exists(data_file):
        if data_file.endswith(".arrow"):
            print(f"Loading Arrow dataset from {data_file}")
            dataset = load_dataset("arrow", data_files={"train": data_file}, split="train")
        else:
            print(f"Loading JSON dataset from {data_file}")
            dataset = load_dataset("json", data_files=data_file, split="train")
    else:
        raise FileNotFoundError(f"Data file {data_file} not found.")

    output_dir = HydraConfig.get().runtime.output_dir
    print(f"Output directory set to: {output_dir}")

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.get("save_strategy", "steps"),
        save_steps=cfg.training.get("save_steps", 500),
        save_total_limit=cfg.training.get("save_total_limit", None),
        num_train_epochs=cfg.training.num_train_epochs,
        report_to=cfg.training.report_to,
        run_name=cfg.training.run_name,
        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        max_length=cfg.data.max_seq_length,
        dataset_text_field=cfg.data.dataset_text_field,
    )

    # PIQA Formatting
    # PIQA: goal, sol1, sol2, label
    # Format: Question: {goal}\nAnswer: {sol1/sol2}
    def piqa_formatting_func(example):
        is_batch = isinstance(example['goal'], list)
        if is_batch:
            output_texts = []
            for i in range(len(example['goal'])):
                goal = example['goal'][i]
                sol1 = example['sol1'][i]
                sol2 = example['sol2'][i]
                label = example['label'][i] # 0 or 1
                
                correct_sol = sol1 if label == 0 else sol2
                text = f"Question: {goal}\nAnswer: {correct_sol}"
                output_texts.append(text)
            return output_texts
        else:
            goal = example['goal']
            sol1 = example['sol1']
            sol2 = example['sol2']
            label = example['label']
            correct_sol = sol1 if label == 0 else sol2
            return f"Question: {goal}\nAnswer: {correct_sol}"

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "processing_class": tokenizer,
        "args": training_args,
        "peft_config": peft_config,
        "formatting_func": piqa_formatting_func
    }
    
    trainer = SFTTrainer(**trainer_kwargs)

    print("Starting training...")
    trainer.train()
    
    final_dir = os.path.join(output_dir, "final_model")
    print(f"Saving model to {final_dir}")
    trainer.save_model(final_dir)

if __name__ == "__main__":
    main()
