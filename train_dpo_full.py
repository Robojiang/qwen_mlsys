import os
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import wandb

@hydra.main(version_base=None, config_path="configs", config_name="dpo_full")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    if cfg.wandb.project:
        os.environ["WANDB_PROJECT"] = cfg.wandb.project
    if cfg.wandb.entity:
        os.environ["WANDB_ENTITY"] = cfg.wandb.entity

    model_name_or_path = cfg.model.model_name_or_path
    if os.path.exists(to_absolute_path(model_name_or_path)):
        model_name_or_path = to_absolute_path(model_name_or_path)

    print(f"Loading model from {model_name_or_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, 
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model (Full Weights)
    print(f"Loading full model from {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.training.bf16 else torch.float16,
        device_map="auto"
    )
    
    # For Full DPO, we usually need a reference model.
    # If we don't provide one, DPOTrainer will copy 'model' to create 'ref_model'.
    # This doubles memory usage (Model + Ref Model).
    # On a single H100 80GB, 4B model (8GB) * 2 = 16GB.
    # Plus Optimizer (16GB) + Gradients (8GB) + Activations.
    # Total ~40GB+. It should fit.
    
    print("Full Fine-Tuning Mode (No PEFT)")

    train_file = cfg.data.train_file
    if os.path.exists(to_absolute_path(train_file)):
        train_file = to_absolute_path(train_file)

    if os.path.exists(train_file):
        print(f"Loading dataset from {train_file}")
        dataset = load_dataset("json", data_files=train_file, split="train")
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
        save_steps=cfg.training.get("save_steps", 100),
        num_train_epochs=cfg.training.num_train_epochs,
        max_steps=cfg.training.get("max_steps", -1),
        report_to=cfg.training.report_to,
        run_name=cfg.training.run_name,
        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,
        beta=cfg.training.beta,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        max_length=2048,
        max_prompt_length=1024,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None, # Trainer will create a copy
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting Full DPO training...")
    trainer.train()
    
    final_dir = os.path.join(output_dir, "final_model")
    print(f"Saving model to {final_dir}")
    trainer.save_model(final_dir)

if __name__ == "__main__":
    main()
