import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import wandb

@hydra.main(version_base=None, config_path="configs", config_name="dpo_hellaswag")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize WandB
    if cfg.wandb.project:
        os.environ["WANDB_PROJECT"] = cfg.wandb.project
    if cfg.wandb.entity:
        os.environ["WANDB_ENTITY"] = cfg.wandb.entity

    print(f"Loading model from {cfg.model.model_name_or_path}")
    
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name_or_path, 
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    print(f"Loading base model from {cfg.model.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.training.bf16 else torch.float16,
        device_map="auto"
    )

    # If SFT adapter is provided, load and merge it
    if cfg.model.get("sft_adapter_path") and os.path.exists(cfg.model.sft_adapter_path):
        print(f"Loading SFT adapter from {cfg.model.sft_adapter_path}")
        # We need to load the adapter and merge it to create the new base for DPO
        # This ensures the reference model (which will be a copy of this) also has the SFT weights
        model = PeftModel.from_pretrained(model, cfg.model.sft_adapter_path)
        print("Merging SFT adapter into base model...")
        model = model.merge_and_unload()
    
    # DPO requires a reference model. 
    # If we don't provide one, DPOTrainer creates a copy.
    # To save memory, we can use PEFT. The 'model' will be optimized (with LoRA), 
    # and the reference model is effectively the base model (frozen).
    # However, strictly speaking, ref_model should be the SFT model.
    # If we use LoRA for DPO, 'model' is SFT+LoRA_DPO, and 'ref_model' is SFT.
    # If we load the SFT model as 'model', and add LoRA, then 'model' becomes SFT+LoRA.
    # The DPOTrainer will treat the underlying model as ref if we don't pass ref_model?
    # No, DPOTrainer constructs ref_model = deepcopy(model) if ref_model is None.
    # If we use PEFT, DPOTrainer is smart enough to use the adapter-disabled model as reference?
    # Actually, with PEFT, we usually don't pass ref_model. The trainer handles it by disabling adapters to get reference logits.
    
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

    # Load Dataset
    if os.path.exists(cfg.data.train_file):
        print(f"Loading dataset from {cfg.data.train_file}")
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
        ref_model=None, # Let trainer handle reference model (via PEFT or copy)
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
