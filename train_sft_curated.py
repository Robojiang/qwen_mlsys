import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import wandb

@hydra.main(version_base=None, config_path="configs", config_name="sft_curated")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Set WandB project and entity via environment variables
    # SFTTrainer will automatically pick these up
    if cfg.wandb.project:
        os.environ["WANDB_PROJECT"] = cfg.wandb.project
    if cfg.wandb.entity:
        os.environ["WANDB_ENTITY"] = cfg.wandb.entity
    
    print(f"Loading model from {cfg.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name_or_path, 
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Force set ChatML template to avoid Jinja errors with base model templates
    print("Setting ChatML template...")
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

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

    if os.path.exists(cfg.data.train_file):
        print(f"Loading dataset from {cfg.data.train_file}")
        dataset = load_dataset("json", data_files=cfg.data.train_file, split="train")
    else:
        raise FileNotFoundError(f"Data file {cfg.data.train_file} not found.")

    output_dir = HydraConfig.get().runtime.output_dir
    print(f"Output directory set to: {output_dir}")

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.get("save_steps", 500),
        save_total_limit=cfg.training.save_total_limit,
        num_train_epochs=cfg.training.num_train_epochs,
        report_to=cfg.training.report_to,
        run_name=cfg.training.run_name,
        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        max_length=cfg.data.max_seq_length,
        dataset_text_field=None, # We use formatting_func
        packing=False, # Can enable packing for efficiency if needed
    )

    def formatting_prompts_func(example):
        output_texts = []
        # Check if the input is a batch (list of lists of messages) or a single example (list of messages)
        # example['messages'] is the field.
        # If batched=False (which seems to be the case here), example is a single dict.
        # example['messages'] is a list of dicts (the conversation).
        
        messages = example['messages']
        
        # Handle single example case (list of dicts)
        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return [text] # Return list of strings as expected by SFTTrainer
            
        # Handle batched case (list of lists of dicts) - just in case
        elif isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], list):
            for msg_list in messages:
                text = tokenizer.apply_chat_template(msg_list, tokenize=False, add_generation_prompt=False)
                output_texts.append(text)
            return output_texts
            
        # Fallback or empty
        return []

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func
    )

    print("Starting SFT training...")
    trainer.train()
    
    final_dir = os.path.join(output_dir, "final_model")
    print(f"Saving model to {final_dir}")
    trainer.save_model(final_dir)

if __name__ == "__main__":
    main()
