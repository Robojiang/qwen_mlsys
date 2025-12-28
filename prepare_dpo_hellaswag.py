import json
import random
import os
import glob
from datasets import load_dataset
from tqdm import tqdm

LOCAL_CACHE_PATH = "/mnt/afs/250010074/qwen/benchmark_cache/datasets/Rowan___hellaswag"

def prepare_hellaswag_dpo():
    # Load HellaSwag dataset
    print("Loading HellaSwag dataset from local cache...")
    
    # Find arrow files
    arrow_files = glob.glob(os.path.join(LOCAL_CACHE_PATH, "**", "*.arrow"), recursive=True)
    train_files = [f for f in arrow_files if "train" in os.path.basename(f)]
    
    dataset = None
    if train_files:
        print(f"Found local train files: {train_files}")
        try:
            dataset = load_dataset("arrow", data_files=train_files, split="train")
        except Exception as e:
            print(f"Error loading local arrow files: {e}")
    
    if dataset is None:
        print("Could not find 'train' arrow files in local cache. Attempting to load any arrow file...")
        if arrow_files:
             # If no specific train file, maybe the dataset is small and in one file, or named differently.
             # We'll try to load the largest one or all.
             try:
                dataset = load_dataset("arrow", data_files=arrow_files, split="train")
             except Exception as e:
                print(f"Error loading arrow files: {e}")

    if dataset is None:
        print("Failed to load dataset from local cache. Please check the path.")
        return

    output_file = "data/dpo_hellaswag.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Processing {len(dataset)} examples...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for example in tqdm(dataset):
            # HellaSwag structure:
            # ctx: The context text
            # endings: List of 4 possible endings
            # label: Index of the correct ending (0-3)
            
            ctx = example['ctx']
            endings = example['endings']
            label = int(example['label'])
            
            # Construct the prompt
            # We might want to format it as a chat or just raw text depending on the model.
            # Since the SFT model is likely chat-tuned (Qwen), we should probably wrap it in a user prompt
            # or just provide the context if the model is expected to complete it.
            # However, DPO usually works on (prompt, chosen, rejected) tuples.
            # If the SFT model was trained with ChatML, we should use ChatML format for the prompt.
            
            # Let's assume standard Qwen chat format:
            # User: Complete the following text: {ctx}
            # Assistant: {ending}
            
            # OR, if it's a continuation task, maybe just the context.
            # But the user mentioned "Ending A vs Ending B", implying a choice or generation.
            # Given Qwen is a chat model, let's frame it as a completion task.
            
            prompt_text = f"Complete the description with the most plausible ending:\n\n{ctx}"
            
            chosen_text = endings[label]
            
            # Select a rejected ending
            possible_rejected_indices = [i for i in range(len(endings)) if i != label]
            rejected_index = random.choice(possible_rejected_indices)
            rejected_text = endings[rejected_index]
            
            # Format for DPO (using chat template structure implicitly or explicitly)
            # The DPO trainer usually handles chat templates if 'messages' column is used, 
            # or 'prompt', 'chosen', 'rejected' strings.
            # If we use strings, we should probably include the chat template formatting if the model expects it.
            # But `train_dpo_arc.py` seems to handle tokenization.
            # Let's look at `train_dpo_arc.py` again. It uses `DPOConfig`.
            # If we provide 'prompt', 'chosen', 'rejected', the trainer tokenizes them.
            
            # To be safe and consistent with SFT (which used ChatML), let's format the content as messages.
            # But standard DPO datasets often just have the text.
            # Let's use the 'prompt', 'chosen', 'rejected' fields but with ChatML formatting applied?
            # No, usually DPO trainer applies template if a tokenizer is provided.
            # Let's stick to simple strings for now, but maybe wrap in chat format if needed.
            # Actually, looking at `train_sft_curated.py`, it applies chat template.
            # So for DPO, if we want to align with SFT, we should probably use the same format.
            
            # Let's create a list of messages for prompt, and strings for responses?
            # TRL's DPOTrainer supports 'prompt', 'chosen', 'rejected' as strings.
            # If the tokenizer has a chat_template, TRL might use it if we pass a specific arg, 
            # but by default it just tokenizes the strings.
            
            # Let's construct the full strings including special tokens if we want to be precise,
            # OR rely on the trainer.
            # A common pattern for Chat DPO is:
            # prompt: "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n"
            # chosen: " ... <|im_end|>"
            # rejected: " ... <|im_end|>"
            
            # Let's try to be generic.
            # We will save the raw content and let the trainer/tokenizer handle it, 
            # OR we pre-format.
            # Given `train_sft_curated.py` manually sets the template, 
            # let's try to produce a dataset that has 'prompt', 'chosen', 'rejected' 
            # where 'prompt' is the user instruction.
            
            # Actually, let's look at `train_dpo_arc.py` dummy data:
            # "prompt": "User: ...", "chosen": "Assistant: ..."
            # It seems they put the role prefixes in the strings.
            
            # I will format it as:
            # prompt: <|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n
            # chosen: {chosen_text}<|im_end|>
            # rejected: {rejected_text}<|im_end|>
            
            # This ensures the model sees exactly what we want.
            
            prompt_str = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
            chosen_str = f"{chosen_text}<|im_end|>"
            rejected_str = f"{rejected_text}<|im_end|>"
            
            entry = {
                "prompt": prompt_str,
                "chosen": chosen_str,
                "rejected": rejected_str
            }
            
            f.write(json.dumps(entry) + "\n")

    print(f"Saved DPO dataset to {output_file}")

if __name__ == "__main__":
    prepare_hellaswag_dpo()
