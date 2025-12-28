import os
import torch
import glob
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import glob
import torch
import json

# ================= Configuration =================
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
DEBUG_SAMPLES = 5

# Base Model Path
BASE_MODEL_PATH = os.getenv("MODEL_PATH", "/mnt/afs/250010074/qwen/Qwen3-4B-Base")
CACHE_ROOT = "/mnt/afs/250010074/qwen/benchmark_cache/datasets"

# Dataset Folder Mappings (Folder Name -> Display Name)
# We will dynamically find sub-datasets for things like MMLU
DATASET_FOLDERS = {
    "allenai___ai2_arc": "ARC",
    "baber___piqa": "PIQA",
    "cais___mmlu": "MMLU",
    "Rowan___hellaswag": "HellaSwag",
    "tau___commonsense_qa": "CommonsenseQA",
    "winogrande": "Winogrande",
    # Generation tasks - skipped for now or handled differently
    # "gsm8k": "GSM8K", 
    # "openai___openai_humaneval": "HumanEval",
    # "google___if_eval": "IF_Eval"
}

# ================= Helper Functions =================

def find_arrow_file(dataset_folder_path, split_preference=["validation", "test", "train"]):
    """Finds the best arrow file for a dataset based on split preference."""
    # First, list all arrow files
    arrow_files = glob.glob(os.path.join(dataset_folder_path, "**", "*.arrow"), recursive=True)
    if not arrow_files:
        return None
    
    # Try to match preferences
    for split in split_preference:
        for f in arrow_files:
            if split in os.path.basename(f):
                return f
    
    # Fallback to the first one found
    return arrow_files[0]

def get_log_prob(model, tokenizer, context, candidate):
    input_text = context + candidate
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    ctx_len = len(tokenizer(context, add_special_tokens=False)['input_ids'])
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs.input_ids[..., 1:].contiguous()
    
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    target_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Only sum log probs for the candidate part
    if ctx_len < target_log_probs.size(1):
        candidate_log_prob = target_log_probs[0, ctx_len:].sum().item()
    else:
        candidate_log_prob = -9999.0
        
    return candidate_log_prob

# ================= Task Evaluators =================

def eval_arc(model, tokenizer, dataset):
    correct = 0
    total = 0
    
    # Map A,B,C,D,E to indices
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    # Sometimes labels are 1,2,3,4
    label_map_num = {'1': 0, '2': 1, '3': 2, '4': 3}

    for i, example in enumerate(tqdm(dataset, desc="Evaluating ARC")):
        if DEBUG and i >= DEBUG_SAMPLES: break
        
        question = example['question']
        choices = example['choices'] # {'text': [], 'label': []}
        answerKey = example['answerKey']
        
        # Determine label index
        if answerKey in label_map:
            label_idx = label_map[answerKey]
        elif answerKey in label_map_num:
            label_idx = label_map_num[answerKey]
        else:
            # Fallback or skip
            continue
            
        candidates = choices['text']
        if label_idx >= len(candidates): continue

        scores = []
        for cand in candidates:
            ctx = f"Question: {question}\nAnswer:"
            cand_text = f" {cand}"
            scores.append(get_log_prob(model, tokenizer, ctx, cand_text))
            
        pred = scores.index(max(scores))
        if pred == label_idx:
            correct += 1
        total += 1
        
    return correct / total if total > 0 else 0

def eval_piqa(model, tokenizer, dataset):
    correct = 0
    total = 0
    for i, example in enumerate(tqdm(dataset, desc="Evaluating PIQA")):
        if DEBUG and i >= DEBUG_SAMPLES: break
        
        goal = example['goal']
        sol1 = example['sol1']
        sol2 = example['sol2']
        label = example['label'] # 0 or 1
        
        ctx = f"Question: {goal}\nAnswer:"
        scores = []
        for cand in [sol1, sol2]:
            scores.append(get_log_prob(model, tokenizer, ctx, f" {cand}"))
            
        pred = scores.index(max(scores))
        if pred == label:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

def eval_mmlu(model, tokenizer, dataset):
    correct = 0
    total = 0
    options = ["A", "B", "C", "D"]
    
    for i, example in enumerate(tqdm(dataset, desc="Evaluating MMLU")):
        if DEBUG and i >= DEBUG_SAMPLES: break
        
        question = example['question']
        choices = example['choices']
        answer = example['answer'] # 0-3
        
        ctx = f"Question: {question}\nAnswer:"
        scores = []
        for cand in choices:
            scores.append(get_log_prob(model, tokenizer, ctx, f" {cand}"))
            
        pred = scores.index(max(scores))
        if pred == answer:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

def eval_hellaswag(model, tokenizer, dataset):
    correct = 0
    total = 0
    for i, example in enumerate(tqdm(dataset, desc="Evaluating HellaSwag")):
        if DEBUG and i >= DEBUG_SAMPLES: break
        
        ctx_text = example['ctx']
        endings = example['endings']
        label = int(example['label']) # '0'-'3' -> int
        
        # HellaSwag is completion. Context is the start, endings are completions.
        # We just score P(ending | ctx)
        scores = []
        for end in endings:
            # HellaSwag endings usually need a space prefix if not present, 
            # but often they are just continuations.
            # Let's assume simple concatenation with a space if needed.
            # Actually, let's just use the provided text.
            scores.append(get_log_prob(model, tokenizer, ctx_text, f" {end}"))
            
        pred = scores.index(max(scores))
        if pred == label:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

def eval_commonsense_qa(model, tokenizer, dataset):
    correct = 0
    total = 0
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    for i, example in enumerate(tqdm(dataset, desc="Evaluating CommonsenseQA")):
        if DEBUG and i >= DEBUG_SAMPLES: break
        
        question = example['question']
        choices = example['choices'] # {'text': [], 'label': []}
        answerKey = example['answerKey']
        
        if answerKey not in label_map: continue
        label_idx = label_map[answerKey]
        
        candidates = choices['text']
        if label_idx >= len(candidates): continue
        
        ctx = f"Question: {question}\nAnswer:"
        scores = []
        for cand in candidates:
            scores.append(get_log_prob(model, tokenizer, ctx, f" {cand}"))
            
        pred = scores.index(max(scores))
        if pred == label_idx:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

def eval_winogrande(model, tokenizer, dataset):
    correct = 0
    total = 0
    for i, example in enumerate(tqdm(dataset, desc="Evaluating Winogrande")):
        if DEBUG and i >= DEBUG_SAMPLES: break
        
        sentence = example['sentence']
        option1 = example['option1']
        option2 = example['option2']
        answer = example['answer'] # "1" or "2"
        
        if answer == '1': label = 0
        elif answer == '2': label = 1
        else: continue
        
        # Split sentence at '_'
        if '_' in sentence:
            parts = sentence.split('_')
            prefix = parts[0]
            suffix = parts[1] if len(parts) > 1 else ""
        else:
            # Fallback if no underscore (shouldn't happen in Winogrande usually)
            prefix = sentence
            suffix = ""
            
        scores = []
        for opt in [option1, option2]:
            # Construct full sentence
            # We treat 'prefix' as context, and 'opt + suffix' as candidate
            cand_text = opt + suffix
            scores.append(get_log_prob(model, tokenizer, prefix, cand_text))
            
        pred = scores.index(max(scores))
        if pred == label:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0


# Map task names to evaluator functions
EVALUATORS = {
    "ARC": eval_arc,
    "PIQA": eval_piqa,
    "MMLU": eval_mmlu,
    "HellaSwag": eval_hellaswag,
    "CommonsenseQA": eval_commonsense_qa,
    "Winogrande": eval_winogrande
}

def main():
    print(f"🚀 Starting Evaluation Script (DEBUG={DEBUG})")
    print(f"📂 Cache Root: {CACHE_ROOT}")
    print(f"🤖 Model: {BASE_MODEL_PATH}")
    
    # 1. Load Model
    print("\nLoading Model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Check if it's an adapter path (contains adapter_config.json)
        is_adapter = os.path.exists(os.path.join(BASE_MODEL_PATH, "adapter_config.json"))
        
        if is_adapter:
            print(f"Detected Adapter path. Loading base model Qwen3-4B-Base and merging adapter...")
            # Hardcoded base model path for now, or could be inferred/configured
            base_model_path = "/mnt/afs/250010074/qwen/Qwen3-4B-Base" 
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path, 
                trust_remote_code=True, 
                device_map="auto", 
                torch_dtype=torch.float16
            )
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, BASE_MODEL_PATH)
            model = model.merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_PATH, 
                trust_remote_code=True, 
                device_map="auto", 
                torch_dtype=torch.float16
            )
        model.eval()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    results = {}

    # 2. Iterate Datasets
    # Scan the cache directory for known folders
    for folder_name in os.listdir(CACHE_ROOT):
        full_folder_path = os.path.join(CACHE_ROOT, folder_name)
        if not os.path.isdir(full_folder_path): continue
        
        # Identify task type
        task_type = None
        display_name = folder_name
        
        if folder_name in DATASET_FOLDERS:
            task_type = DATASET_FOLDERS[folder_name]
            display_name = task_type
        
        # Special handling for MMLU (has subfolders)
        if task_type == "MMLU":
            print(f"\n🔍 Found MMLU container: {folder_name}")
            # Iterate subfolders
            for sub in os.listdir(full_folder_path):
                sub_path = os.path.join(full_folder_path, sub)
                if os.path.isdir(sub_path):
                    # Check if it has arrow files
                    arrow_file = find_arrow_file(sub_path)
                    if arrow_file:
                        sub_task_name = f"MMLU - {sub}"
                        print(f"   👉 Testing {sub_task_name}")
                        try:
                            ds = load_dataset("arrow", data_files={"data": arrow_file}, split="data")
                            acc = eval_mmlu(model, tokenizer, ds)
                            results[sub_task_name] = acc
                            print(f"   ✅ {sub_task_name}: {acc:.2%}")
                        except Exception as e:
                            print(f"   ❌ Error evaluating {sub_task_name}: {e}")
            continue

        # Standard handling for others
        if task_type and task_type in EVALUATORS:
            print(f"\n🔍 Found Dataset: {display_name} ({folder_name})")
            arrow_file = find_arrow_file(full_folder_path)
            if not arrow_file:
                print(f"   ⚠️ No arrow file found for {display_name}")
                continue
                
            print(f"   📂 Loading {arrow_file}")
            try:
                ds = load_dataset("arrow", data_files={"data": arrow_file}, split="data")
                eval_func = EVALUATORS[task_type]
                acc = eval_func(model, tokenizer, ds)
                results[display_name] = acc
                print(f"   ✅ {display_name}: {acc:.2%}")
            except Exception as e:
                print(f"   ❌ Error evaluating {display_name}: {e}")
        else:
            # Unknown or skipped dataset
            # print(f"Skipping unknown/unsupported folder: {folder_name}")
            pass

    print("\n" + "="*40)
    print("📊 Final Results Summary")
    print("="*40)
    
    # Save summary to a separate file
    with open("results_mc_summary.txt", "w") as f:
        f.write("Dataset,Accuracy\n")
        for k, v in results.items():
            print(f"{k:<30} | {v:.2%}")
            f.write(f"{k},{v:.2%}\n")
            
    print("="*40)
    print("📝 Summary saved to results_mc_summary.txt")

if __name__ == "__main__":
    main()
