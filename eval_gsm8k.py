import os
import glob
import torch
import json
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ================= Configuration =================
# Debug mode: only run first 5 samples
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
DEBUG_SAMPLES = 5

# Model Path
BASE_MODEL_PATH = os.getenv("MODEL_PATH", "/mnt/afs/250010074/qwen/Qwen3-4B-Base")
CACHE_ROOT = "/mnt/afs/250010074/qwen/benchmark_cache/datasets"

# Dataset Folder
DATASET_FOLDER_NAME = "gsm8k"
DISPLAY_NAME = "GSM8K"

# ================= Helper Functions =================

def find_arrow_file(dataset_folder_path):
    """Finds the .arrow file in the dataset folder."""
    arrow_files = glob.glob(os.path.join(dataset_folder_path, "**", "*.arrow"), recursive=True)
    if not arrow_files:
        return None
    return arrow_files[0]

def generate_response(model, tokenizer, prompt, max_new_tokens=256):
    """Generates response using greedy decoding."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False, # Greedy for deterministic results
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode, skipping the prompt
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text

# ================= GSM8K Evaluator =================

def extract_answer_gsm8k(text):
    """
    Extracts the answer from GSM8K output.
    Expected format: 'The answer is 42' or '#### 42'.
    """
    # 1. Try extracting after ####
    if "####" in text:
        text = text.split("####")[-1]
    
    # 2. Remove commas (1,000 -> 1000)
    text_no_comma = text.replace(",", "")
    
    # 3. Extract the last number
    numbers = re.findall(r'-?\d+\.?\d*', text_no_comma)
    if numbers:
        return numbers[-1]
    return None

def eval_gsm8k(model, tokenizer, dataset):
    print(f"\n🧮 Evaluating {DISPLAY_NAME} (Math Reasoning)...")
    results = []
    correct = 0
    total = 0
    
    for i, example in enumerate(tqdm(dataset)):
        if DEBUG and i >= DEBUG_SAMPLES: break
        
        question = example['question']
        target = example['answer']
        
        # Construct Prompt
        prompt = f"Question: {question}\nLet's think step by step.\nAnswer:"
        
        response = generate_response(model, tokenizer, prompt, max_new_tokens=256)
        
        # Scoring
        pred_ans = extract_answer_gsm8k(response)
        target_ans = extract_answer_gsm8k(target)
        
        is_correct = False
        if pred_ans is not None and target_ans is not None:
            try:
                # Float comparison with tolerance
                if abs(float(pred_ans) - float(target_ans)) < 1e-6:
                    is_correct = True
            except:
                # String comparison
                if str(pred_ans).strip() == str(target_ans).strip():
                    is_correct = True
        
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "question": question,
            "target": target,
            "prediction": response,
            "extracted_pred": pred_ans,
            "extracted_target": target_ans,
            "correct": is_correct
        })
        
    accuracy = correct / total if total > 0 else 0
    return results, accuracy

# ================= Main =================

def main():
    print(f"🚀 Starting GSM8K Evaluation")
    print(f"🔧 DEBUG Mode: {DEBUG}")
    print(f"🤖 Model: {BASE_MODEL_PATH}")
    
    # 1. Load Model
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Check for Adapter
        is_adapter = os.path.exists(os.path.join(BASE_MODEL_PATH, "adapter_config.json"))
        
        if is_adapter:
            print(f"📦 Detected Adapter. Loading base model Qwen3-4B-Base and merging...")
            base_model_path = "/mnt/afs/250010074/qwen/Qwen3-4B-Base"
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, BASE_MODEL_PATH)
            model = model.merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_PATH,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Load Dataset
    full_path = os.path.join(CACHE_ROOT, DATASET_FOLDER_NAME)
    if not os.path.exists(full_path):
        print(f"❌ Dataset folder not found: {full_path}")
        return
        
    arrow_file = find_arrow_file(full_path)
    if not arrow_file:
        print(f"❌ No arrow file found in {full_path}")
        return
        
    print(f"\n📂 Loading {DISPLAY_NAME} from {arrow_file}...")
    try:
        ds = load_dataset("arrow", data_files={"data": arrow_file}, split="data")
        
        # 3. Run Evaluation
        task_results, score = eval_gsm8k(model, tokenizer, ds)
        
        print(f"\n✅ {DISPLAY_NAME} Accuracy: {score:.2%}")
        
        # 4. Save Results
        output_file = "results_gsm8k_detailed.json"
        with open(output_file, "w") as f:
            json.dump(task_results, f, indent=2)
            
        summary_file = "results_gsm8k_summary.txt"
        with open(summary_file, "w") as f:
            f.write("Dataset,Metric,Value\n")
            f.write(f"{DISPLAY_NAME},Accuracy,{score:.2%}\n")
            
        print("\n" + "="*40)
        print(f"📊 Final Result: {score:.2%}")
        print("="*40)
        print(f"📝 Detailed results saved to {output_file}")
        print(f"📝 Summary saved to {summary_file}")
        
    except Exception as e:
        print(f"❌ Error evaluating {DISPLAY_NAME}: {e}")

if __name__ == "__main__":
    main()
