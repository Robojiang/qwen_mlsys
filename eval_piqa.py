import os
import torch
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

# ================= 配置 =================
# 1. Base 模型路径
BASE_MODEL_PATH = "/mnt/afs/250010074/qwen/Qwen3-4B-Base"

# 2. SFT 模型路径 (全量或 LoRA)
# 假设 SFT 是 LoRA，且已经 merge 成了 final_model
SFT_MODEL_PATH = "/mnt/afs/250010074/qwen/output/sft_piqa_checkpoints_xxxx/final_model" # 需替换为实际路径

# 3. RL (DPO) 模型路径
# DPO 是基于 SFT 训练的 LoRA Adapter
DPO_ADAPTER_PATH = "/mnt/afs/250010074/qwen/rl_output/dpo_piqa_xxxx/final_model" # 需替换为实际路径

CACHE_ROOT = "/mnt/afs/250010074/qwen/benchmark_cache/datasets"

def find_arrow_file(dataset_name, split_name):
    pattern = os.path.join(CACHE_ROOT, f"*{dataset_name}*", "**", f"*{split_name}.arrow")
    files = glob.glob(pattern, recursive=True)
    if files:
        return files[0]
    return None

DATASET_CONFIGS = {
    "piqa": {"name": "piqa", "split": "validation"}, 
}

def load_local_dataset(task_name):
    config = DATASET_CONFIGS[task_name]
    arrow_file = find_arrow_file(config["name"], config["split"])
    if not arrow_file:
        print(f"❌ 未找到 {task_name} 的本地 Arrow 文件。")
        return None
    print(f"📂 加载本地文件: {arrow_file}")
    ds = load_dataset("arrow", data_files={config["split"]: arrow_file}, split=config["split"])
    return ds

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
    if ctx_len < target_log_probs.size(1):
        candidate_log_prob = target_log_probs[0, ctx_len:].sum().item()
    else:
        candidate_log_prob = -9999.0
    return candidate_log_prob

def evaluate_piqa(model, tokenizer, dataset):
    correct = 0
    total = 0
    print("🔄 正在评估 PIQA...")
    for example in tqdm(dataset):
        goal = example['goal']
        sol1 = example['sol1']
        sol2 = example['sol2']
        label = example['label']
        
        ctx = f"Question: {goal}\nAnswer:"
        cand1 = f" {sol1}"
        cand2 = f" {sol2}"
        
        score1 = get_log_prob(model, tokenizer, ctx, cand1)
        score2 = get_log_prob(model, tokenizer, ctx, cand2)
        
        pred = 0 if score1 > score2 else 1
        if pred == label:
            correct += 1
        total += 1
    return correct / total

def main():
    results = {}
    ds_piqa = load_local_dataset("piqa")
    if not ds_piqa:
        return

    # 1. Base
    print(f"\n🚀 [1/3] Evaluating Base Model: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    acc_base = evaluate_piqa(model, tokenizer, ds_piqa)
    results["Base"] = acc_base
    print(f"✅ Base Accuracy: {acc_base:.2%}")
    del model
    torch.cuda.empty_cache()

    # 2. SFT
    print(f"\n🚀 [2/3] Evaluating SFT Model: {SFT_MODEL_PATH}")
    if os.path.exists(SFT_MODEL_PATH):
        model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
        model.eval()
        acc_sft = evaluate_piqa(model, tokenizer, ds_piqa)
        results["SFT"] = acc_sft
        print(f"✅ SFT Accuracy: {acc_sft:.2%}")
        del model
        torch.cuda.empty_cache()
    else:
        print(f"⚠️ SFT Model path not found: {SFT_MODEL_PATH}")

    # 3. RL (DPO)
    print(f"\n🚀 [3/3] Evaluating RL (DPO) Model")
    if os.path.exists(SFT_MODEL_PATH) and os.path.exists(DPO_ADAPTER_PATH):
        model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(model, DPO_ADAPTER_PATH)
        print("🔄 Merging LoRA weights...")
        model = model.merge_and_unload()
        model.eval()
        acc_rl = evaluate_piqa(model, tokenizer, ds_piqa)
        results["RL (DPO)"] = acc_rl
        print(f"✅ RL (DPO) Accuracy: {acc_rl:.2%}")
    else:
        print(f"⚠️ Model paths not found for RL evaluation.")

    print("\n📊 最终对比结果:")
    print(f"{'Stage':<15} | {'Accuracy':<10}")
    print("-" * 30)
    for k, v in results.items():
        print(f"{k:<15} | {v:.2%}")

if __name__ == "__main__":
    main()
