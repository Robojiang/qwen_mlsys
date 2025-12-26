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
SFT_MODEL_PATH = "/mnt/afs/250010074/qwen/output/sft_winogrande_checkpoints_xxxx/final_model" # 需替换为实际路径

# 3. RL (DPO) 模型路径
DPO_ADAPTER_PATH = "/mnt/afs/250010074/qwen/rl_output/dpo_winogrande_xxxx/final_model" # 需替换为实际路径

CACHE_ROOT = "/mnt/afs/250010074/qwen/benchmark_cache/datasets"

def find_arrow_file(dataset_name, split_name):
    pattern = os.path.join(CACHE_ROOT, f"*{dataset_name}*", "**", f"*{split_name}.arrow")
    files = glob.glob(pattern, recursive=True)
    if files:
        return files[0]
    return None

DATASET_CONFIGS = {
    "winogrande": {"name": "winogrande", "split": "validation"} 
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

def get_sentence_log_prob(model, tokenizer, sentence):
    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        loss = outputs.loss
        return -loss.item() * inputs.input_ids.size(1)

def evaluate_winogrande(model, tokenizer, dataset):
    correct = 0
    total = 0
    print("🔄 正在评估 Winogrande...")
    for example in tqdm(dataset):
        sentence = example['sentence']
        option1 = example['option1']
        option2 = example['option2']
        label = int(example['answer']) # "1" or "2" -> 1 or 2
        
        # 简单做法：替换 _
        sent1 = sentence.replace("_", option1)
        sent2 = sentence.replace("_", option2)
        
        score1 = get_sentence_log_prob(model, tokenizer, sent1)
        score2 = get_sentence_log_prob(model, tokenizer, sent2)
        
        pred = 1 if score1 > score2 else 2
        if pred == label:
            correct += 1
        total += 1
    return correct / total

def main():
    results = {}
    ds_wino = load_local_dataset("winogrande")
    if not ds_wino:
        return

    # 1. Base
    print(f"\n🚀 [1/3] Evaluating Base Model: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    acc_base = evaluate_winogrande(model, tokenizer, ds_wino)
    results["Base"] = acc_base
    print(f"✅ Base Accuracy: {acc_base:.2%}")
    del model
    torch.cuda.empty_cache()

    # 2. SFT
    print(f"\n🚀 [2/3] Evaluating SFT Model: {SFT_MODEL_PATH}")
    if os.path.exists(SFT_MODEL_PATH):
        model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
        model.eval()
        acc_sft = evaluate_winogrande(model, tokenizer, ds_wino)
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
        acc_rl = evaluate_winogrande(model, tokenizer, ds_wino)
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
