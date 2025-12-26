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
# 如果 SFT 是全量微调，直接写路径；如果是 LoRA，写 Adapter 路径
# 这里假设 SFT 是 LoRA，且已经 merge 成了 final_model (或者直接加载 final_model)
# 根据之前的对话，SFT 的 final_model 是一个完整的模型目录
SFT_MODEL_PATH = "/mnt/afs/250010074/qwen/output/sft_arc_checkpoints_2025-12-23_17-42-41/final_model"

# 3. RL (DPO) 模型路径
# DPO 是基于 SFT 训练的 LoRA Adapter
DPO_ADAPTER_PATH = "/mnt/afs/250010074/qwen/rl_output/dpo_arc_2025-12-23_19-28-42/final_model"

CACHE_ROOT = "/mnt/afs/250010074/qwen/benchmark_cache/datasets"

# 自动查找 Arrow 文件的函数
def find_arrow_file(dataset_name, split_name):
    pattern = os.path.join(CACHE_ROOT, f"*{dataset_name}*", "**", f"*{split_name}.arrow")
    files = glob.glob(pattern, recursive=True)
    if files:
        return files[0]
    return None

# 数据集配置
DATASET_CONFIGS = {
    "arc": {"name": "ai2_arc", "split": "test"}, 
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
    
    # 简单的 mask 处理，只计算 candidate 部分
    # 注意：这里假设 tokenizer 行为一致，实际可能需要更严谨的处理
    if ctx_len < target_log_probs.size(1):
        candidate_log_prob = target_log_probs[0, ctx_len:].sum().item()
    else:
        candidate_log_prob = -9999.0 # 异常情况
    
    return candidate_log_prob

def evaluate_arc(model, tokenizer, dataset):
    correct = 0
    total = 0
    
    print("🔄 正在评估 ARC-Challenge...")
    for example in tqdm(dataset):
        question = example['question']
        choices = example['choices']
        answerKey = example['answerKey']
        
        scores = []
        labels = choices['label']
        texts = choices['text']
        
        # 构造 Prompt，保持和训练时一致的风格 (虽然这里是评测，但尽量保持一致)
        # 训练时: User: {question}\nChoices:\n{options}\nAnswer:\nAssistant: {answer}
        # 评测时: 给定 Context，看哪个 Option 的概率大
        # 这里使用标准的 Zero-shot 格式: Question: ... Answer: ...
        
        ctx = f"Question: {question}\nAnswer:"
        
        for text in texts:
            cand = f" {text}"
            score = get_log_prob(model, tokenizer, ctx, cand)
            scores.append(score)
            
        best_idx = scores.index(max(scores))
        pred_label = labels[best_idx]
        
        if pred_label == answerKey:
            correct += 1
        total += 1
        
    return correct / total

def main():
    results = {}
    ds_arc = load_local_dataset("arc")
    if not ds_arc:
        return

    # 1. 评估 Base 模型
    print(f"\n🚀 [1/3] Evaluating Base Model: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        trust_remote_code=True, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    model.eval()
    acc_base = evaluate_arc(model, tokenizer, ds_arc)
    results["Base"] = acc_base
    print(f"✅ Base Accuracy: {acc_base:.2%}")
    
    # 释放显存
    del model
    torch.cuda.empty_cache()

    # 2. 评估 SFT 模型
    print(f"\n🚀 [2/3] Evaluating SFT Model: {SFT_MODEL_PATH}")
    # 注意：如果 SFT 是 LoRA Adapter，需要先加载 Base 再加载 Adapter。
    # 这里假设 SFT_MODEL_PATH 是已经 merge 好的完整模型 (根据之前的 final_model 逻辑)
    model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL_PATH, 
        trust_remote_code=True, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    model.eval()
    acc_sft = evaluate_arc(model, tokenizer, ds_arc)
    results["SFT"] = acc_sft
    print(f"✅ SFT Accuracy: {acc_sft:.2%}")

    # 3. 评估 RL (DPO) 模型
    print(f"\n🚀 [3/3] Evaluating RL (DPO) Model")
    print(f"Base for RL: {SFT_MODEL_PATH}")
    print(f"Adapter for RL: {DPO_ADAPTER_PATH}")
    
    # DPO 是基于 SFT 模型的 LoRA，所以基座是 SFT 模型
    # 我们复用刚才加载的 SFT model，直接加载 Adapter
    # 如果刚才释放了，需要重新加载 SFT 模型
    # model = AutoModelForCausalLM.from_pretrained(SFT_MODEL_PATH, ...) 
    
    model = PeftModel.from_pretrained(model, DPO_ADAPTER_PATH)
    print("🔄 Merging LoRA weights...")
    model = model.merge_and_unload()
    model.eval()
    
    acc_rl = evaluate_arc(model, tokenizer, ds_arc)
    results["RL (DPO)"] = acc_rl
    print(f"✅ RL (DPO) Accuracy: {acc_rl:.2%}")
        
    print("\n📊 最终对比结果:")
    print(f"{'Stage':<15} | {'Accuracy':<10}")
    print("-" * 30)
    for k, v in results.items():
        print(f"{k:<15} | {v:.2%}")

if __name__ == "__main__":
    main()
