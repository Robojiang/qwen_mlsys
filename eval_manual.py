import os
import torch
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# ================= 配置 =================
MODEL_PATH = "/mnt/afs/250010074/qwen/Qwen3-4B-Base"
CACHE_ROOT = "/mnt/afs/250010074/qwen/benchmark_cache/datasets"

# 自动查找 Arrow 文件的函数
def find_arrow_file(dataset_name, split_name):
    # 搜索模式：datasets/namespace___dataset_name/**/dataset_name-split.arrow
    # 例如: datasets/baber___piqa/**/piqa-validation.arrow
    pattern = os.path.join(CACHE_ROOT, f"*{dataset_name}*", "**", f"*{split_name}.arrow")
    files = glob.glob(pattern, recursive=True)
    if files:
        return files[0] # 返回找到的第一个
    return None

# 数据集配置
# key: 任务名
# name: 数据集文件名部分 (如 piqa-validation.arrow 中的 piqa)
# split: split 名称
DATASET_CONFIGS = {
    "piqa": {"name": "piqa", "split": "validation"},  # test 集没有公开答案，只有 validation 集可用于本地评测和模型开发
    "arc": {"name": "ai2_arc", "split": "test"}, # ARC-Challenge 通常用 test 集
    "winogrande": {"name": "winogrande", "split": "validation"} # test 集没有公开答案，只有 validation 集可用于本地评测和模型开发
}

def load_local_dataset(task_name):
    config = DATASET_CONFIGS[task_name]
    arrow_file = find_arrow_file(config["name"], config["split"])
    
    if not arrow_file:
        print(f"❌ 未找到 {task_name} 的本地 Arrow 文件。")
        return None
    
    print(f"📂 加载本地文件: {arrow_file}")
    # 使用 arrow 格式直接加载
    ds = load_dataset("arrow", data_files={config["split"]: arrow_file}, split=config["split"])
    return ds

def evaluate_piqa(model, tokenizer, dataset):
    correct = 0
    total = 0
    
    print("🔄 正在评估 PIQA...")
    for example in tqdm(dataset):
        goal = example['goal']
        sol1 = example['sol1']
        sol2 = example['sol2']
        label = example['label'] # 0 or 1
        
        # 构建 prompt (Zero-shot 格式)
        prompt = f"Question: {goal}\nAnswer:"
        
        # 简单的 likelihood 比较
        # 计算 sol1 和 sol2 的 perplexity 或者直接生成
        # 这里使用生成法比较简单，但 likelihood 更准确。为了简单起见，这里沿用生成法或简单的包含判断
        # 但为了更接近 lm-eval，我们应该比较 log-likelihood。
        # 这里为了代码简洁，使用生成法 + 选项匹配 (类似于 test.py 的逻辑，但稍作改进)
        
        # 改进：使用 log-likelihood 比较 (更标准)
        # 构造两个完整的句子
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

def evaluate_arc(model, tokenizer, dataset):
    correct = 0
    total = 0
    
    print("🔄 正在评估 ARC-Challenge...")
    for example in tqdm(dataset):
        question = example['question']
        choices = example['choices'] # {'text': [...], 'label': [...]}
        answerKey = example['answerKey']
        
        scores = []
        labels = choices['label']
        texts = choices['text']
        
        ctx = f"Question: {question}\nAnswer:"
        
        for text in texts:
            cand = f" {text}"
            score = get_log_prob(model, tokenizer, ctx, cand)
            scores.append(score)
            
        # 找到得分最高的索引
        best_idx = scores.index(max(scores))
        pred_label = labels[best_idx]
        
        if pred_label == answerKey:
            correct += 1
        total += 1
        
    return correct / total

def evaluate_winogrande(model, tokenizer, dataset):
    correct = 0
    total = 0
    
    print("🔄 正在评估 Winogrande...")
    for example in tqdm(dataset):
        sentence = example['sentence']
        option1 = example['option1']
        option2 = example['option2']
        label = int(example['answer']) # "1" or "2" -> 1 or 2
        
        # Winogrande 需要替换 _ 为选项
        if "_" not in sentence:
            # 某些样本可能没有 _，作为 fallback
            ctx = sentence + " "
        else:
            ctx = sentence.split("_")[0] # 取 _ 前面的部分作为 context (简化版)
            # 更标准的做法是替换 _ 并计算整个句子的 perplexity
        
        # 简单做法：替换 _
        sent1 = sentence.replace("_", option1)
        sent2 = sentence.replace("_", option2)
        
        # 计算整个句子的 log-prob
        score1 = get_sentence_log_prob(model, tokenizer, sent1)
        score2 = get_sentence_log_prob(model, tokenizer, sent2)
        
        pred = 1 if score1 > score2 else 2
        
        if pred == label:
            correct += 1
        total += 1
        
    return correct / total

def get_log_prob(model, tokenizer, context, candidate):
    # 计算 P(candidate | context)
    input_text = context + candidate
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # 找到 candidate 的 token 范围
    ctx_len = len(tokenizer(context, add_special_tokens=False)['input_ids'])
    # 注意：这里简化处理，假设 tokenizer 不会因为拼接而改变 tokenization
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    # Shift logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs.input_ids[..., 1:].contiguous()
    
    # 只计算 candidate 部分的 loss
    # candidate 对应的 labels 是 shift_labels[ctx_len-1:] (大约)
    # 为了准确，我们计算整个序列的 loss，然后减去 context 的 loss，或者只取后半部分
    # 这里使用简单的 gather 方法
    
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    
    # Gather log probs of the correct tokens
    target_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Sum log probs for the candidate tokens
    # 假设 candidate 从 ctx_len 开始 (这取决于 tokenizer 是否添加 BOS)
    # 这是一个近似实现
    candidate_log_prob = target_log_probs[0, ctx_len:].sum().item()
    
    return candidate_log_prob

def get_sentence_log_prob(model, tokenizer, sentence):
    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        # loss 是负对数似然
        loss = outputs.loss
        # log_prob = -loss * seq_len
        return -loss.item() * inputs.input_ids.size(1)

def main():
    print(f"🚀 加载模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
    
    results = {}
    
    # PIQA
    ds_piqa = load_local_dataset("piqa")
    if ds_piqa:
        acc = evaluate_piqa(model, tokenizer, ds_piqa)
        results["PIQA"] = acc
        print(f"✅ PIQA Accuracy: {acc:.2%}")
        
    # ARC
    ds_arc = load_local_dataset("arc")
    if ds_arc:
        acc = evaluate_arc(model, tokenizer, ds_arc)
        results["ARC-Challenge"] = acc
        print(f"✅ ARC-Challenge Accuracy: {acc:.2%}")
        
    # Winogrande
    ds_wino = load_local_dataset("winogrande")
    if ds_wino:
        acc = evaluate_winogrande(model, tokenizer, ds_wino)
        results["Winogrande"] = acc
        print(f"✅ Winogrande Accuracy: {acc:.2%}")
        
    print("\n📊 最终结果:")
    for k, v in results.items():
        print(f"{k}: {v:.2%}")

if __name__ == "__main__":
    main()
