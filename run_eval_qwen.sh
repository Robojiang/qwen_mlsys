#!/bin/bash

# Stop on error
set -e

# Set GPU (Optional: adjust as needed)
export CUDA_VISIBLE_DEVICES=0
export NUMEXPR_MAX_THREADS=192
# Navigate to project directory
cd /mnt/afs/250010074/qwen

# Activate Conda Environment
# Adjust the path to conda.sh if it differs on your server, 
# but usually 'source activate' or 'conda activate' works if initialized.
# If you need the specific path from the previous script:
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate qwen4b

echo "🚀 Starting Qwen3-4B Evaluation Pipeline..."
echo "Date: $(date)"

# Create logs directory
mkdir -p logs

# 1. Run Multiple Choice Tasks (ARC, PIQA, MMLU, etc.)
echo "----------------------------------------------------------------"
echo ">>> [1/2] Running Multiple Choice Evaluation (ARC, MMLU, etc.)"
echo "----------------------------------------------------------------"
python eval_all_datasets.py > logs/eval_mc_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "✅ Multiple Choice Evaluation Complete."
echo "📊 Summary:"
cat results_mc_summary.txt

# 2. Run Generative Tasks (GSM8K, HumanEval)
echo "----------------------------------------------------------------"
echo ">>> [2/2] Running Generative Evaluation (GSM8K, HumanEval)"
echo "----------------------------------------------------------------"
python eval_generative.py > logs/eval_gen_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "✅ Generative Evaluation Complete."
echo "📊 Summary:"
cat results_gen_summary.txt

echo "----------------------------------------------------------------"
echo "🎉 All evaluations finished!"
echo "📝 Final concise results are saved in:"
echo "   - results_mc_summary.txt"
echo "   - results_gen_summary.txt"

