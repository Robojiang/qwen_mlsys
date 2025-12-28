#!/bin/bash

# Stop on error
set -e

# Set GPU
export CUDA_VISIBLE_DEVICES=0
export NUMEXPR_MAX_THREADS=192

# Navigate to project directory
cd /mnt/afs/250010074/qwen

# Activate environment
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate qwen4b

# --- Configuration ---
# RL Full Finetune Model Path (DPO Full)
# This path contains full weights
LATEST_RL_FULL_DIR=$(ls -td rl_output/dpo_full_finetune_* | head -1)
MODEL_PATH="${LATEST_RL_FULL_DIR}/final_model"

# If final_model doesn't exist, try to find the latest checkpoint
if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️ 'final_model' not found in $LATEST_RL_FULL_DIR. Searching for latest checkpoint..."
    LATEST_CHECKPOINT=$(ls -td ${LATEST_RL_FULL_DIR}/checkpoint-* | head -1)
    if [ -d "$LATEST_CHECKPOINT" ]; then
        MODEL_PATH="$LATEST_CHECKPOINT"
        echo "👉 Using latest checkpoint: $MODEL_PATH"
    else
        echo "❌ Error: No model found in $LATEST_RL_FULL_DIR"
        exit 1
    fi
fi

# Output Directory
RESULTS_DIR="eval_results/RL_Full"
mkdir -p "$RESULTS_DIR"

# Debug Mode
export DEBUG="false"

echo "========================================================"
echo "🚀 STARTING RL FULL FINETUNE EVALUATION (DEBUG=$DEBUG)"
echo "🤖 Model Path: $MODEL_PATH"
echo "📂 Results Dir: $RESULTS_DIR"
echo "========================================================"

export MODEL_PATH="$MODEL_PATH"

# 1. Run Multiple Choice Evaluation
echo "----------------------------------------------------------------"
echo ">>> [1/2] Running Multiple Choice Evaluation (ARC, MMLU, etc.)"
echo "----------------------------------------------------------------"
python eval_all_datasets.py
mv results_mc_summary.txt "${RESULTS_DIR}/results_mc_summary_rl_full.txt"

# 2. Run GSM8K Evaluation
echo "----------------------------------------------------------------"
echo ">>> [2/2] Running GSM8K Evaluation"
echo "----------------------------------------------------------------"
python eval_gsm8k.py
mv results_gsm8k_summary.txt "${RESULTS_DIR}/results_gsm8k_summary_rl_full.txt"
mv results_gsm8k_detailed.json "${RESULTS_DIR}/results_gsm8k_detailed_rl_full.json"

echo "========================================================"
echo "✅ RL FULL FINETUNE EVALUATION COMPLETED"
echo "📊 Results saved to: $RESULTS_DIR"
echo "========================================================"
