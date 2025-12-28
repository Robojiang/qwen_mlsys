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
# LoRA Model Path (SFT Curated Mixture)
# This path contains adapter_config.json, so the eval scripts will automatically load base model + adapter
LATEST_LORA_DIR=$(ls -td output/sft_curated_mixture_* | head -1)
MODEL_PATH="${LATEST_LORA_DIR}/final_model"

# Output Directory
RESULTS_DIR="eval_results/SFT_LoRA"
mkdir -p "$RESULTS_DIR"

# Debug Mode (Set to "true" for 1-sample test, "false" for full run)
export DEBUG="false"

echo "========================================================"
echo "🚀 STARTING LoRA EVALUATION (DEBUG=$DEBUG)"
echo "🤖 Model Path: $MODEL_PATH"
echo "📂 Results Dir: $RESULTS_DIR"
echo "========================================================"

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model path not found at $MODEL_PATH"
    exit 1
fi

export MODEL_PATH="$MODEL_PATH"

# 1. Run Multiple Choice Evaluation
echo "----------------------------------------------------------------"
echo ">>> [1/2] Running Multiple Choice Evaluation (ARC, MMLU, etc.)"
echo "----------------------------------------------------------------"
python eval_all_datasets.py
mv results_mc_summary.txt "${RESULTS_DIR}/results_mc_summary_lora.txt"

# 2. Run GSM8K Evaluation
echo "----------------------------------------------------------------"
echo ">>> [2/2] Running GSM8K Evaluation"
echo "----------------------------------------------------------------"
python eval_gsm8k.py
mv results_gsm8k_summary.txt "${RESULTS_DIR}/results_gsm8k_summary_lora.txt"
mv results_gsm8k_detailed.json "${RESULTS_DIR}/results_gsm8k_detailed_lora.json"

echo "========================================================"
echo "✅ LoRA EVALUATION COMPLETED"
echo "📊 Results saved to: $RESULTS_DIR"
echo "========================================================"
