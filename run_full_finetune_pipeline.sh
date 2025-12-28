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

echo "========================================================"
echo "🚀 STARTING FULL FINE-TUNING PIPELINE (SFT -> DPO)"
echo "========================================================"

# # --- Stage 1: Full SFT ---
# echo "=== Stage 1: Full SFT Training ==="

# # 1. Prepare SFT Data (if needed)
# if [ ! -f "data/sft_mixture.jsonl" ]; then
#     echo "Preparing SFT data mixture..."
#     python prepare_sft_data.py
# fi

# # 2. Run Full SFT Training
# echo "Starting Full SFT training..."
# python train_sft_full.py

# 3. Find the latest SFT output directory
LATEST_SFT_DIR=$(ls -td output/sft_full_finetune_* | head -1)
SFT_MODEL_PATH="${LATEST_SFT_DIR}/final_model"

if [ -d "$SFT_MODEL_PATH" ]; then
    echo "✅ Full SFT completed. Model found at: $SFT_MODEL_PATH"
else
    echo "❌ Error: SFT model path not found at $SFT_MODEL_PATH"
    exit 1
fi

# --- Stage 2: Full DPO ---
echo "=== Stage 2: Full DPO Training (HellaSwag) ==="

# 1. Prepare DPO Data
if [ ! -f "data/dpo_hellaswag.jsonl" ]; then
    echo "Preparing HellaSwag DPO data..."
    python prepare_dpo_hellaswag.py
fi

# 2. Run Full DPO Training
# We pass the SFT model path as the base model for DPO
echo "Starting Full DPO training using SFT model: $SFT_MODEL_PATH"
python train_dpo_full.py model.model_name_or_path="$SFT_MODEL_PATH"

echo "========================================================"
echo "✅ FULL FINE-TUNING PIPELINE COMPLETED SUCCESSFULLY"
echo "========================================================"
