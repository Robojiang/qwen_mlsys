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

# --- Stage 1: SFT ---
echo "=== Stage 1: SFT Training ==="

# 1. Prepare SFT Data
if [ ! -f "data/sft_mixture.jsonl" ]; then
    echo "Preparing SFT data mixture..."
    python prepare_sft_data.py
else
    echo "SFT data already exists."
fi

# 2. Run SFT Training
echo "Starting SFT training..."
python train_sft_curated.py

# 3. Find the latest SFT output directory
# Assuming the output structure is output/sft_curated_mixture_YYYY-MM-DD_HH-MM-SS
LATEST_SFT_DIR=$(ls -td output/sft_curated_mixture_* | head -1)
SFT_MODEL_PATH="${LATEST_SFT_DIR}/final_model"

if [ -d "$SFT_MODEL_PATH" ]; then
    echo "SFT completed. Model saved at: $SFT_MODEL_PATH"
else
    echo "Error: SFT model path not found at $SFT_MODEL_PATH"
    exit 1
fi

# --- Stage 2: DPO ---
echo "=== Stage 2: DPO Training (HellaSwag) ==="

# 1. Prepare DPO Data
if [ ! -f "data/dpo_hellaswag.jsonl" ]; then
    echo "Preparing HellaSwag DPO data..."
    python prepare_dpo_hellaswag.py
else
    echo "DPO data already exists."
fi

# 2. Run DPO Training
# We dynamically pass the SFT adapter path we just found
echo "Starting DPO training using SFT adapter: $SFT_MODEL_PATH"
python train_dpo_hellaswag.py model.sft_adapter_path="$SFT_MODEL_PATH"

echo "========================================================"
echo "✅ FULL PIPELINE COMPLETED SUCCESSFULLY"
echo "========================================================"
