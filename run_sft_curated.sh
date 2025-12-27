#!/bin/bash

# Stop on error
set -e

# Set GPU (Optional: adjust as needed)
export CUDA_VISIBLE_DEVICES=0
export NUMEXPR_MAX_THREADS=192
# Navigate to project directory
cd /mnt/afs/250010074/qwen

# If you need the specific path from the previous script:
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate qwen4b
# 1. Prepare Data
echo "Preparing SFT data mixture..."
python prepare_sft_data.py

# 2. Run Training
echo "Starting SFT training..."
# Using accelerate is recommended for multi-GPU, but python script directly works if device_map="auto" handles it (which it does, but DDP is better).
# However, the script uses trainer.train() which supports accelerate launch.
# Let's use python for simplicity as per existing scripts, or accelerate if available.
# Existing scripts seem to just run python.

python train_sft_curated.py
