#!/bin/bash

# Multi-GPU Training Script for Qwen3-4B on 4x H100

# 1. Prepare Data (if not already done)
if [ ! -f "data/sft_mixture.jsonl" ]; then
    echo "Preparing SFT data mixture..."
    python prepare_sft_data.py
fi

# 2. Run Training with Accelerate
# H100 80GB is very powerful.
# We use 4 GPUs.
# Per device batch size = 16 (configured in yaml)
# Gradient accumulation = 1
# Total Global Batch Size = 4 * 16 * 1 = 64

echo "Starting Multi-GPU SFT training..."

# Using accelerate launch to handle multi-gpu setup
# --multi_gpu: Enable multi-GPU
# --num_processes 4: Use 4 GPUs
# --mixed_precision bf16: Use BF16 (H100 supports it well)

accelerate launch \
    --multi_gpu \
    --num_processes 4 \
    --mixed_precision bf16 \
    train_sft_curated.py

