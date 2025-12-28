# Qwen Fine-tuning & Evaluation Pipeline

This project contains a complete pipeline for SFT (Supervised Fine-Tuning), DPO (Direct Preference Optimization), and related evaluations for Qwen models (e.g., Qwen3-4B).

## 📋 Directory Structure

```text
.
├── benchmark_cache/        # Evaluation dataset cache
├── configs/                # Training configurations (YAML)
├── data/                   # Training data (JSONL)
├── eval_results/           # Evaluation results output
├── logs/                   # Training logs
├── output/                 # Model checkpoints and final model outputs
├── rl_output/              # RL/DPO training outputs
├── Qwen3-4B-Base/          # Base model files
├── *.py                    # Python training and data processing scripts
└── *.sh                    # Shell startup scripts
```

## 🚀 Quick Start

### 1. Environment Setup

Ensure Anaconda is installed and activate the environment:

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate qwen4b
```

### 2. Data Preparation

Before starting training, you need to prepare the training data.

*   **SFT Data**:
    ```bash
    python prepare_sft_data.py
    ```
    This will generate `data/sft_mixture.jsonl`.

*   **DPO Data**:
    ```bash
    python prepare_dpo_hellaswag.py
    ```
    This will generate `data/dpo_hellaswag.jsonl`.

### 3. Training Pipeline

#### Full Pipeline (SFT -> DPO)
Run the complete fine-tuning pipeline:
```bash
bash run_full_finetune_pipeline.sh
```

#### Run SFT Only (Supervised Fine-Tuning)
*   **Full SFT**:
    ```bash
    python train_sft_full.py
    ```
*   **Curated Data SFT**:
    ```bash
    bash run_sft_curated.sh
    ```

#### Run DPO Only (Direct Preference Optimization)
*   **Full DPO**:
    ```bash
    python train_dpo_full.py
    ```
*   **HellaSwag DPO**:
    ```bash
    python train_dpo_hellaswag.py
    ```

### 4. Evaluation

The project provides various evaluation scripts to test model performance at different stages.

*   **Evaluate Qwen Base Model**:
    ```bash
    bash run_eval_qwen.sh
    ```

*   **Evaluate RL/DPO LoRA Model (Debug)**:
    ```bash
    bash run_eval_rl_lora_debug.sh
    ```
    This script automatically finds the latest DPO model in `rl_output` for evaluation, including GSM8K and multiple-choice tasks.

*   **Other Evaluation Scripts**:
    *   `run_eval_full_debug.sh`: Evaluate full fine-tuned model
    *   `run_eval_rl_full_debug.sh`: Evaluate full RL model

## 📊 Viewing Results

Evaluation results are saved in the `eval_results/` directory, categorized by model type:
*   `baseline/`: Base model results
*   `SFT_Full/`, `SFT_LoRA/`: SFT model results
*   `RL_Full/`, `RL_LoRA/`: RL/DPO model results

Each directory typically contains:
*   `*_summary.txt`: Result summary
*   `*_detailed.json`: Detailed results

## 🛠️ Utility Scripts

*   `inspect_datasets.py`: Inspect dataset content
*   `eval_gsm8k.py`: Standalone GSM8K evaluation script
*   `eval_all_datasets.py`: Comprehensive evaluation script (ARC, MMLU, HellaSwag, etc.)
