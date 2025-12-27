import os
import glob
from datasets import load_dataset

CACHE_ROOT = "/mnt/afs/250010074/qwen/benchmark_cache/datasets"

def find_arrow_file(dataset_folder):
    pattern = os.path.join(dataset_folder, "**", "*.arrow")
    files = glob.glob(pattern, recursive=True)
    if files:
        return files[0]
    return None

def inspect_dataset(name, folder_name):
    path = os.path.join(CACHE_ROOT, folder_name)
    arrow_file = find_arrow_file(path)
    if not arrow_file:
        print(f"Skipping {name}: No arrow file found in {path}")
        return

    try:
        ds = load_dataset("arrow", data_files={"data": arrow_file}, split="data")
        print(f"--- {name} ---")
        print(f"Columns: {ds.column_names}")
        print(f"Example: {ds[0]}")
    except Exception as e:
        print(f"Error loading {name}: {e}")

datasets_to_check = {
    "ARC": "allenai___ai2_arc",
    "MMLU": "cais___mmlu",
    "HellaSwag": "Rowan___hellaswag",
    "CommonsenseQA": "tau___commonsense_qa",
    "Winogrande": "winogrande",
    "GSM8K": "gsm8k"
}

for name, folder in datasets_to_check.items():
    inspect_dataset(name, folder)
