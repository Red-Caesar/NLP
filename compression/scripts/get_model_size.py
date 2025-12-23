import os
import json
from pathlib import Path
import pandas as pd
import re

LAB_DIR = Path(__file__).parent.parent.resolve()
QUANTIZED_DIR = LAB_DIR / "models" / "quantized"
FINETUNED_DIR = LAB_DIR / "models" / "finetuned"
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

def extract_quantization_type(folder_name):
    if "FP8" in folder_name:
        return "fp8"
    elif "AWQ" in folder_name:
        return "awq"
    elif "INT4" in folder_name:
        return "int4"
    return "original"

def extract_dataset(folder_name):
    if "mmlu-dataset" in folder_name:
        return "mmlu"
    elif "common-dataset" in folder_name:
        return "common"
    return None

def extract_seq_len_and_calibration(folder_name):
    match = re.search(r'-(\d+)-(\d+)$', folder_name)
    if match:
        seq_len = int(match.group(1))
        calibration_samples = int(match.group(2))
        return seq_len, calibration_samples
    return None, None

def get_model_size_from_index(model_path):
    index_files = [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "model.index.json"
    ]
    
    for index_file in index_files:
        index_path = model_path / index_file
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                    if "metadata" in data and "total_size" in data["metadata"]:
                        return data["metadata"]["total_size"]
            except Exception as e:
                print(f"Error reading {index_path}: {e}")
    return None


def bytes_to_gb(bytes_size):
    return bytes_size / (1024 ** 3)

def get_original_model_size():
    model_cache_name = "models--Qwen--Qwen3-8B"
    model_cache_path = HF_CACHE_DIR / model_cache_name
    
    if not model_cache_path.exists():
        return None
    
    snapshots_dir = model_cache_path / "snapshots"
    
    for snapshot_dir in snapshots_dir.iterdir():
        if snapshot_dir.is_dir():
            size_bytes = get_model_size_from_index(snapshot_dir)
            if size_bytes:
                return size_bytes
    
    return None

def calculate_model_sizes():
    data = []
    
    original_size = get_original_model_size()
    if original_size:
        size_gb = bytes_to_gb(original_size)
        data.append({
            "model_name": "Qwen3-8B",
            "model_quantization": "original",
            "dataset": None,
            "seq_len": None,
            "calibration_samples": None,
            "size_gb": round(size_gb, 2),
            "size_bytes": original_size,
        })
    else:
        print(f"Original model not found in cache: {HF_CACHE_DIR}")
    
    for models_dir in [QUANTIZED_DIR, FINETUNED_DIR]:
        if not models_dir.exists():
            print(f"Models directory not found: {models_dir}")
            continue
        
        for model_folder in os.listdir(models_dir):
            model_path = models_dir / model_folder
            
            if model_path.is_dir() and model_folder != "LoRA":
                quantization_type = extract_quantization_type(model_folder)
                dataset = extract_dataset(model_folder)
                seq_len, calibration_samples = extract_seq_len_and_calibration(model_folder)
                
                size_bytes = get_model_size_from_index(model_path)
                
                if size_bytes:
                    size_gb = bytes_to_gb(size_bytes)
                    
                    data.append({
                        "model_name": model_folder,
                        "model_quantization": quantization_type,
                        "dataset": dataset,
                        "seq_len": seq_len,
                        "calibration_samples": calibration_samples,
                        "size_gb": round(size_gb, 2),
                        "size_bytes": size_bytes,
                    })
                else:
                    print(f"Warning: Could not determine size for {model_folder}")
    
    df = pd.DataFrame(data)
    df = df.sort_values(by=["size_bytes"])
    
    return df

if __name__ == "__main__":
    df = calculate_model_sizes()
    print(df.to_string(index=False))
    
    csv_path = LAB_DIR / "model_sizes_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

