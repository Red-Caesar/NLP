import os
from pathlib import Path
import pandas as pd
import re

LAB_DIR = Path(__file__).parent.parent.resolve()
MODELS_DIR = LAB_DIR / "models" / "quantized"
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

def extract_quantization_type(folder_name):
    if "FP8" in folder_name or "fp8" in folder_name:
        return "fp8"
    elif "AWQ" in folder_name or "awq" in folder_name:
        return "awq"
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

def get_safetensors_size(model_path):
    total_size = 0
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.endswith('.safetensors'):
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
    return total_size

def bytes_to_gb(bytes_size):
    return bytes_size / (1024 ** 3)

def get_original_model_size():
    model_cache_name = "models--Qwen--Qwen3-8B"
    model_cache_path = HF_CACHE_DIR / model_cache_name
    
    if model_cache_path.exists():
        size_bytes = get_safetensors_size(model_cache_path)
        return size_bytes
    return None

def calculate_model_sizes():
    data = []
    
    original_size = get_original_model_size()
    if original_size:
        size_gb = bytes_to_gb(original_size)
        data.append({
            'model_quantization': 'original',
            'dataset': None,
            'seq_len': None,
            'calibration_samples': None,
            'size_gb': round(size_gb, 2),
            'size_bytes': original_size,
        })
    else:
        print(f"Original model not found in cache: {HF_CACHE_DIR}")
    
    if not MODELS_DIR.exists():
        print(f"Models directory not found: {MODELS_DIR}")
        return pd.DataFrame(data)
    
    for model_folder in os.listdir(MODELS_DIR):
        model_path = MODELS_DIR / model_folder
        
        if model_path.is_dir():
            quantization_type = extract_quantization_type(model_folder)
            dataset = extract_dataset(model_folder)
            seq_len, calibration_samples = extract_seq_len_and_calibration(model_folder)
            
            size_bytes = get_safetensors_size(model_path)
            size_gb = bytes_to_gb(size_bytes)
            
            data.append({
                'model_quantization': quantization_type,
                'dataset': dataset,
                'seq_len': seq_len,
                'calibration_samples': calibration_samples,
                'size_gb': round(size_gb, 2),
                'size_bytes': size_bytes,
            })
    
    df = pd.DataFrame(data)
    df = df.sort_values(by=['size_bytes'])
    
    return df

if __name__ == "__main__":
    df = calculate_model_sizes()
    print(df.to_string(index=False))
    
    csv_path = LAB_DIR / "model_sizes_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

