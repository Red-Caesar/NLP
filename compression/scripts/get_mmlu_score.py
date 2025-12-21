import json
import os
from pathlib import Path
import pandas as pd
import re

LAB_DIR = Path(__file__).parent.parent.resolve()
EVAL_RESULTS_DIR = LAB_DIR / "eval_results"

def extract_quantization_type(folder_name):
    if "FP8" in folder_name or "fp8" in folder_name:
        return "fp8"
    elif "AWQ" in folder_name or "awq" in folder_name:
        return "awq"
    return "original"

def extract_date_from_filename(filename):
    match = re.search(r'results_(\d{4}-\d{2}-\d{2})T', filename)
    if match:
        return match.group(1)
    return "unknown"

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

def parse_eval_results():
    data = []
    
    for root, dirs, files in os.walk(EVAL_RESULTS_DIR):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(root)
                
                try:
                    with open(file_path, 'r') as f:
                        content = json.load(f)
                    
                    quantization_type = extract_quantization_type(folder_name)
                    date = extract_date_from_filename(file)
                    limit = content.get('config', {}).get('limit', None)
                    mmlu_score = content.get('results', {}).get('mmlu', {}).get('acc,none', None)
                    dataset = extract_dataset(folder_name)
                    seq_len, calibration_samples = extract_seq_len_and_calibration(folder_name)
                    
                    data.append({
                        'model_quantization': quantization_type,
                        'dataset': dataset,
                        'seq_len': seq_len,
                        'calibration_samples': calibration_samples,
                        'limit': limit if limit else 1,
                        'date': date,
                        'mmlu_score': mmlu_score
                    })
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    df = pd.DataFrame(data)
    df = df.sort_values(by=['mmlu_score'], ascending=False)
    
    return df

if __name__ == "__main__":
    df = parse_eval_results()
    print(df.to_string(index=False))
    
    csv_path = LAB_DIR / "eval_results_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

