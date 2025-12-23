import pandas as pd
from pathlib import Path

LAB_DIR = Path(__file__).parent.parent.resolve()

def calculate_metrics():
    sizes_path = LAB_DIR / "model_sizes_summary.csv"
    scores_path = LAB_DIR / "eval_results_summary.csv"
    
    df_sizes = pd.read_csv(sizes_path)
    df_scores = pd.read_csv(scores_path)
    
    df_scores = df_scores[df_scores['limit'] == 0.1] #change to 1.0 for full analysis
    
    df_merged = pd.merge(
        df_scores, 
        df_sizes,
        on=['model_name', 'model_quantization', 'dataset', 'seq_len', 'calibration_samples'],
        how='inner'
    )
    
    original_size = df_sizes[df_sizes['model_quantization'] == 'original']['size_bytes'].values[0]
    original_mmlu = df_scores[df_scores['model_name'] == 'Qwen3-8B']['mmlu_score'].values[0]
    
    df_compressed = df_merged[df_merged['model_quantization'] != 'original'].copy()
    
    df_compressed['compression_ratio'] = original_size / df_compressed['size_bytes']
    df_compressed['performance_drop'] = (original_mmlu - df_compressed['mmlu_score']) / original_mmlu
    df_compressed['lab_score'] = df_compressed['compression_ratio'] / (1 + df_compressed['performance_drop'])
    
    df_result = df_compressed[[
        'model_name', 'model_quantization', 'finetuned', 'dataset', 'seq_len', 'calibration_samples',
        'size_gb', 'mmlu_score', 'compression_ratio', 'performance_drop', 'lab_score'
    ]].copy()
    
    df_result['compression_ratio'] = df_result['compression_ratio'].round(2)
    df_result['performance_drop'] = df_result['performance_drop'].round(4)
    df_result['lab_score'] = df_result['lab_score'].round(2)
    
    df_result = df_result.sort_values(by='lab_score', ascending=False)
    
    return df_result, original_size, original_mmlu

if __name__ == "__main__":
    df, orig_size, orig_mmlu = calculate_metrics()
    
    print(f"Original Model:")
    print(f"  Size: {orig_size / (1024**3):.2f} GB")
    print(f"  MMLU Score: {orig_mmlu:.4f}")
    print(f"\nCompressed Models:")
    print(df.to_string(index=False))
    
    output_path = LAB_DIR / "lab_scores.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

