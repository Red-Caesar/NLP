#!/bin/bash

set -e

# ============================================
# CONFIGURATION - Edit these values
# ============================================

HF_TOKEN=""

QUANTIZED_MODELS=(
    "RedCaesar/Qwen3-8B-AWQ-mmlu-dataset-512-128"
    "RedCaesar/Qwen3-8B-INT4"
)

FINETUNED_MODELS=(
    "RedCaesar/Qwen3-8B-LoRA-INT4"
)

LORA_ADAPTERS=(
    "RedCaesar/LoRA_MMLU"
)

BASE_MODEL_FOR_LORA="models/quantized/Qwen3-8B-AWQ-mmlu-dataset-512-128"

VLLM_PORT=8000
EVAL_LIMIT=0.1
SKIP_DOWNLOAD=true

# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

QUANTIZED_DIR="models/quantized"
FINETUNED_DIR="models/finetuned"
LORA_DIR="models/finetuned/LoRA"
EVAL_RESULTS_DIR="eval_results"
LOGS_DIR=".logs"

mkdir -p "$QUANTIZED_DIR" "$FINETUNED_DIR" "$LORA_DIR" "$EVAL_RESULTS_DIR" "$LOGS_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

download_model() {
    local model_name=$1
    local target_dir=$2
    
    log "Downloading model: $model_name to $target_dir"
    
    local hf_args=""
    if [ -n "$HF_TOKEN" ]; then
        hf_args="--token $HF_TOKEN"
    fi
    
    hf download "$model_name" \
        --local-dir "$target_dir" \
        $hf_args
    
    log "Downloaded: $model_name"
}

wait_for_vllm() {
    local max_attempts=120
    local attempt=0
    
    log "Waiting for vLLM server to be ready..."
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$VLLM_PORT/health" | grep -q "200"; then
            log "vLLM server is ready!"
            return 0
        fi
        
        attempt=$((attempt + 1))
        sleep 2
    done
    
    log "ERROR: vLLM server failed to start"
    return 1
}

stop_vllm() {
    log "Stopping vLLM server..."
    
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    
    pkill -f "vllm.entrypoints" || true
    sleep 2
}

evaluate_model() {
    local model_path=$1
    local model_name=$(basename "$model_path")
    
    log "=========================================="
    log "Evaluating model: $model_name"
    log "=========================================="
    
    stop_vllm
    
    log "Starting vLLM server for $model_name"
    vllm serve "$model_path" --port "$VLLM_PORT" > "$LOGS_DIR/vllm_${model_name}.log" 2>&1 &
    VLLM_PID=$!
    
    if ! wait_for_vllm; then
        log "ERROR: Failed to start vLLM for $model_name"
        stop_vllm
        return 1
    fi
    
    log "Running lm_eval for $model_name"
    lm_eval \
        --model local-completions \
        --model_args "model=${model_path},base_url=http://localhost:${VLLM_PORT}/v1/completions,num_concurrent=128" \
        --tasks mmlu \
        --output_path "$EVAL_RESULTS_DIR" \
        --write_out \
        --limit "$EVAL_LIMIT" \
        2>&1 | tee "$LOGS_DIR/eval_${model_name}.log"
    
    log "Evaluation completed for $model_name"
    stop_vllm
}

evaluate_lora_model() {
    local base_model=$1
    local lora_path=$2
    local lora_name=$(basename "$lora_path")
    
    log "=========================================="
    log "Evaluating LoRA: $lora_name on base model"
    log "=========================================="
    
    stop_vllm
    
    log "Starting vLLM server with LoRA support"
    vllm serve "$base_model" \
        --port "$VLLM_PORT" \
        --enable-lora \
        --lora-modules "lora_mmlu=${lora_path}" \
        > "$LOGS_DIR/vllm_lora_${lora_name}.log" 2>&1 &
    VLLM_PID=$!
    
    if ! wait_for_vllm; then
        log "ERROR: Failed to start vLLM with LoRA for $lora_name"
        stop_vllm
        return 1
    fi
    
    log "Running lm_eval for LoRA: $lora_name"
    lm_eval \
        --model local-completions \
        --model_args "model=lora_mmlu,base_url=http://localhost:${VLLM_PORT}/v1/completions,num_concurrent=128,tokenizer=${base_model}" \
        --tasks mmlu \
        --output_path "$EVAL_RESULTS_DIR" \
        --write_out \
        --limit "$EVAL_LIMIT" \
        2>&1 | tee "$LOGS_DIR/eval_lora_${lora_name}.log"
    
    log "Evaluation completed for LoRA: $lora_name"
    stop_vllm
}

if [ "$SKIP_DOWNLOAD" = false ]; then
    log "=========================================="
    log "Downloading Models"
    log "=========================================="
    
    for model in "${QUANTIZED_MODELS[@]}"; do
        model_name=$(basename "$model")
        target_path="$QUANTIZED_DIR/$model_name"
        
        if [ -d "$target_path" ]; then
            log "Model already exists: $target_path (skipping)"
        else
            download_model "$model" "$target_path"
        fi
    done
    
    for model in "${FINETUNED_MODELS[@]}"; do
        model_name=$(basename "$model")
        target_path="$FINETUNED_DIR/$model_name"
        
        if [ -d "$target_path" ]; then
            log "Model already exists: $target_path (skipping)"
        else
            download_model "$model" "$target_path"
        fi
    done
    
    for lora in "${LORA_ADAPTERS[@]}"; do
        lora_name=$(basename "$lora")
        target_path="$LORA_DIR/$lora_name"
        
        if [ -d "$target_path" ]; then
            log "LoRA adapter already exists: $target_path (skipping)"
        else
            download_model "$lora" "$target_path"
        fi
    done
else
    log "Skipping model downloads (SKIP_DOWNLOAD=true)"
fi

log "=========================================="
log "Starting Model Evaluations"
log "=========================================="

for model_dir in "$QUANTIZED_DIR"/*; do
    if [ -d "$model_dir" ]; then
        evaluate_model "$model_dir" || log "WARNING: Evaluation failed for $model_dir"
    fi
done

for model_dir in "$FINETUNED_DIR"/*; do
    if [ -d "$model_dir" ] && [ "$model_dir" != "$LORA_DIR" ]; then
        evaluate_model "$model_dir" || log "WARNING: Evaluation failed for $model_dir"
    fi
done

for lora_dir in "$LORA_DIR"/*; do
    if [ -d "$lora_dir" ]; then
        evaluate_lora_model "$BASE_MODEL_FOR_LORA" "$lora_dir" || log "WARNING: LoRA evaluation failed for $lora_dir"
    fi
done

stop_vllm

log "=========================================="
log "Running Analysis Scripts"
log "=========================================="

log "Getting model sizes..."
python scripts/get_model_size.py

log "Getting MMLU scores..."
python scripts/get_mmlu_score.py

log "Calculating lab scores..."
python scripts/calculate_score.py
