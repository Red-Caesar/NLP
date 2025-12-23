# NLP Model Compression Lab

## Installation

1. Create and activate virtual environment:
```bash
uv venv --python 3.12 --python-preference only-managed
source .venv/bin/activate
```

2. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Quantization

Quantize a model using AWQ, FP8 or INT4(W4A16):

```bash
python scripts/quantize.py \
    --model_name Qwen/Qwen3-8B \
    --quantization awq \
    --dataset mmlu \
    --max_calib_samples 128 \
    --max_calib_seq_len 512
```

Options:
- `--quantization`: `awq` or `fp8`
- `--dataset`: `mmlu` or `common`
- `--framework`: `llmcompressor` (for `awq`, `fp8`) or `torchao` (for `int4`)
- `--local_save_path`: Output directory (default: `models/quantized`)

## Finetuning

Train LoRA adapters:

```bash
python scripts/finetune.py 
    --num_epochs 3 \
    --batch_size 2 \
    --max_samples 1000
```

Options:
- `--save_merged`: Save merged model instead of just LoRA adapters

## Evaluation

Start vllm server:
```bash
vllm serve MODEL_PATH
```

Run MMLU evaluation using lm-evaluation-harness:

```bash
lm_eval \
    --model local-completions \
    --model_args model=MODEL_PATH,base_url=http://localhost:8000/v1/completions,num_concurrent=128 \
    --tasks mmlu \
    --output_path ./eval_results \
    --write_out
```

Use `--limit 0.1` to evaluate on 10% of benchmark.

## Analysis Scripts

Calculate sizes of all quantized models:

```bash
python scripts/get_model_size.py
```

Parse evaluation results:

```bash
python scripts/get_mmlu_score.py
```

Compute compression metrics and lab scores based on previos files:

```bash
python scripts/calculate_score.py
```

## Project Structure

```
compression/
├── models/
│   ├── quantized/         
│   └── finetuned/ 
│       ├── LoRA/
│       └── ...
├── eval_results/
├── scripts/
└── run_evaluation.sh
```

## Automated Evaluation Pipeline

The `run_evaluation.sh` script automates the evaluation workflow.

### Configuration

Edit the configuration section at the top of `run_evaluation.sh`:

```bash
HF_TOKEN=""

# Evaluation settings
VLLM_PORT=8000
EVAL_LIMIT=1.0        # Use 0.1 for quick test
SKIP_DOWNLOAD=false
```

### Run

```bash
./run_evaluation.sh
```
## Results


| Model | Quantization | Finetuned | Size (GB) | MMLU Score | Compression Ratio | Performance Drop | Lab Score |
|-------|-------------|-----------|-----------|------------|-------------------|------------------|-----------|
| Qwen3-8B-INT4 | INT4 | No | 5.76 | 0.709 | 2.65 | 0.0297 | **2.57** |
| Qwen3-8B-LoRA-INT4 | INT4 | Yes | 5.76 | 0.486 | 2.65 | 0.3349 | 1.99 |
| Qwen3-8B-AWQ-mmlu-dataset-512-128 | AWQ | Yes | 8.10 | 0.720 | 1.88 | 0.0144 | 1.86 |
| Qwen3-8B-AWQ-common-dataset-512-128 | AWQ | No | 8.10 | 0.717 | 1.88 | 0.0191 | 1.85 |
| Qwen3-8B-AWQ-common-dataset-1024-256 | AWQ | No | 8.10 | 0.713 | 1.88 | 0.0249 | 1.84 |
| Qwen3-8B-AWQ-mmlu-dataset-512-128 | AWQ | No | 8.10 | 0.714 | 1.88 | 0.0230 | 1.84 |
| Qwen3-8B-AWQ-mmlu-dataset-2048-256 | AWQ | No | 8.10 | 0.708 | 1.88 | 0.0316 | 1.83 |
| Qwen3-8B-FP8-mmlu-dataset-512-128 | FP8 | No | 11.11 | 0.724 | 1.37 | 0.0096 | 1.36 |
| Qwen3-8B-FP8-common-dataset-512-128 | FP8 | No | 11.11 | 0.722 | 1.37 | 0.0115 | 1.36 |

While Qwen3-8B-INT4 achieves the highest lab score, it cannot be easily finetuned. Training a LoRA adapter and merging it into the model results in a poor MMLU score of 0.486, however Qwen3-8B-LoRA-INT4 showing a good lab score 1.99 (?).

Our preferred model is **Qwen3-8B-AWQ-mmlu-dataset-512-128**, though training LoRA for it was challenging. To achieve the best results, it's necessary to use the unsloth library with a parameter that makes LoRA aware of future INT4 quantization. More about it [here](https://www.e2enetworks.com/blog/train-4bit-llms-qat-unsloth#the-power-behind-the-recovery-unsloth-and-pytorch-ao).
