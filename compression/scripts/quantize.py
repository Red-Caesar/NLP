from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TorchAoConfig
from argparse import ArgumentParser
import os
import torch
from pathlib import Path
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from typing import List, Any
from torchao.quantization import Int4WeightOnlyConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'

LAB_DIR = Path(__file__).parent.parent.resolve()

def prepare_mmlu_dataset() -> Dataset:
    dataset = load_dataset("cais/mmlu", "all", split="test")
    dataset = dataset.shuffle(seed=42)
    def preprocess(example):
        return {"text": example["question"]}
    dataset = dataset.map(preprocess)
    return dataset


def prepare_common_dataset(num_calibration_samples: int) -> Dataset:
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{num_calibration_samples}]")
    dataset = dataset.shuffle(seed=42)
    def preprocess(example):
        return {"text": example["prompt"]}
    dataset = dataset.map(preprocess)
    return dataset

def get_dataset(dataset_type: str, max_calib_samples: int) -> Dataset:
    if dataset_type == "mmlu":
        ds = prepare_mmlu_dataset()
    elif dataset_type == "common":
        ds = prepare_common_dataset(max_calib_samples)
    else:
        raise ValueError(f"Incorrect dataset type: {dataset_type}")

    return ds

def get_recipe(quantization_type: str) -> List[Any]:
    if quantization_type == "fp8":
        recipe = [
            QuantizationModifier(targets="Linear", scheme="FP8", ignore=["lm_head"])
        ]
    elif quantization_type == "awq":
        recipe = [
            AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),
        ]
    else:
        raise ValueError(f"Incorrect quantization type: {quantization_type}")

    return recipe

def main(args):
    model_name = args.model_name
    if model_name[-1] == "/":
        model_name = model_name[:-1]
    model_name = args.model_name.split("/")[-1]

    if args.framework == "llmcompressor":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

        suffix = f"-{args.quantization.upper()}-{args.dataset}-dataset-{args.max_calib_seq_len}-{args.max_calib_samples}"
        save_path = LAB_DIR.joinpath(args.local_save_path, model_name + suffix)
        os.makedirs(save_path, exist_ok=True)

        print("The quatization is started")
        oneshot(
            model=model, 
            dataset=get_dataset(args.dataset, args.max_calib_samples),
            recipe=get_recipe(args.quantization),
            max_seq_length=args.max_calib_seq_len,
            num_calibration_samples=args.max_calib_samples,
            output_dir=save_path,
        )
        print(f"Saving quantized model to: {save_path}")
        tokenizer.save_pretrained(save_path)
        print(f"Quantized model saved successfully.")
    elif args.framework == "torchao":
        quant_config = Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format="tile_packed_to_4d",
            int4_choose_qparams_algorithm="hqq"
        )
        quantization_config = TorchAoConfig(quant_type=quant_config)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        suffix = "-INT4"
        save_path = LAB_DIR.joinpath(args.local_save_path, model_name + suffix)
        os.makedirs(save_path, exist_ok=True)
        print(f"Saving quantized model to: {save_path}")
        model.save_pretrained(save_path, safe_serialization=False)
        tokenizer.save_pretrained(save_path)
    else:
        raise ValueError(f"Incorrect framework:  {args.framework}")

if __name__ == "__main__":
    parser = ArgumentParser(description="CLI for model quantization and saving")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--local_save_path", type=str, default="models/quantized")
    parser.add_argument("--max_calib_samples", type=int, default=128)
    parser.add_argument("--max_calib_seq_len", type=int, default=512)
    parser.add_argument("--dataset", type=str, choices=["mmlu", "common"], default="mmlu")
    parser.add_argument("--quantization", type=str, choices=["awq", "fp8"], default="awq")
    parser.add_argument("--framework", type=str, choices=["llmcompressor, torchao"], default="llmcompressor")
    args = parser.parse_args()

    main(args)
