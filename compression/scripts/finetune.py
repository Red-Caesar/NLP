from unsloth import FastLanguageModel
from datasets import load_dataset
from argparse import ArgumentParser
from pathlib import Path
from transformers import TorchAoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
import torch
import pandas as pd
from torchao.quantization import Int4WeightOnlyConfig

LAB_DIR = Path(__file__).parent.parent.resolve()

def prepare_mmlu_for_training():
    dataset = load_dataset("cais/mmlu", "all", split="test") #auxiliary_train

    def format_example(example):
        choices = example['choices']
        question = example['question']
        answer_idx = example['answer']
        correct_answer = choices[answer_idx]
        
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += f"Answer: {correct_answer}"
        
        return {"text": prompt}
    
    return dataset.map(format_example)


def main(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=False,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        use_gradient_checkpointing="unsloth",
        use_rslora=False,
        random_state=42,
        qat_scheme="int4", #Important
    )
    model.print_trainable_parameters()
    
    dataset = prepare_mmlu_for_training()
    if args.max_samples:
        dataset = dataset.shuffle(seed=42).select(range(min(args.max_samples, len(dataset))))
        print(f"Limited to {len(dataset)} samples")

    output_dir = LAB_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=50,
        optim="adamw_8bit",
        warmup_steps=5,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=training_args,
    )
    trainer.train()
    
    if args.save_merged:
        print("Saving merged model...")
        merged_dir = output_dir / f"merged"
        merged_dir.mkdir(parents=True, exist_ok=True)

        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

        quant_config = Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format="tile_packed_to_4d",
            int4_choose_qparams_algorithm="hqq"
        )
        quantization_config = TorchAoConfig(quant_type=quant_config)

        model = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(merged_dir)

        merged_dir = output_dir / f"Qwen3-8B-LoRA-INT4"
        model.save_pretrained(merged_dir, safe_serialization=False)
        tokenizer.save_pretrained(merged_dir)
        print(f"Merged model saved to: {merged_dir}")
    else:
        print("Saving LoRA adapters only...")
        lora_dir = output_dir / "LoRA" / "LoRA_MMLU"
        model.save_pretrained(str(lora_dir))
        tokenizer.save_pretrained(str(lora_dir))
        print(f"LoRA adapters saved to: {lora_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", type=str, default="models/finetuned")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_merged", action="store_true")

    args = parser.parse_args()
    main(args)

