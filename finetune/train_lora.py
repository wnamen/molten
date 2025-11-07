#!/usr/bin/env python3
"""
Train MoE-aware LoRA adapters on K2-Instruct for ML engineering specialization.
Uses PEFT library with support for MoE models.
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

@dataclass
class LoRAConfig:
    """LoRA configuration for MoE models."""
    r: int = 16  # Rank
    lora_alpha: int = 32
    target_modules: list = None  # Will set to attention + FFN
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: TaskType = TaskType.CAUSAL_LM

def prepare_model_and_tokenizer(model_path: str):
    """Load base model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def setup_lora_for_moe(model, config: LoRAConfig):
    """Setup LoRA for MoE model (target attention + FFN, avoid router)."""
    # For MoE models, target attention layers and FFN (but not router)
    # K2-Instruct uses standard transformer architecture
    
    if config.target_modules is None:
        # Default: target attention and MLP layers
        config.target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",  # MLP
        ]
    
    lora_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=config.task_type,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def format_instruction(example: dict) -> str:
    """Format instruction-following example."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return prompt

def load_training_data(data_path: Path):
    """Load training data from JSONL."""
    examples = []
    with open(data_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    
    # Format as instruction-following
    formatted = [format_instruction(ex) for ex in examples]
    return formatted

def tokenize_function(examples, tokenizer, max_length=2048):
    """Tokenize examples."""
    return tokenizer(
        examples,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

def train(
    model_path: str,
    data_path: Path,
    output_dir: Path,
    lora_config: Optional[LoRAConfig] = None,
    num_epochs: int = 2,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 2048,
):
    """Train LoRA adapters."""
    lora_config = lora_config or LoRAConfig()
    
    # Load model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(model_path)
    
    # Setup LoRA
    model = setup_lora_for_moe(model, lora_config)
    
    # Load training data
    print(f"Loading training data from {data_path}...")
    examples = load_training_data(data_path)
    
    # Tokenize
    tokenized = tokenize_function(examples, tokenizer, max_length=max_length)
    
    # Create dataset
    from datasets import Dataset
    dataset = Dataset.from_dict({"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]})
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=False,
        bf16=True,  # Use bfloat16 for stability
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        warmup_steps=50,
        report_to="none",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save
    print(f"Saving adapters to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("✓ Training complete!")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train LoRA adapters on K2-Instruct")
    parser.add_argument("--model-path", type=str, default="MoonshotAI/Kimi-K2-Instruct")
    parser.add_argument("--data-path", type=str, required=True, help="Path to training JSONL")
    parser.add_argument("--output-dir", type=str, default="./lora_adapters")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    
    args = parser.parse_args()
    
    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    
    train(
        model_path=args.model_path,
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        lora_config=lora_config,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )

if __name__ == "__main__":
    main()

