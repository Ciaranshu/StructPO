#!/usr/bin/env python3
"""
Merge LoRA adapter weights back into the base model to create a standalone model.

Usage:
    python scripts/merge_lora.py \
        --base models/decor-qwen3-4b-dse \
        --adapter models/structpo-qwen3-4b-stage2 \
        --output models/structpo-qwen3-4b-stage2-merged
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base", required=True, help="Path to base model")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--output", required=True, help="Output path for merged model")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"Loading base model: {args.base}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="cpu",
    )

    print(f"Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)

    print("Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)

    print(f"Done! Merged model saved to {args.output}")
    print(f"  Model files: {list(Path(args.output).glob('*.safetensors'))}")


if __name__ == "__main__":
    main()
