"""
Entry point for LLaMA-Factory training via deepspeed launcher.

Usage:
    deepspeed --num_gpus=N training/train_entry.py <config.yaml>

This script is needed because:
- `llamafactory-cli` doesn't accept --local_rank from deepspeed launcher
- `torchrun` has EADDRINUSE issues on shared CSD3 GPU nodes
- deepspeed launcher + this script avoids both problems

The deepspeed launcher injects --local_rank=N into sys.argv before the
config path. LLaMA-Factory's read_args() expects sys.argv[1] to be the
YAML config file, so we strip --local_rank from sys.argv first and pass
it via the LOCAL_RANK env var (which DeepSpeed/torch already sets).
"""
import os
import sys
from llamafactory.train.tuner import run_exp

if __name__ == "__main__":
    # Strip --local_rank=N injected by deepspeed launcher so that
    # LLaMA-Factory's read_args() sees the YAML path at sys.argv[1]
    cleaned = [a for a in sys.argv if not a.startswith("--local_rank")]
    sys.argv = cleaned
    run_exp()
