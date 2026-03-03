#!/bin/bash
#SBATCH --job-name=structpo-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

# ============================================================
# StructPO Stage 1: SFT on DSE-Cleaned Data
# Adapt paths for your HPC system (CSD3 / BriCS)
# ============================================================

set -euo pipefail

# --- Configuration ---
CONFIG=${1:-configs/sft/qwen3_4b_dse_sft.yaml}
NPROC_PER_NODE=2

# --- Environment ---
# Uncomment and adjust for your system:
# module purge
# conda activate structpo

# CUDA setup (adjust for your cluster)
# export CUDA_HOME=/usr/local/software/cuda/12.1
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# --- Avoid port collision on shared nodes ---
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "=== StructPO SFT Training ==="
echo "Config: $CONFIG"
echo "GPUs: $NPROC_PER_NODE"
echo "Master port: $MASTER_PORT"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

# --- Launch ---
deepspeed \
    --num_gpus=$NPROC_PER_NODE \
    --master_port=$MASTER_PORT \
    src/training/train_entry.py \
    $CONFIG

echo "=== Training complete: $(date) ==="
