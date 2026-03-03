#!/bin/bash
#SBATCH --job-name=structpo-dpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=logs/dpo_%j.out
#SBATCH --error=logs/dpo_%j.err

# ============================================================
# StructPO Stage 2: Structural Preference DPO
# Adapt paths for your HPC system (CSD3 / BriCS)
# ============================================================

set -euo pipefail

# --- Configuration ---
CONFIG=${1:-configs/dpo/qwen3_4b_structural_dpo.yaml}
NPROC_PER_NODE=2

# --- Environment ---
# Uncomment and adjust for your system:
# module purge
# conda activate structpo

# --- Avoid port collision on shared nodes ---
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "=== StructPO DPO Training ==="
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

echo "=== DPO Training complete: $(date) ==="
