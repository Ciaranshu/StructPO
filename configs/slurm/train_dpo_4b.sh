#!/bin/bash
#SBATCH --job-name=structpo-dpo4b
#SBATCH --account=brics.u5gx
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=144
#SBATCH --time=4:00:00
#SBATCH --output=logs/dpo_4b_%j.out
#SBATCH --error=logs/dpo_4b_%j.err

# ============================================================
# StructPO Stage 2: Structural DPO on 4B DSE-SFT checkpoint
# Full pipeline validation before scaling to 8B
# Target: Isambard-AI Phase 2 (BriCS) — GH200, aarch64
# ============================================================

set -euo pipefail

CONFIG=${1:-configs/dpo/qwen3_4b_structural_dpo.yaml}
NPROC_PER_NODE=2
WORKDIR=/home/u5gx/cs2175.u5gx/workspace/StructPO

# --- Environment ---
module load cuda/12.6
export MAMBA_ROOT_PREFIX=/home/u5gx/cs2175.u5gx/micromamba
eval "$(/home/u5gx/cs2175.u5gx/bin/micromamba shell hook -s bash)"
micromamba activate comprl

# --- Avoid port collision on shared nodes ---
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "=== StructPO 4B DPO Training ==="
echo "Config: $CONFIG"
echo "GPUs: $NPROC_PER_NODE"
echo "Master port: $MASTER_PORT"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "Python: $(which python)"
nvidia-smi --list-gpus

cd $WORKDIR
mkdir -p logs

# --- Launch ---
deepspeed \
    --num_gpus=$NPROC_PER_NODE \
    --master_port=$MASTER_PORT \
    src/training/train_entry.py \
    $CONFIG

echo "=== DPO Training complete: $(date) ==="
