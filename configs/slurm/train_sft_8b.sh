#!/bin/bash
#SBATCH --job-name=structpo-sft8b
#SBATCH --account=brics.u5gx
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --mem=115000
#SBATCH --time=12:00:00
#SBATCH --time-min=2:00:00
#SBATCH --output=logs/sft_8b_%j.out
#SBATCH --error=logs/sft_8b_%j.err

# ============================================================
# StructPO Stage 1: SFT on DSE-Cleaned LIMO — Qwen3-8B LoRA
# Target: Isambard-AI Phase 2 (BriCS) — GH200, aarch64
# ============================================================

set -euo pipefail

CONFIG=${1:-configs/sft/qwen3_8b_dse_sft_lora.yaml}
NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
WORKDIR=/home/u5gx/cs2175.u5gx/workspace/StructPO

# --- Environment ---
source /home/u5gx/cs2175.u5gx/workspace/share/scripts/activate_env.sh structpo-train

# --- Avoid port collision on shared nodes ---
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "=== StructPO 8B SFT Training ==="
echo "Config: $CONFIG"
echo "GPUs: $NPROC_PER_NODE"
echo "Master port: $MASTER_PORT"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
nvidia-smi --list-gpus

cd $WORKDIR
mkdir -p logs

# --- Launch ---
deepspeed \
    --num_gpus=$NPROC_PER_NODE \
    --master_port=$MASTER_PORT \
    src/structpo/training/train_entry.py \
    $CONFIG

echo "=== 8B SFT Training complete: $(date) ==="
