#!/bin/bash
#SBATCH --job-name=structpo-dpo8b
#SBATCH --account=brics.u5gx
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --mem=115000
#SBATCH --time=4:00:00
#SBATCH --time-min=1:00:00
#SBATCH --output=logs/dpo_8b_%j.out
#SBATCH --error=logs/dpo_8b_%j.err

# ============================================================
# StructPO Stage 2: Structural DPO — Qwen3-8B LoRA
# Requires: 8B Stage 1 LoRA checkpoint + 8B structural pairs
# Target: Isambard-AI Phase 2 (BriCS) — GH200, aarch64
# ============================================================

set -euo pipefail

CONFIG=${1:-configs/dpo/qwen3_8b_structural_dpo_lora.yaml}
NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
WORKDIR=/home/u5gx/cs2175.u5gx/workspace/StructPO

# --- Environment ---
source /home/u5gx/cs2175.u5gx/workspace/share/scripts/activate_env.sh structpo-train

# Override pref_beta from environment if provided (for β sweep)
if [ -n "${PREF_BETA:-}" ]; then
    echo "Overriding pref_beta=$PREF_BETA via LLAMAFACTORY_PREF_BETA env"
    export LLAMAFACTORY_PREF_BETA=$PREF_BETA
fi

# --- Avoid port collision on shared nodes ---
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "=== StructPO 8B DPO Training ==="
echo "Config: $CONFIG"
echo "GPUs: $NPROC_PER_NODE"
echo "Master port: $MASTER_PORT"
echo "pref_beta: ${PREF_BETA:-default from config}"
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

echo "=== 8B DPO Training complete: $(date) ==="
