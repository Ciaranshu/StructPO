#!/bin/bash
#SBATCH --job-name=dpo8b-1hr
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=115000
#SBATCH --time=1:00:00
#SBATCH --output=logs/dpo_8b_1hr_%j.out
#SBATCH --error=logs/dpo_8b_1hr_%j.err

# ============================================================
# 8B DPO Training — 1hr version (1 epoch instead of 3)
#
# For quick validation: does the signal produce reasonable loss?
# Full 3-epoch training should use train_dpo_8b.sh (4hr).
#
# Usage:
#   sbatch --account=COLLIER-SL3-GPU \
#     --export=ALL,CONFIG=configs/dpo/qwen3_8b_type13_dpo_beta020.yaml \
#     configs/slurm/train_dpo_8b_1hr.sh
# ============================================================

set -euo pipefail

CONFIG=${1:-${CONFIG:-configs/dpo/qwen3_8b_structural_dpo_lora.yaml}}
NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
WORKDIR=/home/cs2175/rds/workspace/StructPO

source /home/cs2175/rds/workspace/share/scripts/activate_env.sh structpo-train

# Override to 1 epoch for quick validation
export LLAMAFACTORY_NUM_TRAIN_EPOCHS=1

if [ -n "${PREF_BETA:-}" ]; then
    export LLAMAFACTORY_PREF_BETA=$PREF_BETA
fi

MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "=== 8B DPO Training (1hr / 1 epoch) ==="
echo "Config: $CONFIG"
echo "GPUs: $NPROC_PER_NODE"
echo "pref_beta: ${PREF_BETA:-default}"
echo "Node: $(hostname), Job: $SLURM_JOB_ID"
echo "Start: $(date)"
nvidia-smi --list-gpus

cd $WORKDIR
mkdir -p logs

deepspeed \
    --num_gpus=$NPROC_PER_NODE \
    --master_port=$MASTER_PORT \
    src/structpo/training/train_entry.py \
    $CONFIG

echo "=== 1hr DPO Training complete: $(date) ==="
