#!/bin/bash
#SBATCH --job-name=structpo-rollout
#SBATCH --account=brics.u5gx
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --time=8:00:00
#SBATCH --output=logs/rollout_%j.out
#SBATCH --error=logs/rollout_%j.err

# ============================================================
# StructPO: Generate Rollouts from Stage 1 Model (vLLM)
# Target: Isambard-AI Phase 2 (BriCS) — GH200, aarch64
# Uses structpo-eval environment (has vLLM)
# ============================================================

set -euo pipefail

WORKDIR=/home/u5gx/cs2175.u5gx/workspace/StructPO
MODEL=${1:-$WORKDIR/models/structpo-qwen3-4b-stage1}
DATASET=${2:-$WORKDIR/data/limo_cleaned/limo_original.json}
NUM_ROLLOUTS=${3:-8}
OUTPUT=${4:-$WORKDIR/data/rollouts/stage1_rollouts.json}

# --- Environment (eval env with vLLM) ---
module load cuda/12.6
export MAMBA_ROOT_PREFIX=/home/u5gx/cs2175.u5gx/micromamba
eval "$(/home/u5gx/cs2175.u5gx/bin/micromamba shell hook -s bash)"
micromamba activate structpo-eval

echo "=== StructPO Rollout Generation ==="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Num rollouts: $NUM_ROLLOUTS"
echo "Output: $OUTPUT"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
nvidia-smi --list-gpus

cd $WORKDIR
mkdir -p logs data/rollouts

python scripts/generate_rollouts.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --num-rollouts "$NUM_ROLLOUTS" \
    --temperature 0.7 \
    --output "$OUTPUT"

echo "=== Rollout generation complete: $(date) ==="
