#!/bin/bash
#SBATCH --job-name=structpo-roll4b
#SBATCH --account=brics.u5gx
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=72
#SBATCH --time=8:00:00
#SBATCH --output=logs/rollout_4b_%j.out
#SBATCH --error=logs/rollout_4b_%j.err

# ============================================================
# StructPO: Generate Rollouts from 4B DSE-SFT Model (vLLM)
# Full pipeline validation on 4B before scaling to 8B
# Target: Isambard-AI Phase 2 (BriCS) — GH200, aarch64
# ============================================================

set -euo pipefail

WORKDIR=/home/u5gx/cs2175.u5gx/workspace/StructPO
MODEL=$WORKDIR/models/decor-qwen3-4b-dse
DATASET=$WORKDIR/data/limo_cleaned/limo_original.json
NUM_ROLLOUTS=${1:-8}
SUBSET=${2:-0}  # 0 = all 817 problems, set to e.g. 50 for quick test
OUTPUT=$WORKDIR/data/rollouts/4b_dse_rollouts.json

# --- Environment (structpo-eval: vLLM 0.16 + pydantic v2) ---
module load cuda/12.6
export MAMBA_ROOT_PREFIX=/home/u5gx/cs2175.u5gx/micromamba
eval "$(/home/u5gx/cs2175.u5gx/bin/micromamba shell hook -s bash)"
micromamba activate structpo-eval

echo "=== StructPO 4B Rollout Generation ==="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Num rollouts: $NUM_ROLLOUTS"
echo "Subset: $SUBSET (0=all)"
echo "Output: $OUTPUT"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
nvidia-smi --list-gpus

cd $WORKDIR
mkdir -p logs data/rollouts

SUBSET_ARG=""
if [ "$SUBSET" -gt 0 ]; then
    SUBSET_ARG="--subset $SUBSET"
fi

python scripts/generate_rollouts.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --num-rollouts "$NUM_ROLLOUTS" \
    --temperature 0.7 \
    --output "$OUTPUT" \
    $SUBSET_ARG

echo "=== Rollout generation complete: $(date) ==="
