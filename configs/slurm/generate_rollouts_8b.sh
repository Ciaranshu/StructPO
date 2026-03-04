#!/bin/bash
#SBATCH --job-name=structpo-roll8b
#SBATCH --account=brics.u5gx
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=4
#SBATCH --mem=115000
#SBATCH --time=6:00:00
#SBATCH --time-min=2:00:00
#SBATCH --output=logs/rollout_8b_%j.out
#SBATCH --error=logs/rollout_8b_%j.err

# ============================================================
# StructPO: Generate Rollouts from 8B Stage 1 Model (vLLM)
# Requires: 8B LoRA merged model at models/structpo-qwen3-8b-stage1-merged
# Target: Isambard-AI Phase 2 (BriCS) — GH200, aarch64
# ============================================================

set -euo pipefail

WORKDIR=/home/u5gx/cs2175.u5gx/workspace/StructPO
MODEL=${1:-$WORKDIR/models/structpo-qwen3-8b-stage1-merged}
DATASET=$WORKDIR/data/limo_cleaned/limo_original.json
NUM_ROLLOUTS=${2:-8}
SUBSET=${3:-0}  # 0 = all 817 problems
OUTPUT=$WORKDIR/data/rollouts/8b_dse_rollouts.json

# --- Environment (structpo-eval: vLLM 0.16 + pydantic v2) ---
source /home/u5gx/cs2175.u5gx/workspace/share/scripts/activate_env.sh structpo-eval

# GH200 workaround: disable custom all-reduce to avoid CUDA graph crash
# (custom_all_reduce.cuh:455 'invalid argument' during CUDA graph capture with TP=4)
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1

echo "=== StructPO 8B Rollout Generation ==="
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
    --tensor-parallel 4 \
    --output "$OUTPUT" \
    $SUBSET_ARG

echo "=== 8B Rollout generation complete: $(date) ==="
