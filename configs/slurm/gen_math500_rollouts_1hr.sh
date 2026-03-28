#!/bin/bash
#SBATCH --job-name=m500-roll
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=115000
#SBATCH --time=1:00:00
#SBATCH --output=logs/m500_roll_%j.out
#SBATCH --error=logs/m500_roll_%j.err

# ============================================================
# Generate K=8 MATH-500 rollouts (sharded for 1hr jobs)
#
# Each shard is designed to finish within 1 hour.
# GPU count is set via NUM_GPUS env var (default: use model-specific defaults).
#
# Usage:
#   # 4B model: 1 GPU per shard, 10 shards of 50 problems (~33min each)
#   for i in $(seq 0 9); do
#     sbatch --account=COLLIER-SL3-GPU --gres=gpu:1 \
#       --export=ALL,MODEL=models/decor-qwen3-4b-dse,SHARD=$i,TOTAL_SHARDS=10,MODEL_SIZE=4b,NUM_GPUS=1 \
#       configs/slurm/gen_math500_rollouts_1hr.sh
#   done
#
#   # 8B model: 4 GPU per shard (TP=4 required), 5 shards of 100 problems (~44min each)
#   for i in $(seq 0 4); do
#     sbatch --account=SHAREGHI-SL3-GPU --gres=gpu:4 \
#       --export=ALL,MODEL=models/structpo-qwen3-8b-stage1-merged,SHARD=$i,TOTAL_SHARDS=5,MODEL_SIZE=8b,NUM_GPUS=4 \
#       configs/slurm/gen_math500_rollouts_1hr.sh
#   done
#
#   # After all shards complete, merge:
#   python scripts/merge_rollout_shards.py \
#     --shards data/rollouts/math500_4b_shard_*.json \
#     --output data/rollouts/math500_4b_rollouts.json
#
# GPU-hrs budget:
#   4B: 10 shards × 1 GPU × 1hr = 10 GPU-hrs
#   8B: 5 shards × 4 GPU × 1hr = 20 GPU-hrs
#   Total: 30 GPU-hrs
# ============================================================

set -euo pipefail

MODEL=${MODEL:?Set MODEL env var}
SHARD=${SHARD:?Set SHARD env var (0-indexed)}
TOTAL_SHARDS=${TOTAL_SHARDS:-10}
MODEL_SIZE=${MODEL_SIZE:-4b}
NUM_GPUS=${NUM_GPUS:-1}
PROBLEMS_TOTAL=${PROBLEMS_TOTAL:-500}
BENCHMARK=${BENCHMARK:-math500}
DATASET_FILE=${DATASET_FILE:-data/math500_problems.json}
WORKDIR=/home/cs2175/rds/workspace/StructPO

# Calculate offset and subset for this shard
PROBLEMS_PER_SHARD=$((PROBLEMS_TOTAL / TOTAL_SHARDS))
OFFSET=$((SHARD * PROBLEMS_PER_SHARD))

# Last shard gets remaining problems
if [ "$SHARD" -eq $((TOTAL_SHARDS - 1)) ]; then
    SUBSET=$((PROBLEMS_TOTAL - OFFSET))
else
    SUBSET=$PROBLEMS_PER_SHARD
fi

OUTPUT="data/rollouts/${BENCHMARK}_${MODEL_SIZE}_shard_${SHARD}.json"

# --- Environment ---
source /home/cs2175/rds/workspace/share/scripts/activate_env.sh structpo-eval

echo "=== MATH-500 Rollout Generation (Shard ${SHARD}/${TOTAL_SHARDS}) ==="
echo "Model: $MODEL (${MODEL_SIZE})"
echo "GPUs: $NUM_GPUS (TP=$NUM_GPUS)"
echo "Problems: offset=$OFFSET, count=$SUBSET (total=$PROBLEMS_TOTAL)"
echo "Output: $OUTPUT"
echo "Node: $(hostname), Job: $SLURM_JOB_ID"
echo "Start: $(date)"
nvidia-smi --list-gpus

cd $WORKDIR
mkdir -p logs data/rollouts

# Check dataset file exists
if [ ! -f "$DATASET_FILE" ]; then
    echo "ERROR: Dataset file not found: $DATASET_FILE"
    echo "Create it first on login node (see docs/phase/phase3_ablations.md)"
    exit 1
fi

python scripts/generate_rollouts.py \
    --model "$MODEL" \
    --dataset "$DATASET_FILE" \
    --num-rollouts 8 \
    --temperature 0.7 \
    --max-tokens 16384 \
    --tensor-parallel $NUM_GPUS \
    --offset $OFFSET \
    --subset $SUBSET \
    --output "$OUTPUT"

echo "=== Shard $SHARD complete: $(date) ==="
