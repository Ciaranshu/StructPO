#!/bin/bash
#SBATCH --job-name=m500-roll
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=115000
#SBATCH --time=1:00:00
#SBATCH --output=logs/m500_roll_%j.out
#SBATCH --error=logs/m500_roll_%j.err

# ============================================================
# Generate K=8 MATH-500 rollouts (sharded for 1hr jobs)
#
# Usage:
#   # 4B model, 4 shards of 125 problems each:
#   for i in 0 1 2 3; do
#     sbatch --account=COLLIER-SL3-GPU \
#       --export=ALL,MODEL=models/decor-qwen3-4b-dse,SHARD=$i,TOTAL_SHARDS=4,MODEL_SIZE=4b \
#       configs/slurm/gen_math500_rollouts_1hr.sh
#   done
#
#   # 8B model, 4 shards:
#   for i in 0 1 2 3; do
#     sbatch --account=SHAREGHI-SL3-GPU \
#       --export=ALL,MODEL=models/structpo-qwen3-8b-stage1-merged,SHARD=$i,TOTAL_SHARDS=4,MODEL_SIZE=8b \
#       configs/slurm/gen_math500_rollouts_1hr.sh
#   done
#
#   # After all shards complete, merge:
#   python scripts/merge_rollout_shards.py \
#     --shards data/rollouts/math500_4b_shard_*.json \
#     --output data/rollouts/math500_4b_rollouts.json
# ============================================================

set -euo pipefail

MODEL=${MODEL:?Set MODEL env var}
SHARD=${SHARD:?Set SHARD env var (0-indexed)}
TOTAL_SHARDS=${TOTAL_SHARDS:-4}
MODEL_SIZE=${MODEL_SIZE:-4b}
PROBLEMS_TOTAL=500
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

OUTPUT="data/rollouts/math500_${MODEL_SIZE}_shard_${SHARD}.json"

# --- Environment ---
source /home/cs2175/rds/workspace/share/scripts/activate_env.sh structpo-eval

echo "=== MATH-500 Rollout Generation (Shard ${SHARD}/${TOTAL_SHARDS}) ==="
echo "Model: $MODEL"
echo "Problems: offset=$OFFSET, count=$SUBSET (total=$PROBLEMS_TOTAL)"
echo "Output: $OUTPUT"
echo "Node: $(hostname), Job: $SLURM_JOB_ID"
echo "Start: $(date)"
nvidia-smi --list-gpus

cd $WORKDIR
mkdir -p logs data/rollouts

# Use MATH-500 as the dataset (eval benchmark, not training data)
# MATH-500 is loaded via evaluate.py's benchmark loader, but for rollouts
# we need it in problem format. Use the LIMO dataset as proxy for now,
# or load MATH-500 directly if available.

# Check if MATH-500 problem file exists
MATH500_DATASET="data/math500_problems.json"
if [ ! -f "$MATH500_DATASET" ]; then
    echo "Generating MATH-500 problem file..."
    python -c "
import json
from pathlib import Path

# Load MATH-500 from the evaluation benchmark
# This is the same set used by evaluate.py
try:
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    problems = []
    for i, item in enumerate(ds):
        problems.append({
            'conversations': [
                {'role': 'user', 'value': item['problem']},
                {'role': 'assistant', 'value': item['solution']}
            ]
        })
    Path('$MATH500_DATASET').write_text(json.dumps(problems, indent=2))
    print(f'Saved {len(problems)} MATH-500 problems')
except Exception as e:
    print(f'Error loading MATH-500: {e}')
    print('Falling back to LIMO subset...')
    # Use first 500 LIMO problems as fallback
    data = json.loads(Path('data/limo_cleaned/limo_original.json').read_text())
    Path('$MATH500_DATASET').write_text(json.dumps(data[:500], indent=2))
    print(f'Saved 500 LIMO problems as fallback')
"
fi

python scripts/generate_rollouts.py \
    --model "$MODEL" \
    --dataset "$MATH500_DATASET" \
    --num-rollouts 8 \
    --temperature 0.7 \
    --max-tokens 16384 \
    --tensor-parallel 4 \
    --offset $OFFSET \
    --subset $SUBSET \
    --output "$OUTPUT"

echo "=== Shard $SHARD complete: $(date) ==="
