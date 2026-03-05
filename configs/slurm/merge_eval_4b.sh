#!/bin/bash
#SBATCH --job-name=structpo-meval
#SBATCH --account=brics.u5gx
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=115000
#SBATCH --time=4:00:00
#SBATCH --time-min=1:00:00
#SBATCH --output=logs/merge_eval_%j.out
#SBATCH --error=logs/merge_eval_%j.err

set -euo pipefail

BETA=${1:?Usage: sbatch merge_eval_4b.sh <beta>}
WORKDIR=/home/u5gx/cs2175.u5gx/workspace/StructPO
BASE_MODEL=$WORKDIR/models/decor-qwen3-4b-dse
ADAPTER=$WORKDIR/models/structpo-qwen3-4b-stage2-beta${BETA}
MERGED=$WORKDIR/models/structpo-qwen3-4b-stage2-beta${BETA}-merged

echo "=== Merge + Eval: beta=$BETA ==="
echo "Base: $BASE_MODEL"
echo "Adapter: $ADAPTER"
echo "Merged: $MERGED"
echo "Node: $(hostname)"
echo "Start: $(date)"

cd $WORKDIR
mkdir -p logs eval_results

# Step 1: Merge LoRA (needs peft → structpo-train env)
echo "--- Step 1: Merging LoRA (structpo-train env) ---"
source /home/u5gx/cs2175.u5gx/workspace/share/scripts/activate_env.sh structpo-train
python scripts/merge_lora.py \
    --base "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --output "$MERGED"

# Step 2: Evaluate on MATH-500 + GPQA (needs vLLM → structpo-eval env)
echo "--- Step 2: Evaluating (structpo-eval env) ---"
source /home/u5gx/cs2175.u5gx/workspace/share/scripts/activate_env.sh structpo-eval
[ -f /home/u5gx/cs2175.u5gx/.env ] && set -a && source /home/u5gx/cs2175.u5gx/.env && set +a
python scripts/evaluate.py \
    --model "$MERGED" \
    --benchmarks math500 gpqa \
    --output "eval_results/structpo-4b-beta${BETA}.json"

echo "=== Merge + Eval complete: $(date) ==="
