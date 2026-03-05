#!/bin/bash
#SBATCH --job-name=structpo-meval8b
#SBATCH --account=brics.u5gx
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=115000
#SBATCH --time=4:00:00
#SBATCH --time-min=1:00:00
#SBATCH --output=logs/merge_eval_8b_%j.out
#SBATCH --error=logs/merge_eval_8b_%j.err

set -euo pipefail

ADAPTER_NAME=${1:?Usage: sbatch merge_eval_8b_generic.sh <adapter_dir_name> <output_name>}
OUTPUT_NAME=${2:-$ADAPTER_NAME}
WORKDIR=/home/u5gx/cs2175.u5gx/workspace/StructPO
BASE_MODEL=$WORKDIR/models/structpo-qwen3-8b-stage1-merged
ADAPTER=$WORKDIR/models/$ADAPTER_NAME
MERGED=$WORKDIR/models/${ADAPTER_NAME}-merged

echo "=== 8B Merge + Eval: $ADAPTER_NAME ==="
echo "Base: $BASE_MODEL"
echo "Adapter: $ADAPTER"
echo "Merged: $MERGED"
echo "Output: eval_results/${OUTPUT_NAME}.json"
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
    --output "eval_results/${OUTPUT_NAME}.json"

echo "=== 8B Merge + Eval complete: $(date) ==="
