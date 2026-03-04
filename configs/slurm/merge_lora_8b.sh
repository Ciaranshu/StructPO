#!/bin/bash
#SBATCH --job-name=structpo-merge8b
#SBATCH --account=brics.u5gx
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=115000
#SBATCH --time=00:30:00
#SBATCH --time-min=00:10:00
#SBATCH --output=logs/merge_8b_%j.out
#SBATCH --error=logs/merge_8b_%j.err

# ============================================================
# StructPO: Merge 8B LoRA adapter into base model
# Usage:
#   sbatch merge_lora_8b.sh                          # defaults
#   sbatch merge_lora_8b.sh <base> <adapter> <output>
# ============================================================

set -euo pipefail

BASE_MODEL=${1:-/home/u5gx/cs2175.u5gx/workspace/CompRL/hpc/models/Qwen3-8B}
ADAPTER=${2:-models/structpo-qwen3-8b-stage2-lora}
OUTPUT=${3:-models/structpo-qwen3-8b-stage2-merged}
WORKDIR=/home/u5gx/cs2175.u5gx/workspace/StructPO

# --- Environment ---
source /home/u5gx/cs2175.u5gx/workspace/share/scripts/activate_env.sh structpo-train

echo "=== StructPO 8B LoRA Merge ==="
echo "Base: $BASE_MODEL"
echo "Adapter: $ADAPTER"
echo "Output: $OUTPUT"
echo "Node: $(hostname)"
echo "Start: $(date)"

cd $WORKDIR
mkdir -p logs

python scripts/merge_lora.py \
    --base "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --output "$OUTPUT"

echo "=== 8B Merge complete: $(date) ==="
