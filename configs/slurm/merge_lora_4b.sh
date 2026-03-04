#!/bin/bash
#SBATCH --job-name=structpo-merge4b
#SBATCH --account=brics.u5gx
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=115000
#SBATCH --time=00:30:00
#SBATCH --time-min=00:10:00
#SBATCH --output=logs/merge_4b_%j.out
#SBATCH --error=logs/merge_4b_%j.err

# ============================================================
# StructPO: Merge LoRA adapter into base model
# Runs on CPU (uses GPU allocation for billing efficiency)
# ============================================================

set -euo pipefail

BASE_MODEL=${1:-models/decor-qwen3-4b-dse}
ADAPTER=${2:-models/structpo-qwen3-4b-stage2}
OUTPUT=${3:-models/structpo-qwen3-4b-stage2-merged}
WORKDIR=/home/u5gx/cs2175.u5gx/workspace/StructPO

# --- Environment ---
module load cuda/12.6
export MAMBA_ROOT_PREFIX=/home/u5gx/cs2175.u5gx/micromamba
eval "$(/home/u5gx/cs2175.u5gx/bin/micromamba shell hook -s bash)"
micromamba activate structpo-train

echo "=== StructPO LoRA Merge ==="
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

echo "=== Merge complete: $(date) ==="
