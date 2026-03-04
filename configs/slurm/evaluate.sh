#!/bin/bash
#SBATCH --job-name=structpo-eval
#SBATCH --account=brics.u5gx
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=115000
#SBATCH --time=4:00:00
#SBATCH --time-min=1:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# ============================================================
# StructPO: Evaluate model on MATH-500 (+ optionally GPQA)
# Target: Isambard-AI Phase 2 (BriCS) — GH200, aarch64
# ============================================================

set -euo pipefail

MODEL=${1:?Usage: sbatch evaluate.sh <model_path> [output_name] [benchmarks]}
OUTPUT_NAME=${2:-$(basename $MODEL)}
BENCHMARKS=${3:-math500}
WORKDIR=/home/u5gx/cs2175.u5gx/workspace/StructPO

# --- Environment (structpo-eval: vLLM + pydantic v2) ---
module load cuda/12.6
export MAMBA_ROOT_PREFIX=/home/u5gx/cs2175.u5gx/micromamba
eval "$(/home/u5gx/cs2175.u5gx/bin/micromamba shell hook -s bash)"
micromamba activate structpo-eval

echo "=== StructPO Evaluation ==="
echo "Model: $MODEL"
echo "Benchmarks: $BENCHMARKS"
echo "Output: eval_results/${OUTPUT_NAME}.json"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
nvidia-smi --list-gpus

cd $WORKDIR
mkdir -p logs eval_results

python scripts/evaluate.py \
    --model "$MODEL" \
    --benchmarks $BENCHMARKS \
    --output "eval_results/${OUTPUT_NAME}.json"

echo "=== Evaluation complete: $(date) ==="
