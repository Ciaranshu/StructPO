#!/bin/bash
#SBATCH --job-name=l3-parse
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=115000
#SBATCH --time=1:00:00
#SBATCH --output=logs/l3_parse_%j.out
#SBATCH --error=logs/l3_parse_%j.err

# ============================================================
# L3-base: Qwen3-0.5B zero-shot parsing of GPQA rollouts
#
# Usage:
#   sbatch --account=COLLIER-SL3-GPU \
#     configs/slurm/run_l3_parser_1hr.sh
# ============================================================

set -euo pipefail

WORKDIR=/home/cs2175/rds/workspace/StructPO
ROLLOUTS=${ROLLOUTS:-data/rollouts/gpqa_4b_rollouts.json}
OUTPUT=${OUTPUT:-data/l3_annotations/gpqa_4b_l3_base.json}
MODEL=${L3_MODEL:-Qwen/Qwen3-0.5B}
MAX_TRACES=${MAX_TRACES:-0}

source /home/cs2175/rds/workspace/share/scripts/activate_env.sh structpo-train

echo "=== L3-base Parser: Qwen3-0.5B Zero-Shot ==="
echo "Model: $MODEL"
echo "Rollouts: $ROLLOUTS"
echo "Output: $OUTPUT"
echo "Max traces: $MAX_TRACES (0=all)"
echo "Node: $(hostname), Job: $SLURM_JOB_ID"
echo "Start: $(date)"
nvidia-smi --list-gpus

cd $WORKDIR
mkdir -p logs data/l3_annotations

PYTHONPATH=src:${PYTHONPATH:-} python scripts/analysis/l3_annotate_rollouts.py \
    --rollouts "$ROLLOUTS" \
    --output "$OUTPUT" \
    --model "$MODEL" \
    --max-traces $MAX_TRACES

echo "=== L3 parsing complete: $(date) ==="
