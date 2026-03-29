#!/bin/bash
#SBATCH --job-name=hint-pilot
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=115000
#SBATCH --time=1:00:00
#SBATCH --output=logs/hint_pilot_%j.out
#SBATCH --error=logs/hint_pilot_%j.err

# ============================================================
# Hint Injection Pilot: Can hints expand the training zone?
#
# Tests whether hints can make "all-wrong" problems solvable,
# expanding the effective training zone for RL + StructPRM.
#
# Usage:
#   sbatch --account=COLLIER-SL3-GPU configs/slurm/hint_pilot_1hr.sh
# ============================================================

set -euo pipefail

WORKDIR=/home/cs2175/rds/workspace/StructPO

source /home/cs2175/rds/workspace/share/scripts/activate_env.sh structpo-eval

echo "=== Hint Injection Pilot ==="
echo "Node: $(hostname), Job: $SLURM_JOB_ID"
echo "Start: $(date)"
nvidia-smi --list-gpus

cd $WORKDIR
mkdir -p logs data/hint_pilot

PYTHONPATH=src:${PYTHONPATH:-} python scripts/analysis/hint_injection_pilot.py \
    --rollouts data/rollouts/math500_4b_rollouts.json \
    --model models/decor-qwen3-4b-dse \
    --output data/hint_pilot/results.json \
    --num-rollouts 4 \
    --hint-type both

echo "=== Hint pilot complete: $(date) ==="
