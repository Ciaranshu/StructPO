#!/bin/bash
#SBATCH --job-name=rt-dag-exp
#SBATCH --account=brics.u5gx
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=115000
#SBATCH --time=4:00:00
#SBATCH --time-min=1:00:00
#SBATCH --output=logs/rt_dag_phase_b_%j.out
#SBATCH --error=logs/rt_dag_phase_b_%j.err

# ============================================================
# Real-Time DAG Feedback Experiment — Phase B
# Compare baseline vs structurally-guided generation on MATH-500
# ============================================================

set -euo pipefail

WORKDIR=/home/u5gx/cs2175.u5gx/workspace/StructPO
MODEL=${MODEL:-models/decor-qwen3-4b-dse}
SUBSET=${SUBSET:-100}

# --- Environment ---
module load cuda/12.6
export MAMBA_ROOT_PREFIX=/home/u5gx/cs2175.u5gx/micromamba
eval "$(/home/u5gx/cs2175.u5gx/bin/micromamba shell hook -s bash)"
micromamba activate structpo-eval

[ -f /home/u5gx/cs2175.u5gx/.env ] && set -a && source /home/u5gx/cs2175.u5gx/.env && set +a

echo "=== RT-DAG Phase B Experiment ==="
echo "Model: $MODEL"
echo "Subset: $SUBSET problems"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
nvidia-smi --list-gpus

cd $WORKDIR
mkdir -p logs eval_results

PYTHONPATH=src python scripts/experiment_rt_dag.py \
    --phase B \
    --model "$MODEL" \
    --output eval_results/rt_dag_phase_b.json \
    --subset "$SUBSET"

echo "=== Phase B complete: $(date) ==="
