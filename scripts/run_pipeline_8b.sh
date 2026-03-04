#!/bin/bash
# ============================================================
# StructPO 8B: Full pipeline with Slurm job dependencies
# Chains: SFT → Merge → Rollout → Annotate → DPO → Merge → Eval
#
# Usage:
#   bash scripts/run_pipeline_8b.sh              # Full pipeline
#   bash scripts/run_pipeline_8b.sh <sft_job_id> # Skip SFT, start from merge
# ============================================================

set -euo pipefail

WORKDIR=/home/u5gx/cs2175.u5gx/workspace/StructPO
cd $WORKDIR

SFT_JOB=${1:-}

echo "=== StructPO 8B Full Pipeline ==="
echo "SFT job dependency: ${SFT_JOB:-will submit new}"

# --- Step 1: Stage 1 SFT (if no existing job) ---
if [ -z "$SFT_JOB" ]; then
    SFT_JOB=$(sbatch --parsable configs/slurm/train_sft_8b.sh)
    echo "Submitted 8B SFT: $SFT_JOB"
fi

# --- Step 2: Merge Stage 1 LoRA (needed for vLLM rollouts) ---
# We need a merge script for 8B Stage 1
MERGE_S1_JOB=$(sbatch --parsable --dependency=afterok:${SFT_JOB} \
    configs/slurm/merge_lora_8b.sh \
    /home/u5gx/cs2175.u5gx/workspace/CompRL/hpc/models/Qwen3-8B \
    models/structpo-qwen3-8b-stage1-lora \
    models/structpo-qwen3-8b-stage1-merged)
echo "Submitted 8B Stage 1 merge: $MERGE_S1_JOB (after SFT $SFT_JOB)"

# --- Step 3: Generate rollouts from Stage 1 model ---
ROLL_JOB=$(sbatch --parsable --dependency=afterok:${MERGE_S1_JOB} \
    configs/slurm/generate_rollouts_8b.sh)
echo "Submitted 8B rollouts: $ROLL_JOB (after merge $MERGE_S1_JOB)"

# --- Step 4: Annotate + build DPO pairs (CPU, inline after rollouts) ---
# This step runs quickly on CPU; we chain a short GPU job that does annotation
# then starts DPO. For simplicity, the DPO script should check if pairs exist.
echo ""
echo "NOTE: After rollout job $ROLL_JOB completes, run on login node:"
echo "  python scripts/annotate_and_build_pairs.py \\"
echo "    --rollouts data/rollouts/8b_dse_rollouts.json \\"
echo "    --output data/structural_pairs/8b_structural_dpo_pairs.json"
echo ""
echo "Then submit DPO:"
echo "  sbatch configs/slurm/train_dpo_8b.sh"
echo ""

# --- Step 5: Stage 1 baseline eval (parallel with rollouts) ---
EVAL_S1_MATH=$(sbatch --parsable --dependency=afterok:${MERGE_S1_JOB} \
    configs/slurm/evaluate.sh \
    models/structpo-qwen3-8b-stage1-merged \
    8b_dse_sft \
    math500)
EVAL_S1_GPQA=$(sbatch --parsable --dependency=afterok:${MERGE_S1_JOB} \
    configs/slurm/evaluate.sh \
    models/structpo-qwen3-8b-stage1-merged \
    8b_dse_sft_gpqa \
    gpqa)
echo "Submitted 8B Stage 1 eval: MATH=$EVAL_S1_MATH GPQA=$EVAL_S1_GPQA"

echo ""
echo "=== 8B Pipeline Summary ==="
echo "  SFT:        $SFT_JOB"
echo "  Merge S1:   $MERGE_S1_JOB  (after $SFT_JOB)"
echo "  Rollouts:   $ROLL_JOB  (after $MERGE_S1_JOB)"
echo "  Eval MATH:  $EVAL_S1_MATH  (after $MERGE_S1_JOB)"
echo "  Eval GPQA:  $EVAL_S1_GPQA  (after $MERGE_S1_JOB)"
echo ""
echo "Manual steps after rollouts:"
echo "  1. python scripts/annotate_and_build_pairs.py (CPU)"
echo "  2. sbatch configs/slurm/train_dpo_8b.sh"
echo "  3. Merge 8B Stage 2 LoRA"
echo "  4. Eval 8B Stage 2"
echo ""
echo "Monitor: squeue --me"
