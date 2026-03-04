#!/bin/bash
# ============================================================
# StructPO 4B: Automated pipeline with Slurm job dependencies
# Chains: merge_lora → eval_math500 → eval_gpqa
#
# Usage:
#   bash scripts/run_pipeline_4b.sh [train_job_id]
#
# If train_job_id is provided, the pipeline waits for it to
# finish successfully before starting the merge step.
# ============================================================

set -euo pipefail

WORKDIR=/home/u5gx/cs2175.u5gx/workspace/StructPO
cd $WORKDIR

TRAIN_JOB=${1:-}
MERGED_MODEL=models/structpo-qwen3-4b-stage2-merged

echo "=== StructPO 4B Pipeline ==="
echo "Train job dependency: ${TRAIN_JOB:-none}"

# --- Step 1: Merge LoRA ---
if [ -n "$TRAIN_JOB" ]; then
    MERGE_JOB=$(sbatch --parsable --dependency=afterok:${TRAIN_JOB} \
        configs/slurm/merge_lora_4b.sh)
else
    MERGE_JOB=$(sbatch --parsable configs/slurm/merge_lora_4b.sh)
fi
echo "Submitted merge job: $MERGE_JOB (depends on: ${TRAIN_JOB:-none})"

# --- Step 2: Eval MATH-500 (depends on merge) ---
EVAL_MATH_JOB=$(sbatch --parsable --dependency=afterok:${MERGE_JOB} \
    configs/slurm/evaluate.sh \
    "$MERGED_MODEL" "4b_structpo_stage2" "math500")
echo "Submitted MATH-500 eval job: $EVAL_MATH_JOB (depends on: $MERGE_JOB)"

# --- Step 3: Eval GPQA (depends on merge, parallel with MATH-500) ---
EVAL_GPQA_JOB=$(sbatch --parsable --dependency=afterok:${MERGE_JOB} \
    configs/slurm/evaluate.sh \
    "$MERGED_MODEL" "4b_structpo_stage2_gpqa" "gpqa")
echo "Submitted GPQA eval job: $EVAL_GPQA_JOB (depends on: $MERGE_JOB)"

echo ""
echo "=== Pipeline submitted ==="
echo "  Merge:      $MERGE_JOB  (after train ${TRAIN_JOB:-immediate})"
echo "  MATH-500:   $EVAL_MATH_JOB  (after merge $MERGE_JOB)"
echo "  GPQA:       $EVAL_GPQA_JOB  (after merge $MERGE_JOB)"
echo ""
echo "Monitor: squeue -u \$(whoami)"
echo "Cancel all: scancel $MERGE_JOB $EVAL_MATH_JOB $EVAL_GPQA_JOB"
