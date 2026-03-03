#!/bin/bash
# Post-rollout pipeline: annotate rollouts + build structural DPO pairs
# Run after rollout generation completes.
# Usage: bash scripts/post_rollout.sh [rollout_json] [output_json]

set -euo pipefail

WORKDIR=/home/u5gx/cs2175.u5gx/workspace/StructPO
ROLLOUTS=${1:-$WORKDIR/data/rollouts/4b_dse_rollouts.json}
OUTPUT=${2:-$WORKDIR/data/structural_pairs/structural_dpo_pairs.json}

export MAMBA_ROOT_PREFIX=/home/u5gx/cs2175.u5gx/micromamba
eval "$(/home/u5gx/cs2175.u5gx/bin/micromamba shell hook -s bash)"
micromamba activate comprl

cd $WORKDIR

echo "=== Post-Rollout Pipeline ==="
echo "Rollouts: $ROLLOUTS"
echo "Output: $OUTPUT"

if [ ! -f "$ROLLOUTS" ]; then
    echo "ERROR: Rollouts file not found: $ROLLOUTS"
    exit 1
fi

# Step 1: Annotate + build pairs
python scripts/annotate_and_build_pairs.py \
    --rollouts "$ROLLOUTS" \
    --output "$OUTPUT"

echo ""
echo "=== Pairs ready for DPO training ==="
echo "To train: sbatch configs/slurm/train_dpo_4b.sh"
