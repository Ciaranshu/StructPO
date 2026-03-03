#!/bin/bash
# ============================================================
# StructPO: Download Models to BriCS SCRATCH
# ============================================================
# Run on login node. Models go to SCRATCHDIR for space.
# Then symlink into workspace/StructPO/models/.
#
# Usage:
#   bash scripts/download_models.sh
# ============================================================

set -euo pipefail

WORKDIR=/home/u5gx/cs2175.u5gx/workspace/StructPO
SCRATCH=${SCRATCHDIR:-/scratch/u5gx/cs2175.u5gx}
MODEL_STORE=$SCRATCH/models

echo "=== StructPO Model Download ==="
echo "Model store: $MODEL_STORE"
echo "Workspace: $WORKDIR"

mkdir -p "$MODEL_STORE"
mkdir -p "$WORKDIR/models"

# --- Activate environment ---
export MAMBA_ROOT_PREFIX=/home/u5gx/cs2175.u5gx/micromamba
eval "$(/home/u5gx/cs2175.u5gx/bin/micromamba shell hook -s bash)"
micromamba activate structpo

# --- Download Qwen3-8B base model ---
echo ""
echo "[1/3] Downloading Qwen3-8B..."
if [ -d "$MODEL_STORE/Qwen3-8B" ]; then
    echo "  Already exists, skipping."
else
    huggingface-cli download Qwen/Qwen3-8B --local-dir "$MODEL_STORE/Qwen3-8B"
fi
ln -sfn "$MODEL_STORE/Qwen3-8B" "$WORKDIR/models/Qwen3-8B"

# --- Download Stage 1 checkpoint (4B, from DecoR) ---
echo ""
echo "[2/3] Downloading decor-qwen3-4b-dse (Stage 1 pretrained)..."
if [ -d "$MODEL_STORE/decor-qwen3-4b-dse" ]; then
    echo "  Already exists, skipping."
else
    huggingface-cli download Ciaranshu/decor-qwen3-4b-dse --local-dir "$MODEL_STORE/decor-qwen3-4b-dse"
fi
ln -sfn "$MODEL_STORE/decor-qwen3-4b-dse" "$WORKDIR/models/decor-qwen3-4b-dse"

# --- Download baseline (4B, original) ---
echo ""
echo "[3/3] Downloading decor-qwen3-4b-original (baseline)..."
if [ -d "$MODEL_STORE/decor-qwen3-4b-original" ]; then
    echo "  Already exists, skipping."
else
    huggingface-cli download Ciaranshu/decor-qwen3-4b-original --local-dir "$MODEL_STORE/decor-qwen3-4b-original"
fi
ln -sfn "$MODEL_STORE/decor-qwen3-4b-original" "$WORKDIR/models/decor-qwen3-4b-original"

echo ""
echo "=== All models downloaded ==="
echo "Symlinks in $WORKDIR/models/:"
ls -la "$WORKDIR/models/"
