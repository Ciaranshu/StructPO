#!/bin/bash
# ============================================================
# StructPO: BriCS Evaluation Environment Setup (vLLM)
# ============================================================
# Separate env for inference/rollout generation because vLLM
# has dependency conflicts with LLaMA-Factory (pydantic versions).
#
# Usage:
#   bash scripts/setup_brics_eval_env.sh
# ============================================================

set -euo pipefail

export MAMBA_ROOT_PREFIX=/home/u5gx/cs2175.u5gx/micromamba
MICROMAMBA=/home/u5gx/cs2175.u5gx/bin/micromamba

echo "=== StructPO Eval Environment Setup ==="
echo "Date: $(date)"

# --- Create environment ---
echo "[1/4] Creating structpo-eval environment..."
$MICROMAMBA create -n structpo-eval python=3.11 -c conda-forge -y 2>/dev/null || \
    echo "  Environment 'structpo-eval' already exists, skipping create."

eval "$($MICROMAMBA shell hook -s bash)"
micromamba activate structpo-eval

# --- Install PyTorch CUDA first ---
echo "[2/4] Installing PyTorch with CUDA 12.6..."
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu126

# --- Install vLLM ---
echo "[3/4] Installing vLLM..."
pip install --quiet "vllm>=0.11.0"

# IMPORTANT: vLLM may downgrade torch to CPU version. Reinstall CUDA torch.
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu126

# Install StructPO for answer extraction utilities
pip install --quiet "pydantic>=2.0" "numpy>=1.24" "tqdm>=4.65"
cd /home/u5gx/cs2175.u5gx/workspace/StructPO
pip install --quiet -e .

# --- Verify ---
echo "[4/4] Verification..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import vllm
print(f'  vLLM: {vllm.__version__}')
print('  All checks passed!')
"

echo ""
echo "=== Eval environment ready ==="
echo "Activate with: micromamba activate structpo-eval"
