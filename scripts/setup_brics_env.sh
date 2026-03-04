#!/bin/bash
# ============================================================
# StructPO: BriCS (Isambard-AI) Environment Setup
# ============================================================
# Run this ONCE on the login node to create the structpo environment.
#
# Usage:
#   bash scripts/setup_brics_env.sh
#
# Prerequisites:
#   - micromamba already installed at /home/u5gx/cs2175.u5gx/bin/micromamba
#   - MAMBA_ROOT_PREFIX=/home/u5gx/cs2175.u5gx/micromamba
# ============================================================

set -euo pipefail

export MAMBA_ROOT_PREFIX=/home/u5gx/cs2175.u5gx/micromamba
MICROMAMBA=/home/u5gx/cs2175.u5gx/bin/micromamba

echo "=== StructPO BriCS Environment Setup ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Arch: $(uname -m)"

# --- Step 1: Create conda environment ---
echo ""
echo "[1/5] Creating structpo environment (Python 3.11, aarch64)..."
$MICROMAMBA create -n structpo python=3.11 -c conda-forge -y 2>/dev/null || \
    echo "  Environment 'structpo' already exists, skipping create."

# Activate
eval "$($MICROMAMBA shell hook -s bash)"
micromamba activate structpo

echo "  Python: $(python --version)"
echo "  Path: $(which python)"

# --- Step 2: Install PyTorch with CUDA (aarch64) ---
echo ""
echo "[2/5] Installing PyTorch with CUDA 12.6..."
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu126

python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# --- Step 3: Install training dependencies ---
echo ""
echo "[3/5] Installing training dependencies (LLaMA-Factory, DeepSpeed, PEFT)..."
pip install --quiet \
    "llamafactory==0.9.3" \
    "deepspeed==0.16.9" \
    "peft>=0.12" \
    "accelerate>=0.34" \
    "datasets>=2.16" \
    "pydantic>=2.0" \
    "numpy>=1.24" \
    "tqdm>=4.65" \
    "matplotlib>=3.7"

# --- Step 4: Install StructPO package ---
echo ""
echo "[4/5] Installing StructPO package..."
cd /home/u5gx/cs2175.u5gx/workspace/StructPO
pip install --quiet -e .

# --- Step 5: Verify ---
echo ""
echo "[5/5] Verification..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')

import pydantic
print(f'  pydantic: {pydantic.__version__}')

from structpo.structural_parser.reachability import full_structural_analysis
r = full_structural_analysis('Let me compute 2+2.\n\n2+2=4.\n\nLet me verify: 2+2=4. Yes.\n\nSo the answer is \\\boxed{4}.')
print(f'  Structural parser: {r[\"num_steps\"]} steps, DSR={r[\"dsr\"]:.0%}')

print('  All checks passed!')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate in future sessions:"
echo "  export MAMBA_ROOT_PREFIX=/home/u5gx/cs2175.u5gx/micromamba"
echo "  eval \"\$(/home/u5gx/cs2175.u5gx/bin/micromamba shell hook -s bash)\""
echo "  micromamba activate structpo"
echo ""
echo "NOTE: vLLM must be installed in a SEPARATE environment (requires pydantic v2"
echo "      and may conflict with LLaMA-Factory). Create 'structpo-eval' for inference."
