# StructPO: Structural Preference Optimization for Reasoning Quality

> *Learning When to Explore: Using DAG Reachability as Preference Signal for Reasoning*

## Overview

StructPO addresses a fundamental limitation of outcome-based RL for reasoning: **the Structural Reward Gap**. 

Standard RLVR (e.g., GRPO) assigns the same reward to all tokens in a trace based solely on whether the final answer is correct. Our empirical analysis (DecoR, COLM 2026) reveals that **dead steps** — reasoning steps unreachable from the conclusion via dependency DAG — are **correctness-independent** (appear equally in correct and incorrect traces). This means RL's implicit credit assignment is *blind* to structural waste, leading to phenomena like **Verification Theater** (5.6× more dead verification in RL-trained models).

StructPO proposes a **three-stage curriculum** to teach models exploration quality:

| Stage | Goal | Method | Status |
|:------|:-----|:-------|:-------|
| **Stage 1** | Learn efficient reasoning | SFT on DSE-cleaned data | ✅ Validated (MATH +3.2%) |
| **Stage 2** | Learn *when* to explore | Structural Preference DPO | 🔬 Core contribution |
| **Stage 3** | Learn *how* to explore | Difficulty-adaptive curriculum | 📋 Planned |

## Key Idea: Structural Preference Pairs

Unlike prior work that uses **length** or **perplexity** as preference signals, we use **DAG reachability**:

```
For each reasoning trace:
  1. Segment into paragraphs (reasoning steps)
  2. Classify step types (verification, computation, exploration, ...)
  3. Build dependency DAG (sequential + content-overlap edges)
  4. Backward reachability from conclusion → live/dead per step
  5. Compute Dead Step Ratio (DSR)

Preference pairs:
  Preferred:  correct + low DSR (structurally efficient)
  Rejected:   correct + high DSR (structurally wasteful)
```

**Why not length?** A short trace can be all dead steps (wrong derivation chain). A long trace can be all live steps (productive cross-domain deliberation on GPQA). **Structure ≠ Length.**

## Project Structure

```
StructPO/
├── src/
│   ├── structural_parser/    # Fast regex-based step classifier + DAG builder
│   │   ├── classifier.py     # Step type classification (regex, <10ms/trace)
│   │   ├── dag_builder.py    # Dependency DAG construction
│   │   └── reachability.py   # Backward reachability analysis (live/dead)
│   ├── dse/                  # Dead Step Elimination (from DecoR)
│   │   ├── schemas.py        # R-IR schema definitions
│   │   └── dse_core.py       # DSE algorithm
│   ├── preference_builder/   # Stage 2: Structural preference pair construction
│   │   ├── annotator.py      # Structural annotation pipeline
│   │   └── pair_builder.py   # Build DPO preference pairs (4 types)
│   └── training/             # Training utilities
│       └── train_entry.py    # LLaMA-Factory training entry point
├── configs/
│   ├── sft/                  # Stage 1 SFT configs (4B full, 8B LoRA)
│   ├── dpo/                  # Stage 2 DPO configs (4B full, 8B LoRA)
│   ├── deepspeed/            # DeepSpeed ZeRO-2/3 configs
│   └── slurm/                # BriCS Slurm job scripts
├── data/
│   ├── limo_cleaned/         # DSE-cleaned LIMO datasets (original + DSE)
│   └── structural_pairs/     # Stage 2 DPO preference pairs (generated)
├── scripts/                  # Setup, download, pipeline scripts
├── experiments/              # Analysis & evaluation scripts
├── docs/                     # Design documents & analysis
└── requirements.txt
```

## Empirical Foundation (from DecoR)

| Finding | Implication for StructPO |
|:--------|:------------------------|
| F2: Dead steps are correctness-independent | GRPO's implicit PRM cannot learn to eliminate them |
| F3: RL produces 5.6× more dead verification | Outcome reward causes "Verification Theater" |
| F5: Efficient agents have low dead action ratio | Structural credit extends to agent trajectories |
| F6: Less exploration → higher accuracy (math) | DSE-SFT is a valid Stage 1 |
| F7: More verification → higher accuracy (GPQA) | Domain-dependent exploration value → Stage 2 must preserve useful deliberation |

## Stage 1 Results (Already Validated at 4B)

| Model | MATH-500 | GPQA | Avg Trace Length | Training Time |
|:------|:--------:|:----:|:----------------:|:-------------:|
| Qwen3-4B + Original SFT | 69.6% | 55.6% | 18.7k chars | baseline |
| Qwen3-4B + DSE-SFT | **72.8%** | 49.0% | 15.3k chars | -28% |

Stage 1 improves math but loses GPQA deliberation → **Stage 2 (Structural DPO) aims to recover GPQA while keeping math gains**.

## Compute Plan (BriCS — Isambard-AI Phase 2)

**Budget: 500 NHR (~2000 GPU-hours on GH200)**

| Phase | NHR | GPU-hrs equiv | Timeline |
|:------|----:|:-------------|:---------|
| Stage 1: DSE-SFT 8B (LoRA, 2 GPU) | 15 | 60h | Day 3-4 |
| Rollout generation (vLLM, 1 GPU) | 10 | 40h | Day 5-6 |
| Structural annotation (CPU only) | 0 | — | Day 6 |
| Stage 2: Structural DPO × 3 configs | 20 | 80h | Day 7-10 |
| Evaluation (4 benchmarks × models) | 15 | 60h | Day 10-12 |
| Ablations + Stage 3 (if time) | 20 | 80h | Day 12-16 |
| Buffer / re-runs | 20 | 80h | — |
| **Total** | **~100** | **~400h** | **~16 days** |

> **Note**: 1 NHR = 4 GPU-hours (1 full node × 1 hour). Using 2 GPUs = 0.5 NHR/hour.
> With 500 NHR budget, we have substantial headroom for re-runs and ablations.

## Related Work & Differentiation

| Approach | Signal | Method | Limitation |
|:---------|:-------|:-------|:-----------|
| Prune-on-Logic | Perplexity | DAG + pruning → SFT | No preference learning; no domain analysis |
| BRIDGE | Length | 3-stage curriculum + GRPO | Penalizes ALL long traces, hurts GPQA |
| DeCS | Length | Decoupled rewards + curriculum | Length ≠ structural quality |
| Step-DPO | Correctness | Step-level DPO | Correct steps can still be dead (our F2) |
| **StructPO** | **DAG reachability** | **Structural preference DPO** | **Preserves useful deliberation** |

## Getting Started (BriCS Quickstart)

### 1. Environment Setup

```bash
# On BriCS login node:
cd /home/u5gx/cs2175.u5gx/workspace/StructPO

# Create training environment (LLaMA-Factory + DeepSpeed)
bash scripts/setup_brics_env.sh

# Create eval environment (vLLM — separate due to pydantic conflict)
bash scripts/setup_brics_eval_env.sh

# Download models to SCRATCH (Qwen3-8B + DecoR 4B checkpoints)
bash scripts/download_models.sh
```

### 2. Activate Environment

```bash
export MAMBA_ROOT_PREFIX=/home/u5gx/cs2175.u5gx/micromamba
eval "$(/home/u5gx/cs2175.u5gx/bin/micromamba shell hook -s bash)"
micromamba activate structpo    # for training
# micromamba activate structpo-eval  # for vLLM inference
```

### 3. Full Pipeline

```bash
# Stage 1: SFT on DSE-cleaned data (8B LoRA, ~15 NHR)
sbatch configs/slurm/train_sft.sh configs/sft/qwen3_8b_dse_sft_lora.yaml

# Generate rollouts from Stage 1 model (vLLM, ~10 NHR)
sbatch configs/slurm/generate_rollouts.sh

# Annotate traces + build structural preference pairs (CPU, no NHR)
python scripts/annotate_and_build_pairs.py \
    --rollouts data/rollouts/stage1_rollouts.json \
    --output data/structural_pairs/structural_dpo_pairs.json

# Stage 2: Structural DPO (8B LoRA, ~7 NHR per config)
sbatch configs/slurm/train_dpo.sh configs/dpo/qwen3_8b_structural_dpo_lora.yaml
```

### 4. Quick Smoke Test (no GPU needed)

```bash
python -c "
from src.structural_parser.reachability import full_structural_analysis
r = full_structural_analysis('''Let me try computing 2+2.

2+2 = 4, that's straightforward.

Wait, let me verify: 2+2 = 4. Yes.

Actually, let me try a different approach entirely.

Hmm, that didn't work.

So the answer is \boxed{4}.''')
print(f'Steps: {r[\"num_steps\"]}, DSR: {r[\"dsr\"]:.0%}, Dead: {r[\"num_dead\"]}')
for s in r['steps']:
    tag = '✓' if s['is_live'] else '✗'
    print(f'  [{tag}] Step {s[\"id\"]}: {s[\"type\"]:15s} ({s[\"char_length\"]:4d} chars)')
"
```

### Environment Notes

- **Training env** (`structpo`): Python 3.11, PyTorch 2.9+ (CUDA 12.6), LLaMA-Factory 0.9.3, DeepSpeed 0.16.9
- **Eval env** (`structpo-eval`): Python 3.11, vLLM ≥ 0.11.0 (separate due to pydantic conflict)
- **Structural analysis**: Only needs `pydantic>=2` — runs on CPU, <10ms/trace
- **BriCS**: ARM aarch64, GH200 96GB HBM3, Slurm `workq` partition, max 24h/job

### Available Configs

| Config | Model | Method | GPUs | Est. Time |
|:-------|:------|:-------|:----:|----------:|
| `sft/qwen3_4b_dse_sft.yaml` | 4B | Full SFT | 2 | ~3h |
| `sft/qwen3_8b_dse_sft_lora.yaml` | 8B | LoRA SFT | 2 | ~8h |
| `dpo/qwen3_4b_structural_dpo.yaml` | 4B | Full DPO | 2 | ~2h |
| `dpo/qwen3_8b_structural_dpo_lora.yaml` | 8B | LoRA DPO | 2 | ~4h |

### Data Included

- `data/limo_cleaned/limo_dse.json` — 817 DSE-cleaned LIMO samples (11MB)
- `data/limo_cleaned/limo_original.json` — 817 original LIMO samples (16MB)
- `data/structural_pairs/` — Placeholder for Stage 2 preference pairs (generated by pipeline)

## Citation

```bibtex
@article{structpo2026,
  title={Learning When to Explore: Structural Preference Optimization for Reasoning Quality},
  author={Shu, Chang},
  year={2026}
}
```

## License

MIT
