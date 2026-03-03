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
│   │   ├── rollout_gen.py    # Generate rollouts from Stage 1 model
│   │   ├── annotator.py      # Structural annotation pipeline
│   │   └── pair_builder.py   # Build DPO preference pairs
│   └── training/             # Training utilities
│       └── train_entry.py    # LLaMA-Factory training entry point
├── configs/
│   ├── sft/                  # Stage 1 SFT configs
│   ├── dpo/                  # Stage 2 DPO configs (to be created)
│   ├── deepspeed/            # DeepSpeed ZeRO configs
│   └── slurm/                # HPC job scripts
├── data/
│   ├── limo_cleaned/         # DSE-cleaned LIMO datasets (5 variants)
│   └── eval_results/         # 4B evaluation results from DecoR
├── scripts/                  # Utility scripts
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

## Stage 1 Results (Already Validated)

| Model | MATH-500 | GPQA | Avg Trace Length | Training Time |
|:------|:--------:|:----:|:----------------:|:-------------:|
| Qwen3-4B + Original SFT | 69.6% | 55.6% | 18.7k chars | baseline |
| Qwen3-4B + DSE-SFT | **72.8%** | 49.0% | 15.3k chars | -28% |

Stage 1 improves math but loses GPQA deliberation → **Stage 2 (Structural DPO) aims to recover GPQA while keeping math gains**.

## Compute Plan (BriCS NHR)

| Phase | GPU-hrs | Timeline |
|:------|--------:|:---------|
| Stage 1: DSE-SFT (8B + 14B) | 60 | Day 3-4 |
| Rollout generation (60k traces) | 80 | Day 5-7 |
| Stage 2: Structural DPO (3 configs) | 125 | Day 8-11 |
| Evaluation | 100 | Day 11-13 |
| Analysis + Stage 3 | 90 | Day 13-17 |
| DecoR 14B scaling (parallel) | 200 | Day 3-15 |
| Reserve | ~1245 | — |
| **Total budget** | **2400** | **20 days** |

## Related Work & Differentiation

| Approach | Signal | Method | Limitation |
|:---------|:-------|:-------|:-----------|
| Prune-on-Logic | Perplexity | DAG + pruning → SFT | No preference learning; no domain analysis |
| BRIDGE | Length | 3-stage curriculum + GRPO | Penalizes ALL long traces, hurts GPQA |
| DeCS | Length | Decoupled rewards + curriculum | Length ≠ structural quality |
| Step-DPO | Correctness | Step-level DPO | Correct steps can still be dead (our F2) |
| **StructPO** | **DAG reachability** | **Structural preference DPO** | **Preserves useful deliberation** |

## Getting Started (BriCS Quickstart)

### 1. Clone and install

```bash
git clone https://github.com/Ciaranshu/StructPO.git
cd StructPO
pip install -e .
```

### 2. Download Stage 1 model from HuggingFace

```bash
# Stage 1 checkpoint (DSE-SFT, required for rollout generation)
huggingface-cli download Ciaranshu/decor-qwen3-4b-dse --local-dir models/decor-qwen3-4b-dse

# Baseline (for comparison experiments)
huggingface-cli download Ciaranshu/decor-qwen3-4b-original --local-dir models/decor-qwen3-4b-original
```

### 3. Full Pipeline (Stage 2: Structural Preference DPO)

```bash
# Step 1: Generate rollouts from Stage 1 model (needs GPU + vllm)
python scripts/generate_rollouts.py \
    --model models/decor-qwen3-4b-dse \
    --dataset data/limo_cleaned/limo_original.json \
    --num-rollouts 8 \
    --temperature 0.7 \
    --output data/rollouts/stage1_rollouts.json

# Step 2: Annotate traces + build structural preference pairs
python scripts/annotate_and_build_pairs.py \
    --rollouts data/rollouts/stage1_rollouts.json \
    --output data/structural_pairs/structural_dpo_pairs.json

# Step 3: Train Stage 2 DPO (needs 2× GPU)
deepspeed --num_gpus=2 src/training/train_entry.py \
    configs/dpo/qwen3_4b_structural_dpo.yaml

# Or via Slurm:
sbatch configs/slurm/train_dpo.sh
```

### 4. Quick smoke test (no GPU needed)

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

### Environment notes

- **Training env**: Python 3.10+, PyTorch 2.5+, LLaMA-Factory 0.9.3, DeepSpeed 0.16.9
- **Eval env**: Separate env with vLLM ≥ 0.11.0 (needs PyTorch ≥ 2.6)
- **Structural analysis**: Only needs `pydantic` — runs on CPU, <10ms/trace

### Available models on HuggingFace

| Model | HuggingFace | Description |
|:------|:------------|:------------|
| `decor-qwen3-4b-dse` | [Ciaranshu/decor-qwen3-4b-dse](https://huggingface.co/Ciaranshu/decor-qwen3-4b-dse) | Stage 1 checkpoint (MATH +3.2%, training −28%) |
| `decor-qwen3-4b-original` | [Ciaranshu/decor-qwen3-4b-original](https://huggingface.co/Ciaranshu/decor-qwen3-4b-original) | Baseline (unmodified LIMO SFT) |

### Data included in repo

- `data/limo_cleaned/limo_dse.json` — 817 DSE-cleaned LIMO samples (12MB)
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
