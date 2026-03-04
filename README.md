# StructPO: Structural Preference Optimization for Reasoning Quality

> *Learning When to Explore: Using DAG Reachability as Preference Signal for Reasoning*

## Overview

Reasoning models face a fundamental **exploration-exploitation dilemma**. They must explore to solve hard problems, but unguided exploration is the primary source of inefficiency. Prior work either compresses all steps uniformly (Draft-Thinking) or uses step-level correctness (Full-Step-DPO), missing a critical distinction: **not all dead-end exploration is bad**.

Our empirical analysis shows that for hard problems (MATH Level 4-5), **23-30% of correct solutions involve substantial exploration** (DSR ≥ 0.3). The question is not "how to eliminate exploration" but **"how to make exploration productive."**

StructPO uses **DAG reachability analysis** to distinguish *productive* exploration (steps that inform the solution path, even indirectly) from *wasteful* exploration (steps disconnected from the reasoning graph). Our four-type structural preference signal teaches models:

| Signal | What it teaches | Pair Type |
|:-------|:----------------|:----------|
| **When NOT to explore** | On clear problems, derive directly | Type 1: Structural Efficiency |
| **How to verify productively** | Verification should discover, not confirm | Type 2: Productive Exploration |
| **When to abandon dead ends** | Explore with direction, cut losses early | Type 3: Exploration Direction |
| **Which patterns are toxic** | Surgically target wasteful motifs | Type 4: Structural Contrastive |

| Stage | Goal | Method | Status |
|:------|:-----|:-------|:-------|
| **Stage 1** | Learn efficient reasoning | SFT on DSE-cleaned data | ✅ Validated (82.8% MATH-500) |
| **Stage 2** | Learn *when and how* to explore | Structural Preference DPO | 🔬 Core contribution |

## Key Idea: Structural Preference Pairs

Unlike prior work that uses **length** or **perplexity** as preference signals, we use **DAG reachability**:

```
For each reasoning trace:
  1. Segment into paragraphs (reasoning steps)
  2. Classify step types (verification, computation, exploration, ...)
  3. Build dependency DAG (sequential + content-overlap edges)
  4. Backward reachability from conclusion → live/dead per step
  5. Compute Dead Step Ratio (DSR)
  6. Extract structural motifs (dead cascade, verification theater, ...)

Four types of preference pairs (jointly = a complete exploration policy):
  Type 1: correct+low_DSR  >  correct+high_DSR        (efficiency)
  Type 2: high_live_verif  >  low_live_verif           (productive verification)
  Type 3: correct+directed >  incorrect+undirected     (exploration direction)
  Type 4: trace_without_motif > trace_with_motif       (contrastive — motif-level)
```

**Why not length?** 27% of our preferred (chosen) solutions are LONGER than rejected ones. A short trace can be all dead steps (wrong derivation). A long trace can be all live steps (productive deliberation on hard problems). **Structure ≠ Length.**

**Why not correctness?** Dead steps are correctness-independent (Pearson r = 0.011). A correct trace can be full of wasteful detours; an incorrect trace can be structurally clean. DSR measures **reasoning quality**, not **reasoning outcome**.

## Project Structure

```
StructPO/
├── src/structpo/              # Python package (import as `from structpo.xxx`)
│   ├── structural_parser/     # Fast regex-based step classifier + DAG builder
│   │   ├── classifier.py      # Step type classification (regex, <10ms/trace)
│   │   ├── dag_builder.py     # Dependency DAG construction
│   │   ├── reachability.py    # Backward reachability analysis (live/dead)
│   │   └── motif.py           # Structural motif extraction (4 anti-patterns)
│   ├── dse/                   # Dead Step Elimination (from DecoR)
│   │   ├── schemas.py         # R-IR schema definitions
│   │   └── dse_core.py        # DSE algorithm
│   ├── preference_builder/    # Stage 2: Structural preference pair construction
│   │   ├── annotator.py       # Structural annotation pipeline
│   │   ├── pair_builder.py    # Build DPO preference pairs (Type 1-3 + Type 4)
│   │   └── contrastive_builder.py  # Type 4: Motif-level contrastive pairs
│   └── training/              # Training utilities
│       └── train_entry.py     # LLaMA-Factory training entry point
├── configs/
│   ├── sft/                   # Stage 1 SFT configs (4B full, 8B LoRA + control)
│   ├── dpo/                   # Stage 2 DPO configs (4B full, 8B LoRA)
│   ├── deepspeed/             # DeepSpeed ZeRO-2/3 configs
│   └── slurm/                 # BriCS Slurm job scripts (all with --mem=115000)
├── scripts/
│   ├── analysis/              # Exploratory analysis (structural reward gap, DSE, etc.)
│   ├── run_pipeline_4b.sh     # 4B pipeline orchestration
│   ├── run_pipeline_8b.sh     # 8B pipeline orchestration
│   ├── annotate_and_build_pairs.py  # Rollout → preference pairs
│   ├── evaluate.py            # MATH-500 / GPQA evaluation via vLLM
│   └── smoke_test.py          # Full component smoke test
├── data/
│   ├── limo_cleaned/          # DSE-cleaned LIMO datasets (original + DSE)
│   ├── rollouts/              # Model rollouts for pair construction
│   └── structural_pairs/      # Stage 2 DPO preference pairs (Type 1-3: 2,377; Type 4: TBD)
├── docs/
│   ├── phase/                 # NeurIPS experiment plan (5 phases)
│   └── notes/                 # Research notes from ideation phase
├── eval_results/              # Evaluation outputs (gitignored)
└── pyproject.toml
```

## Empirical Foundation (from DecoR + StructPO analysis)

### DecoR Findings → StructPO Motivation

| Finding | Implication for StructPO |
|:--------|:------------------------|
| F2: Dead steps are correctness-independent (r=0.011) | Outcome reward cannot distinguish productive vs wasteful exploration |
| F3: RL produces 5.6× more dead verification | Outcome reward causes "Verification Theater" — superstitious checking |
| F6: Less exploration → higher accuracy (math) | For straightforward problems, exploration is a sign of being lost |
| F7: More verification → higher accuracy (GPQA) | For cross-domain problems, exploration IS the reasoning |

### New StructPO Findings (4B DSE-SFT on MATH-500)

| Finding | Data | Implication |
|:--------|:-----|:------------|
| **Overthinking effect** | Incorrect: 2.1× tokens, 2.3× steps vs correct | Unguided exploration wastes compute and hurts accuracy |
| **Exploration necessity scales with difficulty** | Level 1: 12% correct need high DSR → Level 5: 23% | Hard problems genuinely require exploration |
| **Productive vs wasteful high-DSR** | Correct+HighDSR: 97 steps, dead/live=3.5 vs Incorrect+HighDSR: 219 steps, dead/live=4.3 | Productive exploration is more compact and directed |
| **Not a length preference** | 27% of chosen solutions are longer than rejected | Structure ≠ Length: what matters is connectedness, not brevity |
| **Subject-dependent exploration** | Precalculus: 44% correct need exploration vs Algebra: 18% | Exploration value is domain-dependent, not just difficulty-dependent |

### Stage 1 Results (Validated at 4B)

| Model | MATH-500 | DSR | Avg Tokens | Avg Steps |
|:------|:--------:|:---:|:----------:|:---------:|
| Qwen3-4B + DSE-SFT (Stage 1) | **82.8%** | 22.0% | 3,132 | 71.8 |

Stage 1 teaches efficient reasoning. **Stage 2 (Structural DPO) teaches when and how to explore** — maintaining accuracy while learning a complete exploration policy.

## Compute Plan (BriCS — Isambard-AI Phase 2)

**Budget: ~1,000 NHR (~4,000 GPU-hours on GH200) | Deadline: 2026-03-22**

| Phase | Focus | NHR | Timeline |
|:------|:------|----:|:---------|
| [Phase 1](docs/phase/phase1_foundation.md) | 4B validation + 8B SFT | 40 | Day 1-5 |
| [Phase 2](docs/phase/phase2_structural_dpo.md) | Structural DPO (core) | 120 | Day 5-10 |
| [Phase 3](docs/phase/phase3_ablations.md) | Ablations + baselines | 150 | Day 10-14 |
| [Phase 4](docs/phase/phase4_analysis.md) | Structural behavior analysis | 80 | Day 14-17 |
| [Phase 5](docs/phase/phase5_writing.md) | Writing + final experiments | 100 | Day 17+ |
| CompRL (concurrent) | Agent RL experiments | 250 | Parallel |
| Reserve | Buffer / re-runs | 260 | — |
| **Total** | | **~1,000** | **17 days** |

> **Note**: 1 NHR = 4 GPU-hours (1 full node × 1 hour). Always use 4 GPUs + `--mem=115000`.
> See [`docs/phase/`](docs/phase/) for the detailed experiment plan.

## Related Work & Differentiation

> **Collision check (2026-03-04): No direct collision found.** See `docs/notes/36-competitive-landscape-2026.md`.

| Approach | Signal Type | Method | Why StructPO is different |
|:---------|:------------|:-------|:-------------------------|
| **Length-based methods** | | | |
| Draft-Thinking (Mar 2026) | Length | Progressive SFT+RL | Compresses ALL steps uniformly; StructPO selectively targets dead exploration |
| BRIDGE (Feb 2026) | Length | 3-stage curriculum + GRPO | Penalizes all long traces; StructPO allows long productive exploration |
| O1-Pruner (2025) | Length ratio | PPO + length-harmonizing reward | Length-based pruning; blind to structural quality |
| DAST / CoT-Valve (2025) | Token budget | SimPO / parameter mixing | Controls length, not structure |
| **Correctness-based methods** | | | |
| Full-Step-DPO (ACL 2025) | Correctness (PRM) | Step-level DPO on all steps | Correct steps can be dead (r=0.011); correctness ≠ structural quality |
| Step-DPO (NeurIPS 2024) | Correctness (first error) | Step-level DPO | Dead correct steps are invisible to correctness signals |
| PORT (NAACL 2025) | Outcome correctness | DPO on reasoning traces | Outcome-level signal; no structural analysis |
| Uni-DPO (ICLR 2026) | Data quality + dynamics | Unified DPO framework | Orthogonal — about training dynamics, not signal construction |
| **Graph-based analysis** | | | |
| MAP (EMNLP 2025) | Graph structure | Cluster + directed graph | Analysis-only — no training signal, no reachability, no dead steps |
| Graph of Thoughts (AAAI 2024) | Graph structure | Graph-structured prompting | Inference-time framework, not training signal |
| **Inference-time methods** | | | |
| PRISM (Mar 2026) | PRM score | MCMC refinement at inference | Complementary (inference-time); StructPO is training-time |
| **StructPO (ours)** | **DAG reachability** | **Structural preference DPO** | **Teaches exploration policy: when, how, and when to stop** |

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
# Option A: Shared helper (recommended)
source /home/u5gx/cs2175.u5gx/workspace/share/scripts/activate_env.sh structpo-train

# Option B: Manual
export MAMBA_ROOT_PREFIX=/home/u5gx/cs2175.u5gx/micromamba
eval "$(/home/u5gx/cs2175.u5gx/bin/micromamba shell hook -s bash)"
micromamba activate structpo-train   # for training (LLaMA-Factory + DeepSpeed)
# micromamba activate structpo-eval  # for vLLM inference/evaluation
```

### 3. Full Pipeline

```bash
# --- 4B quick validation (one-click) ---
TRAIN_JOB=$(sbatch --parsable configs/slurm/train_dpo_4b.sh)
bash scripts/run_pipeline_4b.sh $TRAIN_JOB

# --- 8B full pipeline (automated with dependencies) ---
bash scripts/run_pipeline_8b.sh

# --- Or step by step (8B) ---
# Stage 1: SFT on DSE-cleaned data (8B LoRA, ~4 NHR)
sbatch configs/slurm/train_sft_8b.sh

# Generate rollouts from Stage 1 model (vLLM tp=4, ~3 NHR)
sbatch configs/slurm/generate_rollouts_8b.sh

# Annotate traces + build structural preference pairs (CPU, no NHR)
python scripts/annotate_and_build_pairs.py \
    --rollouts data/rollouts/8b_dse_rollouts.json \
    --output data/structural_pairs/8b_structural_dpo_pairs.json

# Stage 2: Structural DPO (8B LoRA, ~2 NHR per β)
sbatch configs/slurm/train_dpo_8b.sh
```

### 4. Quick Smoke Test (no GPU needed)

```bash
python -c "
from structpo.structural_parser.reachability import full_structural_analysis
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

- **Training env** (`structpo-train`): Python 3.11, torch 2.9.1+cu126, LLaMA-Factory 0.9.3, DeepSpeed 0.16.9
- **Eval env** (`structpo-eval`): Python 3.11, vLLM 0.16.0 (separate due to pydantic conflict)
- **CompRL env** (`comprl2`): appworld + pydantic v1 (do NOT cross-install)
- **Structural analysis**: Only needs `pydantic>=2` — runs on CPU, <10ms/trace
- **BriCS**: ARM aarch64, GH200 96GB HBM3, Slurm `workq`, always use `--mem=115000`

### Available Configs

| Config | Model | Method | GPUs | Est. Time |
|:-------|:------|:-------|:----:|----------:|
| `sft/qwen3_4b_dse_sft.yaml` | 4B | Full SFT | 4 | ~1.5h |
| `sft/qwen3_8b_dse_sft_lora.yaml` | 8B | LoRA SFT | 4 | ~4h |
| `sft/qwen3_8b_original_sft_lora.yaml` | 8B | LoRA SFT (control) | 4 | ~4h |
| `dpo/qwen3_4b_structural_dpo.yaml` | 4B | Full DPO | 4 | ~1h |
| `dpo/qwen3_8b_structural_dpo_lora.yaml` | 8B | LoRA DPO | 4 | ~2h |

### Data Included

- `data/limo_cleaned/limo_dse.json` — 817 DSE-cleaned LIMO samples (11MB)
- `data/limo_cleaned/limo_original.json` — 817 original LIMO samples (16MB)
- `data/structural_pairs/structural_dpo_pairs.json` — 2,377 4B structural preference pairs (Type 1-3, 93MB)
- `data/structural_pairs/` — Type 4 contrastive pairs generated after K=8 rollout completion

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
