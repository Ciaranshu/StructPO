# StructPO: NeurIPS 2026 Experiment Plan

> **Target**: NeurIPS 2026 (deadline ~May 15, 2026)
> **Goal**: Strong Accept — novel structural preference optimization for reasoning quality
> **Compute**: ~1,000 NHR on Isambard-AI (4× GH200 per node), expires Mar 22

## Paper Thesis

> Standard outcome-based RL is blind to structural waste because dead steps are
> correctness-independent. We propose StructPO: a three-stage curriculum that uses
> DAG reachability to teach models (1) efficient reasoning, (2) when exploration
> is productive, and (3) how to adapt exploration to problem difficulty.

## Phase Overview

| Phase | Focus | Timeline | NHR | Key Deliverable |
|:------|:------|:---------|----:|:----------------|
| [Phase 1](phase1_foundation.md) | Foundation: 4B Validation + 8B SFT | Day 1-5 | 40 | Stage 1 baselines across 4B/8B |
| [Phase 2](phase2_structural_dpo.md) | Core: Structural DPO (Stage 2) | Day 5-10 | 120 | **Main result tables** |
| [Phase 3](phase3_ablations.md) | Ablations + Baselines | Day 10-14 | 150 | Ablation tables, baseline comparisons |
| [Phase 4](phase4_analysis.md) | Structural Behavior Analysis | Day 14-17 | 80 | Analysis figures, qualitative examples |
| [Phase 5](phase5_writing.md) | Paper Writing + Final Experiments | Day 17+ | 100 | Camera-ready experiments |
| **Total** | | **17 days** | **~490** | |
| **Reserve** | Buffer, re-runs, CompRL | | **~510** | |

## Main Result Table (Target)

**Table 1: Main Results on Mathematical Reasoning**

| Model | Method | MATH-500 | GPQA Diamond | Avg DSR ↓ | Avg Tokens ↓ |
|:------|:-------|:--------:|:------------:|:---------:|:------------:|
| Qwen3-4B | Base (Original SFT) | 69.6% | 55.6% | ~30% | 18.7k |
| Qwen3-4B | DSE-SFT (Stage 1) | 72.8% | 49.0% | ~15% | 15.3k |
| Qwen3-4B | **StructPO (Stage 1+2)** | **?** | **?** | **?** | **?** |
| Qwen3-8B | Base (Original SFT) | ? | ? | ? | ? |
| Qwen3-8B | DSE-SFT (Stage 1) | ? | ? | ? | ? |
| Qwen3-8B | **StructPO (Stage 1+2)** | **?** | **?** | **?** | **?** |

**Expected**: StructPO ≥ DSE-SFT on MATH, significantly > DSE-SFT on GPQA, lowest DSR.

**Table 2: Ablation — What Makes Structural Preferences Work?**

| Preference Signal | MATH-500 | GPQA | DSR |
|:------------------|:--------:|:----:|:---:|
| Random pairs (control) | ? | ? | ? |
| Length-based (shorter=better) | ? | ? | ? |
| Correctness-based (correct>incorrect) | ? | ? | ? |
| **Structural (DSR-based, ours)** | **?** | **?** | **?** |
| Structural + Type-weighted | ? | ? | ? |

**Table 3: DPO Hyperparameter Sensitivity**

| β_DPO | MATH-500 | GPQA | DSR | Notes |
|:------|:--------:|:----:|:---:|:------|
| 0.05 | ? | ? | ? | More deviation from ref |
| 0.1 | ? | ? | ? | Default |
| 0.2 | ? | ? | ? | Less deviation |

**Table 4: Pair Type Contribution**

| Pair Types Used | # Pairs | MATH-500 | GPQA | DSR |
|:----------------|--------:|:--------:|:----:|:---:|
| Efficiency only | 1,074 | ? | ? | ? |
| Productive Exploration only | 790 | ? | ? | ? |
| Direction only | 513 | ? | ? | ? |
| **All types (full)** | **2,377** | **?** | **?** | **?** |

## Key Figures (Planned)

1. **Figure 1**: Paper overview diagram (3-stage pipeline)
2. **Figure 2**: DSR distribution: Base vs Stage 1 vs Stage 2 (violin plot)
3. **Figure 3**: Accuracy vs DSR scatter (shows structural efficiency ≠ length)
4. **Figure 4**: Per-difficulty breakdown (MATH L1-L5) — Stage 2 preserves hard-problem exploration
5. **Figure 5**: Qualitative examples: preferred vs rejected reasoning traces
6. **Figure 6**: Verification behavior analysis (live vs dead verification rates)

## Compute Budget Allocation

```
StructPO (main paper):
  Phase 1: 4B pipeline (validated)     ~10 NHR
  Phase 1: 8B SFT + rollouts           ~30 NHR
  Phase 2: 8B DPO × 3 β values         ~20 NHR
  Phase 2: 4B DPO × 3 β values         ~10 NHR
  Phase 2: Eval all models              ~30 NHR
  Phase 3: Ablation baselines           ~80 NHR
  Phase 3: Pair type ablations          ~40 NHR
  Phase 4: Structural analysis runs     ~30 NHR
  Phase 5: Final experiments            ~50 NHR
  Buffer                                ~190 NHR
  ────────────────────────────────
  Subtotal:                             ~490 NHR

CompRL (concurrent project):
  14B GRPO training (resumed)           ~30 NHR
  14B pass@k eval (base/sft/rl)         ~50 NHR
  8B GRPO re-run + eval                 ~40 NHR
  Compositional eval                    ~30 NHR
  Buffer                                ~100 NHR
  ────────────────────────────────
  Subtotal:                             ~250 NHR

Reserve:                                ~260 NHR
  ════════════════════════════════
  TOTAL:                                ~1,000 NHR
```
