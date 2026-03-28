# StructPRM: NeurIPS 2026 Experiment Plan

> **Target**: NeurIPS 2026 (deadline ~May 15, 2026)
> **Goal**: Strong Accept — first graph-structured Process Reward Model for reasoning
> **Compute**: ~5,365 GPU-hrs on Cambridge HPC (A100-80GB), SL3 free tier
> **Updated**: 2026-03-28 — Pivoted from StructPO (DPO pairs) to StructPRM (graph-structured PRM)

## Paper Thesis

> All existing Process Reward Models treat reasoning as a linear sequence.
> They evaluate "is this step correct?" but cannot evaluate "does this step
> structurally contribute to the answer?" This blind spot is the root cause
> of verification theater, echo traps, and policy collapse in reasoning RL.
>
> We propose StructPRM: the first graph-structured PRM that uses DAG backward
> reachability to score structural contribution. StructPRM provides reward signal
> orthogonal to both correctness and length, preserving gradient signal in 95%
> of cases where outcome reward is saturated.

## Phase Overview

| Phase | Focus | GPU-hrs | Status |
|:------|:------|--------:|:-------|
| [Phase 1](phase1_foundation.md) | DSE-SFT baselines (4B + 8B) | ~50 | ✅ Done |
| [Phase 2](phase2_structural_dpo.md) | StructPRM as DPO signal | ~100 | ✅ Core done |
| [Phase 3](phase3_ablations.md) | Best-of-N + Signal ablation + Baselines | ~120 | 🔄 In progress |
| [Phase 4](phase4_analysis.md) | Analysis + StructParser distillation | ~60 | ⏳ Pending |
| [Phase 5](phase5_writing.md) | Paper writing + Online RL pilot | ~80 | ⏳ Pending |
| **Total** | | **~410** | |
| **Reserve** | Buffer, re-runs, 14B scaling | **~4,955** | |

## Main Result Tables (Target)

**Table 1: Best-of-N Selection with Different Reward Models**

| Reward Model | Signal Type | MATH-500 | Tokens ↓ | DSR ↓ |
|:-------------|:------------|:--------:|:--------:|:-----:|
| Random | None | baseline | baseline | baseline |
| ORM | Correctness | ? | ? | ? |
| Length | Token count | ? | ? | ? |
| APR-anchor | Post-answer position | ? | ? | ? |
| **StructPRM-L0** | Raw DSR | ? | ? | ? |
| **StructPRM-L1** | Quality-aware structural | **?** | **?** | **?** |
| StructPRM-L1 + ORM | Combined | ? | ? | ? |

**Table 2: Signal Ablation (DPO)**

| Signal | MATH-500 | GPQA | DSR |
|:-------|:--------:|:----:|:---:|
| Random | ? | ? | ? |
| Length | ? | ? | ? |
| Correctness | ? | ? | ? |
| APR-anchor | ? | ? | ? |
| **StructPRM-L0** | ? | ? | ? |
| **StructPRM-L1** | **?** | **?** | **?** |

**Table 3: Pair Type Contribution**

**Table 4: Scaling (4B → 8B → 14B)**

## Key Findings Already Validated

| Finding | Source | Date |
|:--------|:-------|:-----|
| DSR ⊥ correctness (r ≈ -0.21) | K=8 rollout analysis | 2026-03-28 |
| Echo trap breaking (95% signal retention) | verify_structural_reward_variance.py | 2026-03-28 |
| Dead step taxonomy (7 types, 8% productive) | verify_dead_step_quality.py | 2026-03-28 |
| Signal preservation ratio 66-68% | verify_structural_reward_variance.py | 2026-03-28 |
| Type 4 contrastive DPO failure | Exp 05, 07 | 2026-03-05 |
| 4B DPO: MATH 82.2%, GPQA 54.0% | Exp 04 | 2026-03-07 |
| 8B DPO: MATH 80.8% (β=0.20 best) | Exp 06 | 2026-03-08 |
