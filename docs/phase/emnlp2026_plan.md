# StructPRM: EMNLP 2026 Experiment Plan

> **Target**: EMNLP 2026 (deadline ~June 2026, estimated)
> **Framing**: Analysis paper — first graph-structured evaluation of reasoning quality
> **Companion**: DecoR (COLM 2026) — diagnosis of Structural Reward Gap
> **Compute**: ~5,300 GPU-hrs remaining (CSD3 A100-80GB)

## Paper Thesis

> All existing Process Reward Models evaluate reasoning as a linear sequence,
> scoring step correctness. We show that structural quality — whether a step
> contributes to the answer via the reasoning dependency graph — is an
> independent dimension (r ≈ -0.09 with correctness) that current PRMs
> completely miss. We introduce StructPRM, the first graph-structured process
> reward model, and demonstrate it provides meaningful signal in exactly the
> regime where outcome rewards fail (reward saturation), while reducing
> structural waste by 84% without sacrificing accuracy.

## Paper Structure (8 pages)

| Section | Pages | Content |
|:--------|:-----:|:--------|
| 1. Introduction | 1.0 | Linear PRM blind spot → new dimension → StructPRM |
| 2. Background | 0.5 | PRM landscape, DecoR findings, DAG reachability |
| 3. Structural Quality: A New Dimension | 2.0 | DSR definition, ⊥ correctness proof, dead step taxonomy, echo trap |
| 4. StructPRM: Graph-Structured Process Rewards | 1.5 | 3-level reward (L1/L2/L3), quality classification, parser comparison |
| 5. Experiments | 2.0 | Best-of-N, ProcessBench, DPO ablation, scaling |
| 6. Related Work | 0.5 | vs GRP, APR, ThinkPRM, length-based methods |
| 7. Discussion + Conclusion | 0.5 | Limitations, future work (online RL, capability expansion) |

## Core Claims (each needs experimental support)

| # | Claim | Evidence | Status |
|:-:|:------|:---------|:------:|
| 1 | All existing PRMs are linear-thinking | 18+ paper survey | ✅ Done |
| 2 | DSR ⊥ correctness | r ≈ -0.09 on MATH-500 (4000 traces) | ✅ Done |
| 3 | StructPRM breaks echo trap | 95% signal retention on saturated problems | ✅ Done |
| 4 | Dead step taxonomy is meaningful | 7 types, productive dead ends exist (8%) | ✅ Done |
| 5 | StructPRM improves Best-of-N | +0.7pp accuracy, 84% DSR reduction | ✅ Done |
| 6 | Structural signal > length/random in DPO | Signal ablation table | ⏳ TODO |
| 7 | L2 (LLM) parser >> L1 (regex) parser | r(L1,L2) = -0.01, generic_dead 69%→19% | ✅ Done |
| 8 | StructPRM on ProcessBench | Step-level error detection comparison | ⏳ TODO |

## Experiment Plan

### Already Completed (zero additional cost)

| Experiment | Finding | Table/Figure |
|:-----------|:--------|:-------------|
| Echo trap analysis (4B + 8B) | 95% signal retention | Figure 2 |
| Dead step taxonomy (4B) | 7 types, 8% productive | Figure 3 |
| Best-of-N MATH-500 (4B, K=8) | 90.2% acc, 0.030 DSR | Table 1 |
| Best-of-N GPQA (4B, K=8) | StructPRM ⊥ accuracy | Table 1 |
| L1 vs L2 parser comparison | r = -0.01 | Table 3 |
| DSR ⊥ correctness | r ≈ -0.09 | Section 3 |
| Answer normalization analysis | 55→22 true all-wrong | Appendix |
| Hint injection (negative result) | Can't expand boundary | Discussion |

### TODO: Core Experiments

#### E1: Best-of-N at K=64 (MATH-500 + AIME-2024)
**Purpose**: Test-time compute scaling curves — does StructPRM scale with more samples?

| Task | Method | GPU-hrs |
|:-----|:-------|:-------:|
| MATH-500 K=64 rollouts (4B) | 28 shards × 1 GPU × 1hr | 28 |
| AIME-2024 K=64 rollouts (4B) | 2 shards × 1 GPU × 1hr | 2 |
| Analysis + scaling curves | CPU | 0 |

**Deliverable**: Table 1 extended with N=8/16/64 columns, Figure showing scaling curves

#### E2: ProcessBench Evaluation
**Purpose**: Standard PRM benchmark — how does StructPRM perform on step-level error detection?

| Task | Method | GPU-hrs |
|:-----|:-------|:-------:|
| Download ProcessBench (3,400 traces) | CPU | 0 |
| StructPRM-L1 scoring | CPU | 0 |
| StructPRM-L2 scoring | DeepSeek API ~$1 | 0 |
| Comparison vs published ThinkPRM/Math-Shepherd results | Paper numbers | 0 |

**Note**: ProcessBench tests "find the first error step." StructPRM finds "structurally dead steps."
These are related but not identical. Need to design an adapter:
- If a dead step is the CAUSE of subsequent errors → StructPRM detected the error
- If a dead step is just redundant (correct but useless) → different signal

**Deliverable**: Table 2 with ProcessBench results

#### E3: Signal Ablation DPO
**Purpose**: Prove structural signal is better than length/correctness/random for DPO training

| Signal | Pairs | GPU-hrs (1-epoch smoke) | GPU-hrs (3-epoch full) |
|:-------|------:|:----------------------:|:---------------------:|
| Random | ~2,300 | 4 | 12 |
| Length | ~2,300 | 4 | 12 |
| Correctness | ~2,300 | 4 | 12 |
| StructPRM-L1 | ~2,300 | 4 | 12 |

**Strategy**: 1-epoch smoke test first (4 GPU-hrs each), check loss curves, then full 3-epoch.
**Deliverable**: Table 4 with MATH-500 + GPQA + DSR results

#### E4: 8B GPQA + LR Fix
**Purpose**: Complete main results table for 8B

| Task | GPU-hrs |
|:-----|:-------:|
| 8B β=0.20 lr=2e-5 DPO | 12 |
| 8B all β GPQA eval | 6 |

**Deliverable**: Table 5 scaling comparison (4B vs 8B)

#### E5: Majority Voting Baseline
**Purpose**: Standard baseline — how does StructPRM compare to consensus voting?

| Task | GPU-hrs |
|:-----|:-------:|
| Implement majority voting on K=8/64 rollouts | 0 (CPU) |

**Deliverable**: Added to Table 1

#### E6: ThinkPRM Baseline Comparison
**Purpose**: Compare against current SOTA PRM

| Task | GPU-hrs |
|:-----|:-------:|
| Download ThinkPRM checkpoint | 0 |
| Score our K=8/64 rollouts with ThinkPRM | ~4 |
| Best-of-N with ThinkPRM scoring | 0 (CPU) |

**Deliverable**: ThinkPRM row in Table 1

#### E7: StructPRM + ORM Combined Selector
**Purpose**: Show StructPRM is complementary to correctness-based selection

| Task | GPU-hrs |
|:-----|:-------:|
| Implement: majority vote → filter correct → StructPRM select cleanest | 0 (CPU) |

**Deliverable**: "StructPRM + MajVote" row in Table 1 (likely best overall)

### TODO: Analysis + Figures

| Figure | Content | Status |
|:-------|:--------|:------:|
| Fig 1 | Hero: Linear PRM vs StructPRM (same trace, different scoring) | ⏳ Design |
| Fig 2 | Echo trap: outcome vs structural variance by difficulty | ✅ Data ready |
| Fig 3 | Dead step taxonomy distribution | ✅ Data ready |
| Fig 4 | Test-time scaling: accuracy vs N for different selectors | ⏳ Needs K=64 |
| Fig 5 | Per-difficulty (L1-L5) DSR analysis | ✅ Data ready |
| Fig 6 | Qualitative: DAG visualization of good vs bad trace | ⏳ Design |
| Fig 7 | L1 vs L2 parser: DSR scatter + quality type distribution | ✅ Data ready |

### Total GPU Budget

| Phase | GPU-hrs | Timeline |
|:------|:-------:|:---------|
| E1: K=64 rollouts | 30 | Week 1 |
| E2: ProcessBench | 0 | Week 1 |
| E3: DPO ablation (smoke) | 16 | Week 2 |
| E3: DPO ablation (full) | 48 | Week 3 |
| E4: 8B fix | 18 | Week 2 |
| E5: Majority voting | 0 | Week 1 |
| E6: ThinkPRM baseline | 4 | Week 2 |
| E7: Combined selector | 0 | Week 1 |
| **Total** | **~116** | |
| **Reserve** | ~5,180 | |

## Differentiation from GRP (2601.12995)

Must be crystal clear in Related Work:
1. GRP changes output format → StructPRM evaluates existing CoT (format-agnostic)
2. GRP rewards are topology compliance → StructPRM scores structural contribution
3. GRP doesn't analyze DSR ⊥ correctness → we prove this empirically
4. GRP penalizes all dead exploration uniformly → we classify 7 types
5. GRP is a training method (SFT+GRPO) → StructPRM is an evaluation method (PRM)
6. Complementary: StructPRM scoring could improve GRP's rewards

## Key Tables

**Table 1: Best-of-N on MATH-500 (hero table)**
| Selector | N=8 | N=64 | DSR | Tokens |
|:---------|:---:|:----:|:---:|:------:|
| Greedy | 82.8% | — | 0.220 | 3,132 |
| Majority Vote | ? | ? | — | — |
| Random correct | 89.5% | ? | 0.193 | 3,130 |
| Shortest correct | 90.0% | ? | 0.167 | 1,760 |
| ThinkPRM | ? | ? | ? | ? |
| StructPRM-L1 | 90.2% | ? | 0.030 | 2,625 |
| MajVote + StructPRM | ? | ? | ? | ? |

**Table 2: ProcessBench step-level error detection**

**Table 3: Parser level comparison (L1 vs L2)**

**Table 4: DPO signal ablation**

**Table 5: Scaling (4B vs 8B)**
