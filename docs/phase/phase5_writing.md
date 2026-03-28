# Phase 5: Paper Writing + Final Experiments

> **Timeline**: Week 3+ | **GPU-hrs**: ~80 (remaining budget) | **Risk**: Low
> **Updated**: 2026-03-28 — Restructured for StructPRM paper

## Paper Outline

### Title Options (ranked)
1. "Not All Correct Steps Are Useful: Graph-Structured Process Rewards for Reasoning"
2. "Beyond Linear Verification: Structural Process Reward Models for Reasoning Quality"
3. "StructPRM: Teaching Reasoning Models What Productive Exploration Looks Like"

### Section Structure (8 pages + references + appendix)

| Section | Pages | Content |
|:--------|:-----:|:--------|
| 1. Introduction | 1.0 | Linear PRM blind spot → Structural Reward Gap → StructPRM |
| 2. Background | 0.5 | PRM landscape, DAG reachability (from DecoR), DPO |
| 3. The Linear PRM Blind Spot | 1.0 | Empirical evidence: DSR ⊥ correctness, echo trap, dead step taxonomy |
| 4. StructPRM | 2.0 | DAG construction, backward reachability, 3-level reward, quality classification |
| 5. Experiments | 2.0 | Best-of-N (Table 1), DPO signal ablation (Table 2), pair type (Table 3), scaling (Table 4) |
| 6. Related Work | 0.5 | PRM taxonomy + positioning vs APR/Draft-Thinking/MAP |
| 7. Discussion + Conclusion | 1.0 | Collapse unification, limitations, future work |

### Core Claims

1. **All existing PRMs are linear-thinking** — they cannot detect correct-but-dead steps
2. **StructPRM provides orthogonal signal** — DSR ⊥ correctness (r ≈ -0.21)
3. **StructPRM breaks the echo trap** — 95% of saturated problems retain signal
4. **Quality-aware reward protects productive exploration** — 8% of dead steps are productive
5. **StructPRM improves Best-of-N selection** — higher accuracy AND efficiency
6. **Structural signal > length/correctness signal** — ablation evidence

### Key Tables

**Table 1: Best-of-N Selection with Different Reward Models**
| Reward Model | MATH-500 | Tokens ↓ | DSR ↓ |
|:-------------|:--------:|:--------:|:-----:|
| Random (among correct) | ? | ? | ? |
| ORM (outcome) | ? | ? | ? |
| Length (shortest) | ? | ? | ? |
| APR-anchor | ? | ? | ? |
| **StructPRM-L1** | **?** | **?** | **?** |
| StructPRM-L1 + ORM | ? | ? | ? |

**Table 2: Signal Ablation (DPO)**
| Signal | MATH | GPQA | DSR |
|:-------|:----:|:----:|:---:|
| Random | ? | ? | ? |
| Length | ? | ? | ? |
| Correctness | ? | ? | ? |
| **Structural (ours)** | **?** | **?** | **?** |

**Table 3: Pair Type Contribution**

**Table 4: Scaling (4B → 8B → 14B)**

### Key Figures

1. **Figure 1 (Hero)**: Linear PRM vs StructPRM — same trace, different scoring
2. **Figure 2**: Dead step taxonomy distribution
3. **Figure 3**: Echo trap analysis — outcome vs structural variance
4. **Figure 4**: Per-difficulty exploration necessity
5. **Figure 5**: Qualitative example — productive dead end vs verification theater
6. **Figure 6**: Signal comparison radar chart

## Gap-Filling Experiments (if needed)

| Experiment | Trigger | GPU-hrs |
|:-----------|:--------|--------:|
| Additional β values | No clear trend | 8 |
| 14B DSE-SFT + StructPRM | Scale evidence needed | 60 |
| AIME 2024/2025 evaluation | Reviewer expects hard benchmark | 8 |
| StructParser-0.5B distillation | Level 3 claim needs validation | 20 |
| Online RL pilot | Anti-collapse claim needs evidence | 30 |

## Writing Timeline

| Week | Task |
|:-----|:-----|
| 3 | Draft introduction + method (§1-4) |
| 3 | Draft experiments (§5) with all tables |
| 4 | Draft analysis + related work (§3, 6, 7) |
| 4 | Internal review, identify gaps |
| 5 | Gap-filling experiments |
| 5-6 | Polish, rewrite, finalize figures |
| 6+ | Submit to NeurIPS (deadline ~May 15) |

## Related Work Taxonomy

### Process Reward Models (Linear-Thinking)
PRM800K, Math-Shepherd, ThinkPRM, GenPRM, CRM, PRIME, GRPO-as-PRM, R-PRM, PURE

### Efficient Reasoning (Length-Based)
Draft-Thinking, O1-Pruner, DAST, CoT-Valve, L1, BRIDGE, ThinkPrune, Leash

### Structural Analysis (No Training Signal)
MAP (EMNLP 2025), Topology of Reasoning, From Chains to DAGs, DAG-Math

### Structure-Aware Methods (Partial Overlap)
APR (anchor-based, post-answer only), TACReward (process mining, needs teacher)

**StructPRM positioning**: First PRM that (1) models reasoning as DAG, (2) uses backward
reachability, (3) scores structural contribution not correctness, (4) works online in RL.

## Appendix Content

- Full structural parser algorithm (pseudocode)
- All hyperparameter settings
- Extended per-difficulty results
- Dead step taxonomy examples (10+)
- StructParser-0.5B training details
- Compute cost breakdown
