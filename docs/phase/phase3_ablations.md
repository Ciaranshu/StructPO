# Phase 3: StructPRM Validation — Best-of-N + Signal Ablation

> **Timeline**: Week 1-2 | **GPU-hrs**: ~120 | **Risk**: Medium
> **Updated**: 2026-03-28 — Restructured around StructPRM (was: DPO ablations)

## Motivation

The core claim is: **StructPRM provides a reward signal orthogonal to correctness
and length, and this signal is useful for selecting and training better reasoners.**

Phase 3 validates this through two key experiments:

1. **Best-of-N selection** (zero training cost) — does StructPRM select better solutions?
2. **Signal ablation** (DPO training) — is structural signal better than length/correctness/random?

## 3.1 Best-of-N Selection (Table 1 — Hero Table)

Use existing K=8 rollouts. For each problem, select the "best" rollout according to
different reward models. Compare accuracy and efficiency.

| Selector | Signal | Training needed? |
|:---------|:-------|:----------------:|
| Random (among correct) | None | No |
| ORM (outcome) | Correctness only | No |
| Shortest correct | Length | No |
| Longest correct | Length (inverse) | No |
| APR-style (least post-anchor) | Answer anchor position | No |
| StructPRM-L0 (lowest DSR) | Raw DSR | No |
| **StructPRM-L1 (quality-aware)** | Quality-classified structural reward | No |
| StructPRM-L1 + ORM (combined) | Structural × Correctness | No |

**Implementation**: `scripts/analysis/best_of_n_evaluation.py`

**Eval metrics per selector**:
- MATH-500 accuracy (from the selected rollout)
- Average tokens (efficiency)
- Average DSR (structural quality)
- Average steps

**Why this is the hero table**: Best-of-N is a **direct test of reward model quality**.
If StructPRM selects solutions that are both more accurate AND more efficient than
ORM/Length selection, it proves the signal has practical value.

**GPU cost**: 0 (uses existing rollout data + eval results)

## 3.2 Signal Ablation — DPO (Table 2)

All use 8B DSE-SFT as base, same # pairs (~2,300), same β (best from Phase 2).
Only the preference signal differs.

| Experiment | Signal | Pairs |
|:-----------|:-------|------:|
| Random | Random chosen/rejected from correct traces | ~2,300 |
| Length | Shorter correct preferred | ~2,300 |
| Correctness | Correct > incorrect (standard DPO) | ~2,300 |
| APR-anchor | Shorter post-anchor tail preferred | ~2,300 |
| **StructPRM-L0** | Low DSR > high DSR (current method) | ~2,300 |
| **StructPRM-L1** | Quality-aware structural reward | ~2,300 |

**Implementation needed**:
- `scripts/build_ablation_pairs.py` — generate alternative pair types
- New DPO configs for each ablation

**GPU cost**: 6 × (12 train + 4 eval) = ~96 GPU-hrs

## 3.3 Pair Type Contribution (Table 3)

Train with subsets of structural pairs:

| Types | Pairs | GPU-hrs |
|:------|------:|--------:|
| Type 1 (Efficiency) only | ~1,074 | 16 |
| Type 2 (Productive Exploration) only | ~790 | 16 |
| Type 3 (Direction) only | ~513 | 16 |
| Type 1+2 | ~1,864 | 16 |
| **Type 1+2+3 (all)** | ~2,377 | 16 |

**GPU cost**: ~80 GPU-hrs

## 3.4 8B Fix — Learning Rate Tuning

| lr | β | GPU-hrs |
|:---|:--|--------:|
| 2e-5 | 0.20 | 16 |
| 1e-5 | 0.20 | 16 |

Plus GPQA evaluation for all 8B models (3 existing β + 2 new lr): ~10 GPU-hrs

## Execution Plan

### Smoke Tests (Interactive Node, 1hr)
1. Build all ablation pairs → verify counts and distributions
2. 1-step DPO training with each config → verify no errors
3. Best-of-N script dry run on 10 problems

### Batch Jobs
**Wave 1** (parallel, Day 1):
- Jobs 1-2: 8B lr tuning (COLLIER account)
- Jobs 3-6: Signal ablation DPO × 4 (SHAREGHI account)

**Wave 2** (after Wave 1 results, Day 3):
- Jobs 7-11: Pair type ablation × 5
- Jobs 12+: Merge + eval for all models

## Success Criteria

- [ ] StructPRM Best-of-N selects solutions with higher accuracy than random/length/ORM
- [ ] StructPRM DPO signal outperforms random/length/correctness on ≥1 metric
- [ ] Pair type ablation shows complementary contributions
- [ ] 8B accuracy drop resolved to ≤ 1pp
