# Phase 2: Core — Structural DPO (Stage 2)

> **Timeline**: Day 5-10 | **NHR**: ~120 | **Risk**: Medium (core novelty)
> **Updated**: 2026-03-04 — Added Structural Contrastive DPO (Type 4), coverage gap analysis

## Motivation

This is the **central contribution** of the paper. Stage 2 uses structural preference
pairs (built from DAG reachability analysis) to teach the model a complete **exploration
policy** via DPO. Unlike length-based or correctness-based preferences, structural
preferences distinguish *productive* exploration from *wasteful* exploration.

**Core thesis (updated 2026-03-04)**: Not all dead-end exploration is bad. For hard
problems (MATH Level 4-5), 23-30% of correct solutions involve substantial exploration
(DSR ≥ 0.3). The question is not "how to eliminate exploration" but "how to make
exploration productive." The four pair types jointly teach:
- **When NOT to explore** (Type 1: Efficiency) — on clear problems, derive directly
- **How to verify productively** (Type 2: Productive Exploration) — discover, don't confirm
- **When to abandon dead ends** (Type 3: Direction) — explore with direction, cut losses
- **Which patterns are toxic** (Type 4: Contrastive) — surgically target structural motifs

This is orthogonal to length — 27% of chosen solutions are LONGER than rejected.
**Structure ≠ Length.**

### Coverage Gap Problem (discovered 2026-03-04)

Type 1-3 trace-level pairs cover only 534/817 (65%) of LIMO problems. 283 problems
have zero training signal because:
- 129 easy problems: 8/8 correct with uniformly low DSR (< 0.15)
- 109 moderate: insufficient DSR variance between rollouts
- 34 impossible: 0/8 correct
- 11 marginal: only 1/8 correct

**Type 4 contrastive pairs** recover 108 of these through motif excision (DSR-threshold-
independent). Coverage: 65% → 79%.

## Experiments

### 2.1 DPO Training — β Sweep (4B)

| Experiment | β_DPO | Config | GPUs | Time | NHR |
|:-----------|:-----:|:-------|:----:|:----:|----:|
| 4B StructPO β=0.05 | 0.05 | `qwen3_4b_structural_dpo.yaml` | 4 | ~1h | 1 |
| 4B StructPO β=0.10 | 0.10 | Same, modified β | 4 | ~1h | 1 |
| 4B StructPO β=0.20 | 0.20 | Same, modified β | 4 | ~1h | 1 |

**Training data (Type 1-3 only)**: 2,377 structural preference pairs
- 1,074 Structural Efficiency pairs (45%)
- 790 Productive Exploration pairs (33%)
- 513 Exploration Direction pairs (22%)

### 2.1b Contrastive DPO Training (4B) — NEW

| Experiment | Pair Types | Config | GPUs | Time | NHR |
|:-----------|:----------:|:-------|:----:|:----:|----:|
| 4B Contrastive DPO (best β) | Type 1-4 | Same, expanded data | 4 | ~1h | 1 |

**Training data (Type 1-4)**: ~2,834+ pairs (after K=8 rollouts)
- Type 1-3: ~2,377 trace-level pairs
- Type 4A: Excision pairs (~200-400, from motif removal)
- Type 4B: Replacement pairs (~100-300, from clean/dirty rollout swap)
- Type 4C: Contrast pairs (~100-300, from motif-specific cross-trace pairing)

**Key ablation**: Type 1-3 only vs Type 1-4 (contrastive). Expected to show:
- Coverage improvement: 65% → 79% of problems have training signal
- Motif-specific DSR reduction (especially verification theater)

### 2.2 DPO Training — β Sweep (8B)

| Experiment | β_DPO | Config | GPUs | Time | NHR |
|:-----------|:-----:|:-------|:----:|:----:|----:|
| 8B StructPO β=0.05 | 0.05 | `qwen3_8b_structural_dpo_lora.yaml` | 4 | ~2h | 2 |
| 8B StructPO β=0.10 | 0.10 | Same, modified β | 4 | ~2h | 2 |
| 8B StructPO β=0.20 | 0.20 | Same, modified β | 4 | ~2h | 2 |

**Training data**: 8B structural pairs (built in Phase 1.4, expected ~2,000-3,000 Type 1-3 pairs + Type 4 contrastive)

### 2.2b Contrastive DPO Training (8B) — NEW

| Experiment | Pair Types | Config | GPUs | Time | NHR |
|:-----------|:----------:|:-------|:----:|:----:|----:|
| 8B Contrastive DPO (best β) | Type 1-4 | Same, expanded data | 4 | ~2h | 2 |

**8B has more circular revisit motifs** (259 vs 59 in 4B), so Type 4 may be even more impactful.

### 2.3 Evaluation — All Stage 2 Models

For each of the 6 StructPO models (3 β × 2 sizes):

| Benchmark | GPUs | Time/model | NHR/model |
|:----------|:----:|:----------:|----------:|
| MATH-500 | 1 | ~2h (4B) / ~3h (8B) | 0.5 / 0.75 |
| GPQA Diamond | 1 | ~2h / ~3h | 0.5 / 0.75 |
| Structural analysis (DSR) | 0 | CPU | 0 |

Total eval: 6 models × 2 benchmarks × ~0.6 NHR = ~7 NHR

### 2.4 DPO Training — Epoch Sweep (best β)

After identifying best β from 2.1/2.2:

| Experiment | Epochs | GPUs | Time | NHR |
|:-----------|:------:|:----:|:----:|----:|
| 4B StructPO 1 epoch | 1 | 4 | ~20min | 0.3 |
| 4B StructPO 3 epochs | 3 | 4 | ~1h | 1 |
| 4B StructPO 5 epochs | 5 | 4 | ~1.5h | 1.5 |
| 8B StructPO 1 epoch | 1 | 4 | ~40min | 0.7 |
| 8B StructPO 3 epochs | 3 | 4 | ~2h | 2 |
| 8B StructPO 5 epochs | 5 | 4 | ~3h | 3 |

## Main Result Table (populated after Phase 2)

**Table 1: Main Results** — target population:

| Model | Method | MATH-500 | GPQA | DSR ↓ | Tokens ↓ |
|:------|:-------|:--------:|:----:|:-----:|:--------:|
| Qwen3-4B | Original SFT | 69.6% | 55.6% | ~30% | 18.7k |
| Qwen3-4B | DSE-SFT (Stage 1) | 82.8% | 49.0% | ~22% | ~15k |
| Qwen3-4B | **StructPO (best)** | **?** | **?** | **?** | **?** |
| Qwen3-8B | Original SFT | from P1 | from P1 | from P1 | from P1 |
| Qwen3-8B | DSE-SFT (Stage 1) | from P1 | from P1 | from P1 | from P1 |
| Qwen3-8B | **StructPO (best)** | **?** | **?** | **?** | **?** |

**Table 3: β Sensitivity** — fully populated:

| β | 4B MATH | 4B GPQA | 8B MATH | 8B GPQA |
|:--|:-------:|:-------:|:-------:|:-------:|
| 0.05 | ? | ? | ? | ? |
| 0.10 | ? | ? | ? | ? |
| 0.20 | ? | ? | ? | ? |

## Slurm Commands

```bash
cd /home/u5gx/cs2175.u5gx/workspace/StructPO

# 4B DPO β sweep (can run 3 concurrently with --mem=115000!)
for BETA in 0.05 0.10 0.20; do
  sbatch --parsable --export=ALL,PREF_BETA=$BETA configs/slurm/train_dpo_4b.sh
done

# 8B DPO β sweep (after Phase 1 8B SFT + pairs are ready)
for BETA in 0.05 0.10 0.20; do
  sbatch --parsable --export=ALL,PREF_BETA=$BETA configs/slurm/train_dpo_8b.sh
done

# Eval all models
for MODEL in 4b_beta005 4b_beta010 4b_beta020 8b_beta005 8b_beta010 8b_beta020; do
  sbatch --time=4:00:00 configs/slurm/evaluate.sh --model $MODEL --benchmark math500
  sbatch --time=4:00:00 configs/slurm/evaluate.sh --model $MODEL --benchmark gpqa
done
```

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|:-----|:----------:|:------:|:-----------|
| DPO doesn't improve over Stage 1 | Medium | Critical | Multiple β values; pair type ablation in Phase 3 |
| GPQA doesn't recover | Medium | Medium | Still informative result; reframe as analysis |
| 8B pairs too few/noisy | Low | Medium | Fall back to 4B pairs transferred to 8B |
| Training diverges at low β | Low | Medium | Monitor loss; increase β |

## Decision Points

- **After 4B β sweep**: If no β improves MATH or GPQA → investigate pair quality, possibly regenerate pairs with different thresholds
- **After 8B β sweep**: Pick best β → proceed to Phase 3 ablations
- **Type 1-3 vs Type 1-4**: If contrastive DPO shows improvement → use Type 1-4 as main result
- **If coverage still insufficient after Type 4**: Supplement DPO prompt pool with Big-Math ~2-3K medium-difficulty problems
- **If GPQA doesn't recover**: Reframe contribution as "structural preferences improve efficiency without accuracy loss" (still publishable)

## Success Criteria

- [x] At least one β gives MATH-500 ≥ Stage 1 baseline (β=0.10: 82.4% ≈ 82.8%)
- [ ] At least one β recovers some GPQA loss (> 49.0% for 4B)
- [x] DSR decreases relative to Stage 1 for best model (β=0.10: 18.9% < 22.0%)
- [ ] Clear trend in β sensitivity (not random noise) — awaiting β=0.05, 0.20
- [ ] Contrastive DPO (Type 1-4) improves over Type 1-3 only (coverage or accuracy)
- [ ] 8B StructPO matches or exceeds 4B results
