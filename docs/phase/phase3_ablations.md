# Phase 3: Ablations + Baselines

> **Timeline**: Day 10-14 | **NHR**: ~150 | **Risk**: Low (standard experiments)

## Motivation

Strong Accept requires thorough ablations showing that **structural preferences**
are the key ingredient, not just "any DPO" or "shorter=better". We need to
demonstrate that DAG reachability provides signal that length, correctness, and
random baselines do not.

## Experiments

### 3.1 Preference Signal Ablation (Table 2)

All use the same base model (8B DSE-SFT), same # epochs, same β (best from Phase 2).
Only the preference pairs differ.

| Experiment | Preference Signal | # Pairs | Config | GPUs | Time | NHR |
|:-----------|:-----------------|--------:|:-------|:----:|:----:|----:|
| Control: Random | Random chosen/rejected from correct traces | ~2,300 | custom | 4 | ~2h | 2 |
| Baseline: Length | Shorter correct = preferred | ~2,300 | custom | 4 | ~2h | 2 |
| Baseline: Correctness | Correct > incorrect (standard DPO) | ~2,300 | custom | 4 | ~2h | 2 |
| Baseline: Length + Correct | Correct+short > incorrect+long | ~2,300 | custom | 4 | ~2h | 2 |
| **Ours: Structural** | **DSR-based (low DSR > high DSR)** | **~2,300** | existing | 4 | ~2h | 2 |
| Ours: Structural + Type | DSR + type-weighted scoring | ~2,300 | custom | 4 | ~2h | 2 |

**Pair construction scripts needed**:
```bash
# Generate alternative preference pairs for ablation
python scripts/build_ablation_pairs.py \
    --rollouts data/rollouts/8b_dse_rollouts.json \
    --method random|length|correctness|length_correct|structural_typed \
    --output data/ablation_pairs/
```

Eval: 6 models × 2 benchmarks = 12 eval runs × ~0.75 NHR = ~9 NHR

### 3.2 Pair Type Contribution (Table 4)

Train with subsets of structural pairs to understand which pair type matters most.

| Experiment | Pair Types | # Pairs | GPUs | Time | NHR |
|:-----------|:----------|--------:|:----:|:----:|----:|
| Efficiency only | Type 1 | 1,074 | 4 | ~1.5h | 1.5 |
| Productive Exploration only | Type 2 | 790 | 4 | ~1h | 1 |
| Direction only | Type 3 | 513 | 4 | ~1h | 1 |
| Efficiency + Exploration | Types 1+2 | 1,864 | 4 | ~2h | 2 |
| **All types** | Types 1+2+3 | 2,377 | 4 | ~2h | 2 |

Eval: 5 models × 2 benchmarks = ~8 NHR

### 3.3 External Baselines (if compute allows)

Compare against published efficient reasoning methods on same base model:

| Baseline | Method | Implementation | GPUs | Time | NHR |
|:---------|:-------|:--------------|:----:|:----:|----:|
| Standard SFT (no DSE) | SFT on original LIMO | Existing config | 4 | ~4h | 4 |
| Length-penalized DPO | TokenSqueeze-style | Custom pairs | 4 | ~2h | 2 |
| DSE-SFT + standard DPO | Stage 1 + correct>incorrect DPO | Ablation 3.1 | — | — | — |

### 3.4 Scaling Comparison (4B vs 8B)

Show that structural preferences scale:

| Metric | 4B Improvement | 8B Improvement |
|:-------|:--------------:|:--------------:|
| MATH Δ (Stage 2 vs Stage 1) | ? | ? |
| GPQA Δ | ? | ? |
| DSR Δ | ? | ? |

This comes from existing Phase 1+2 results — no additional compute needed.

## Table Targets

**Table 2: Preference Signal Ablation** (key table for reviewers)

| Preference Signal | MATH-500 | GPQA | DSR ↓ | Interpretation |
|:------------------|:--------:|:----:|:-----:|:---------------|
| No DPO (Stage 1 only) | baseline | baseline | baseline | — |
| Random pairs | ? | ? | ? | DPO format alone doesn't help |
| Length-based | ? | ? | ? | Length ≠ structure |
| Correctness-based | ? | ? | ? | Standard DPO, no structural signal |
| **Structural (ours)** | **best?** | **best?** | **best?** | DAG reachability is the key |
| Structural + Type-weighted | ? | ? | ? | Type-awareness adds value? |

**Table 4: Pair Type Contribution**

| Types Used | MATH-500 | GPQA | DSR | Key Insight |
|:-----------|:--------:|:----:|:---:|:------------|
| Efficiency only | ? | ? | ? | Reduces waste but may hurt deliberation |
| Prod. Exploration only | ? | ? | ? | Teaches productive verification |
| Direction only | ? | ? | ? | Teaches exploration direction |
| **All types** | **?** | **?** | **?** | Complementary contributions |

## Implementation Needed

### New script: `scripts/build_ablation_pairs.py`
```python
"""Build alternative preference pairs for ablation experiments.

Methods:
  random:           Random chosen/rejected from correct traces
  length:           Shorter correct trace preferred
  correctness:      Correct > incorrect (standard DPO)
  length_correct:   Correct+short > incorrect+long
  structural_typed: Structural with type-dependent weighting
"""
```

### New config variants
```yaml
# configs/dpo/ablation_random.yaml   — same as structural but dataset=random_pairs
# configs/dpo/ablation_length.yaml   — same but dataset=length_pairs
# configs/dpo/ablation_correct.yaml  — same but dataset=correctness_pairs
```

## Slurm Commands

```bash
cd /home/u5gx/cs2175.u5gx/workspace/StructPO

# Build ablation pairs (CPU, login node)
for METHOD in random length correctness length_correct structural_typed; do
  python scripts/build_ablation_pairs.py \
    --rollouts data/rollouts/8b_dse_rollouts.json \
    --method $METHOD \
    --output data/ablation_pairs/${METHOD}_pairs.json
done

# Train ablation models (can run multiple concurrently!)
for METHOD in random length correctness length_correct structural_typed; do
  sbatch --parsable --export=ALL,DATASET=$METHOD configs/slurm/train_dpo_8b.sh
done

# Pair type ablations
for TYPES in efficiency exploration direction eff_expl all; do
  sbatch --parsable --export=ALL,PAIR_TYPES=$TYPES configs/slurm/train_dpo_8b.sh
done
```

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|:-----|:----------:|:------:|:-----------|
| Length baseline beats structural | Low | High | Would challenge core thesis; reframe analysis |
| All ablations similar performance | Medium | Medium | Focus on DSR/structural metrics, not just accuracy |
| Not enough compute for all ablations | Low | Medium | Prioritize signal ablation (Table 2) over pair type (Table 4) |

## Success Criteria

- [ ] Structural preferences outperform random, length, and correctness baselines on ≥1 metric
- [ ] Clear separation between structural and non-structural signals
- [ ] Pair type ablation shows complementary contributions
- [ ] Tables 2 and 4 fully populated
