# Phase 4: Structural Behavior Analysis

> **Timeline**: Day 14-17 | **NHR**: ~80 | **Risk**: Low (analysis, not training)

## Motivation

NeurIPS Strong Accept requires not just accuracy numbers, but deep understanding of
**what changed** in model behavior. We need to show that structural preferences
actually change the model's reasoning *structure*, not just its accuracy. This phase
produces the key figures and qualitative analysis that distinguish a strong paper.

## Analysis Experiments

### 4.1 DSR Distribution Analysis (Figure 2)

Generate traces from each model variant on MATH-500, compute DSR for each trace.

| Model | Source | Traces | GPUs | Time | NHR |
|:------|:-------|-------:|:----:|:----:|----:|
| 8B Original SFT | Phase 1 eval logs | 500 | 0 | CPU | 0 |
| 8B DSE-SFT (Stage 1) | Phase 1 eval logs | 500 | 0 | CPU | 0 |
| 8B StructPO (Stage 2, best) | Phase 2 eval logs | 500 | 0 | CPU | 0 |
| 8B Length-DPO (ablation) | Phase 3 eval logs | 500 | 0 | CPU | 0 |

**Figure 2**: Violin/box plot showing DSR distributions across models.
**Expected**: StructPO has the tightest, lowest DSR distribution.

### 4.2 Accuracy vs DSR Scatter (Figure 3)

Per-problem scatter plot: x=DSR, y=correct/incorrect, colored by model.

**Key claim to demonstrate**: Structure ≠ Length. Show examples where:
- Short trace, high DSR (short but wasteful)
- Long trace, low DSR (long but all productive)

### 4.3 Per-Difficulty Breakdown (Figure 4)

| Difficulty | Original SFT | DSE-SFT | StructPO | Δ(StructPO vs DSE) |
|:-----------|:------------:|:-------:|:--------:|:-------------------:|
| MATH L1 | ? | ? | ? | ? |
| MATH L2 | ? | ? | ? | ? |
| MATH L3 | ? | ? | ? | ? |
| MATH L4 | ? | ? | ? | ? |
| MATH L5 | ? | ? | ? | ? |
| GPQA | ? | ? | ? | ? |

**Expected**: StructPO matches DSE-SFT on L1-L3, improves on L4-L5 and GPQA
(because structural preferences teach when exploration is productive).

### 4.4 Verification Behavior Analysis (Figure 6)

For each model, compute:
- **Total verification steps** per trace
- **Live verification rate**: % of verifications that are backward-reachable
- **Verification→Correction rate**: % of verifications that lead to actual corrections

| Metric | Original SFT | DSE-SFT | StructPO |
|:-------|:------------:|:-------:|:--------:|
| Verif steps/trace | ? (high) | ? (low) | ? (medium) |
| Live verif rate | ? | ? | ? (highest) |
| Verif→Correction rate | ? | ? | ? (highest) |

**Key insight**: StructPO doesn't eliminate verification — it makes verification *productive*.

### 4.5 Step Type Distribution Shift

| Step Type | Original SFT | DSE-SFT | StructPO |
|:----------|:------------:|:-------:|:--------:|
| Computation | ? | ? | ? |
| Derivation | ? | ? | ? |
| Verification | ? | ? | ? |
| Exploration | ? | ? | ? |
| Conclusion | ? | ? | ? |

### 4.6 Qualitative Examples (Figure 5)

Hand-pick 3-5 examples showing:

1. **StructPO eliminates dead verification**: Same problem, Original has 3 dead verifications, StructPO has 0
2. **StructPO preserves productive exploration**: Hard problem where StructPO explores (live) but DSE-SFT doesn't
3. **Length ≠ Structure**: Show a short trace with high DSR vs a long trace with low DSR

Format: Side-by-side comparison with color-coded live/dead annotations.

### 4.7 Training Dynamics Analysis

Plot across DPO training steps:
- DPO loss curve
- Reward margin (chosen vs rejected)
- Implicit reward of chosen traces
- Implicit reward of rejected traces

**Purpose**: Show stable training and meaningful preference learning.

## Implementation

### Analysis script: `scripts/analyze_structural_behavior.py`

```python
"""Comprehensive structural behavior analysis.

Inputs: evaluation traces from multiple models
Outputs: 
  - DSR distributions (for Figure 2)
  - Per-difficulty breakdown (for Figure 4)
  - Verification behavior stats (for Figure 6)
  - Step type distributions
  - Qualitative examples
"""
```

### Visualization script: `scripts/plot_results.py`

```python
"""Generate paper-quality figures.

Figures:
  - fig2_dsr_distribution.pdf
  - fig3_accuracy_vs_dsr.pdf
  - fig4_per_difficulty.pdf
  - fig5_qualitative_examples.pdf
  - fig6_verification_behavior.pdf
"""
```

## Compute Requirements

Most analysis runs on CPU using saved evaluation traces. Additional GPU needed only
for generating traces from models not yet evaluated.

| Task | GPUs | Time | NHR |
|:-----|:----:|:----:|----:|
| Generate additional traces (if needed) | 4 | ~4h | 4 |
| Re-run structural analysis on all traces | 0 | CPU | 0 |
| Generate figures | 0 | CPU | 0 |
| Additional AIME 2025/2026 eval (stretch) | 1×4 | ~4h | 4 |

## Success Criteria

- [ ] Figure 2 shows clear DSR distribution shift
- [ ] Figure 4 shows per-difficulty pattern (StructPO helps on hard problems)
- [ ] Verification live rate is highest for StructPO
- [ ] At least 3 compelling qualitative examples
- [ ] All figures are paper-quality (PDF, proper fonts, legends)
