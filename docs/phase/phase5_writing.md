# Phase 5: Paper Writing + Final Experiments

> **Timeline**: Day 17+ (post-BriCS) | **NHR**: ~100 (remaining budget) | **Risk**: Low

## Motivation

With all experimental results collected, Phase 5 focuses on writing a compelling
NeurIPS submission and running any gap-filling experiments identified during writing.

## Paper Outline

### Title Options
1. "StructPO: Structural Preference Optimization for Reasoning Quality"
2. "Learning When to Explore: DAG Reachability as Preference Signal for Reasoning"
3. "Beyond Length: Structural Preferences for Efficient Reasoning"

### Section Structure (8 pages + references + appendix)

| Section | Pages | Content |
|:--------|:-----:|:--------|
| 1. Introduction | 1.0 | Structural Reward Gap → StructPO solution |
| 2. Background | 0.5 | DecoR findings, DPO, DAG reachability |
| 3. Method | 2.0 | Three-stage pipeline, structural parser, pair construction |
| 4. Experiments | 2.5 | Tables 1-4, Figures 2-6 |
| 5. Analysis | 1.0 | Structural behavior deep-dive |
| 6. Related Work | 0.5 | Differentiation from BRIDGE, DeCS, Prune-on-Logic |
| 7. Conclusion | 0.5 | Summary + future work (Direction A online RL) |

### Key Claims to Support

1. **Dead steps are correctness-independent** → GRPO's implicit PRM cannot learn to eliminate them (from DecoR)
2. **Structural preferences ≠ length preferences** → DAG reachability captures something length doesn't
3. **StructPO improves efficiency without sacrificing accuracy** → MATH ≥ Stage 1, GPQA recovers
4. **Domain-dependent exploration value** → StructPO preserves GPQA deliberation
5. **Each pair type contributes** → Efficiency + Exploration + Direction are complementary

## Gap-Filling Experiments (if needed)

| Experiment | Trigger | GPUs | NHR |
|:-----------|:--------|:----:|----:|
| Additional β values (0.01, 0.5) | No clear trend in β sweep | 4×2 | 4 |
| More rollouts (K=16) for 8B | Too few structural pairs from K=8 | 4 | 8 |
| AIME 2025 eval | Reviewer expects hard benchmark | 1 | 2 |
| AIME 2026 eval | Reviewer expects latest benchmark | 1 | 2 |
| Qwen3-14B scaling (if time) | Show scaling to larger model | 4 | 20 |
| Stage 3 pilot (difficulty-adaptive) | Stage 2 works well, want full story | 4 | 20 |

## Writing Timeline

| Day | Task |
|:----|:-----|
| 17-18 | Draft introduction + method sections |
| 18-19 | Draft experiment section with all tables/figures |
| 19-20 | Draft analysis + related work |
| 20-21 | Internal review, identify gaps |
| 21-22 | Gap-filling experiments |
| 22-25 | Polish, rewrite, finalize figures |
| 25+ | Submit to NeurIPS (deadline ~May 15) |

## Appendix Content

- Full structural parser algorithm (pseudocode)
- All hyperparameter settings
- Extended per-difficulty results
- Additional qualitative examples (10+)
- Compute cost breakdown
- Structural pair statistics (histograms)
- Training curves for all models

## Reproducibility Checklist

- [ ] All configs committed to repo
- [ ] All evaluation scripts documented
- [ ] Random seeds specified (42, 123, 456)
- [ ] Compute requirements documented
- [ ] Data generation pipeline reproducible
- [ ] Model checkpoints saved (at least best models)

## Success Criteria

- [ ] Complete paper draft (8 pages + appendix)
- [ ] All tables populated with final numbers
- [ ] All figures generated in paper quality
- [ ] Related work thoroughly covers 2025-2026 concurrent work
- [ ] Clear differentiation from BRIDGE, DeCS, Prune-on-Logic
- [ ] Compelling story: efficiency + deliberation preservation
