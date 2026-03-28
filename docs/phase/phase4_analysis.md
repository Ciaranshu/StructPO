# Phase 4: Structural Behavior Analysis + StructParser Distillation

> **Timeline**: Week 2-3 | **GPU-hrs**: ~60 | **Risk**: Low-Medium
> **Updated**: 2026-03-28 — Added StructParser distillation, dead step taxonomy analysis

## Motivation

Phase 4 generates the figures, analysis, and the distilled StructParser model that
make the paper compelling. Two tracks:

1. **Analysis track** (CPU): Deep structural behavior analysis for paper figures
2. **Distillation track** (GPU): Train StructParser-0.5B for online RL compatibility

## 4.1 Analysis Track (CPU only)

### Dead Step Taxonomy (Figure 2)
- Distribution of 7 dead step quality types across 4B and 8B models
- How distribution changes after StructPRM-DPO (before vs after)
- Key message: verification theater and circular revisit are the dominant anti-patterns

### Echo Trap Analysis (Figure 3)
- Outcome reward variance vs StructPRM variance across problem difficulty
- Show that StructPRM variance persists even at L1-L2 (easiest) problems
- Validated finding: 95% of saturated problems retain signal

### Per-Difficulty Breakdown (Figure 4)
- L1-L5 accuracy x DSR scatter plots
- Show exploration necessity increases with difficulty (12% -> 23%)
- Show StructPRM preserves productive exploration on hard problems

### Productive Dead End Analysis (Figure 5)
- Qualitative examples of productive dead ends vs verification theater
- Same problem, two rollouts: one with productive dead ends (quality reward ~ 1.0),
  one with verification theater (quality reward ~ 0.7)
- Key message: raw DSR penalizes both equally, quality-aware reward distinguishes them

### Signal Comparison (Figure 6)
- Radar chart: random / length / correctness / APR / StructPRM-L0 / StructPRM-L1
- Axes: MATH accuracy, GPQA accuracy, DSR reduction, token efficiency

### Scripts
```bash
python scripts/analysis/verify_structural_reward_variance.py  # done
python scripts/analysis/verify_dead_step_quality.py           # done
python scripts/analysis/generate_paper_figures.py             # new
```

## 4.2 StructParser Distillation (Level 3)

### Goal
Train a Qwen-0.5B model to perform step classification + quality annotation,
approximating DecoR's LLM parser (Level 2) at <20ms/trace inference speed.

### Data Preparation
1. Use DecoR's LLM parser to annotate ~10K traces from existing rollouts
   - Input: raw reasoning trace
   - Output: step types + dependency edges + quality labels
2. Convert to supervised training format

### Training
- Base model: Qwen2.5-0.5B
- Task: Given a reasoning trace, output JSON with step types and dependencies
- Training: LoRA, ~2 epochs, ~4 GPU-hrs
- Validation: Compare against DecoR LLM parser on held-out 1K traces

### Evaluation Criteria
- Step type accuracy >= 90% (vs DecoR parser)
- DSR correlation r >= 0.90 (vs DecoR parser)
- Quality classification agreement >= 85%
- Inference speed < 20ms/trace on A100

### GPU cost: ~20 GPU-hrs (annotation + training + eval)

## 4.3 Online RL Pilot (if time permits)

Small-scale experiment to validate anti-collapse:
1. 4B DSE-SFT model
2. GRPO on 100 LIMO problems, 3 epochs
3. Reward A: outcome only
4. Reward B: outcome + StructPRM-L1
5. Track reward variance curve over training steps

If Reward B maintains higher variance -> "StructPRM prevents echo trap" claim validated.

### GPU cost: ~30 GPU-hrs

## Success Criteria

- [ ] All paper figures generated
- [ ] StructParser-0.5B achieves >= 90% step type accuracy
- [ ] StructParser inference < 20ms/trace
- [ ] (Optional) Online RL pilot shows variance preservation
