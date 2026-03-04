# Phase 1: Foundation — 4B Validation + 8B SFT Scaling

> **Timeline**: Day 1-5 | **NHR**: ~40 | **Risk**: Low (4B validated, 8B is scaling)

## Motivation

Stage 1 (DSE-SFT) is the foundation for all subsequent stages. We have already validated
the 4B pipeline (MATH-500: 82.8%, DSR ~22%). Phase 1 scales to 8B and collects all
Stage 1 baselines needed for the paper's main result table.

## Experiments

### 1.1 4B Pipeline Validation (already running)

| Experiment | Config | GPUs | Time | NHR | Status |
|:-----------|:-------|:----:|:----:|----:|:-------|
| 4B DPO training (3 epochs) | `configs/dpo/qwen3_4b_structural_dpo.yaml` | 4 | ~1h | 1 | ✅ Running |
| 4B LoRA merge | `scripts/merge_lora.py` | 1 | ~5min | 0.02 | Chained |
| 4B MATH-500 eval | `scripts/evaluate.py --benchmark math500` | 1 | ~2h | 0.5 | Chained |
| 4B GPQA eval | `scripts/evaluate.py --benchmark gpqa` | 1 | ~2h | 0.5 | Chained |

**Expected results (4B StructPO Stage 2)**:
- MATH-500: ≥ 82.8% (at least match Stage 1)
- GPQA: > 49.0% (recover some cross-domain deliberation lost in Stage 1)
- DSR: < 22% (more structurally efficient)

### 1.2 8B SFT (Stage 1)

| Experiment | Config | GPUs | Time | NHR | Status |
|:-----------|:-------|:----:|:----:|----:|:-------|
| Download Qwen3-8B | `scripts/download_models.sh` | 0 | — | 0 | ✅ Done |
| 8B DSE-SFT (LoRA, 10 epochs) | `configs/sft/qwen3_8b_dse_sft_lora.yaml` | 4 | ~4h | 4 | Pending |
| 8B Original SFT (LoRA, 10 epochs, control) | Need config | 4 | ~4h | 4 | Pending |

**Data**: `data/limo_cleaned/limo_dse.json` (817 samples) and `limo_original.json` (817 samples)

### 1.3 8B Rollout Generation

| Experiment | Config | GPUs | Time | NHR | Status |
|:-----------|:-------|:----:|:----:|----:|:-------|
| 8B DSE-SFT rollouts (MATH train, K=8) | `configs/slurm/generate_rollouts_8b.sh` | 4 | ~4h | 4 | Pending |
| 8B Original-SFT rollouts (control) | Same script, different model | 4 | ~4h | 4 | Pending |

**Output**: ~6,500 traces (817 problems × 8 rollouts)
**Purpose**: Generate rollout pool for structural annotation → DPO pair construction

### 1.4 8B Structural Annotation + Pair Building (CPU, no NHR)

```bash
# On login node (no GPU needed, <10 min)
python scripts/annotate_and_build_pairs.py \
    --rollouts data/rollouts/8b_dse_rollouts.json \
    --output data/structural_pairs/8b_structural_dpo_pairs.json
```

### 1.5 8B Baseline Eval

| Experiment | Config | GPUs | Time | NHR | Status |
|:-----------|:-------|:----:|:----:|----:|:-------|
| 8B DSE-SFT MATH-500 | `scripts/evaluate.py` | 1 | ~3h | 0.75 | After SFT |
| 8B DSE-SFT GPQA | `scripts/evaluate.py` | 1 | ~3h | 0.75 | After SFT |
| 8B Original-SFT MATH-500 | Same | 1 | ~3h | 0.75 | After SFT |
| 8B Original-SFT GPQA | Same | 1 | ~3h | 0.75 | After SFT |

## Deliverables

After Phase 1, we populate these cells in Table 1:

| Model | Method | MATH-500 | GPQA | DSR | Tokens |
|:------|:-------|:--------:|:----:|:---:|:------:|
| Qwen3-4B | Original SFT | 69.6% | 55.6% | ~30% | 18.7k |
| Qwen3-4B | DSE-SFT | 82.8% | 49.0% | ~15% | 15.3k |
| Qwen3-4B | **StructPO** | **Phase 1** | **Phase 1** | **Phase 1** | **Phase 1** |
| Qwen3-8B | Original SFT | **Phase 1** | **Phase 1** | **Phase 1** | **Phase 1** |
| Qwen3-8B | DSE-SFT | **Phase 1** | **Phase 1** | **Phase 1** | **Phase 1** |

## Slurm Commands

```bash
cd /home/u5gx/cs2175.u5gx/workspace/StructPO

# 4B pipeline (already submitted)
TRAIN_JOB=$(sbatch --parsable configs/slurm/train_dpo_4b.sh)
bash scripts/run_pipeline_4b.sh $TRAIN_JOB

# 8B SFT (after 4B validates)
SFT_JOB=$(sbatch --parsable configs/slurm/train_sft_8b.sh)

# 8B rollouts (after SFT)
ROLL_JOB=$(sbatch --parsable --dependency=afterok:$SFT_JOB configs/slurm/generate_rollouts_8b.sh)

# 8B eval (after SFT)
sbatch --dependency=afterok:$SFT_JOB configs/slurm/evaluate.sh --model 8b --benchmark math500
sbatch --dependency=afterok:$SFT_JOB configs/slurm/evaluate.sh --model 8b --benchmark gpqa
```

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|:-----|:----------:|:------:|:-----------|
| 8B SFT doesn't converge | Low | High | 4B already validated; LoRA is stable |
| Rollout generation slow | Low | Medium | vLLM batch inference, 4 GPU |
| GPQA eval noisy | Medium | Low | Use 3 seeds, report mean±std |

## Success Criteria

- [ ] 4B StructPO (Stage 2) MATH-500 ≥ 82.8%
- [ ] 4B StructPO GPQA > 49.0% (ideally > 52%)
- [ ] 8B DSE-SFT MATH-500 > 4B DSE-SFT (82.8%)
- [ ] 8B rollouts generated, structural pairs built
- [ ] All Phase 1 baselines populated in Table 1
