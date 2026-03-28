# 实验 10: StructPRM Analysis + StructParser Distillation (计划中)

> **状态**: ⏳ 计划中
> **阶段**: Phase 4 — Analysis + Distillation
> **更新**: 2026-03-28 — 完全重写，增加 StructParser distillation 和 online RL pilot

---

## 1. 动机

Phase 4 有三个目标：
1. 生成论文需要的所有分析 figures
2. 训练 StructParser-0.5B (Level 3)，验证 online RL 可行性
3. (Optional) 小规模 online RL pilot 验证 anti-collapse

## 2. Analysis Figures (CPU only, 0 GPU-hrs)

### 已完成的分析脚本
- `scripts/analysis/verify_structural_reward_variance.py` — Echo trap 验证 ✅
- `scripts/analysis/verify_dead_step_quality.py` — Dead step taxonomy ✅

### 需要新写的分析
- `scripts/analysis/generate_paper_figures.py` — 生成所有 paper figures
  - Figure 2: Dead step taxonomy 分布 (bar chart)
  - Figure 3: Echo trap — outcome vs structural variance (scatter)
  - Figure 4: Per-difficulty exploration necessity (L1-L5)
  - Figure 5: Qualitative example — DAG 可视化
  - Figure 6: Signal comparison radar chart

### 需要的数据
- K=8 rollouts (已有): `data/rollouts/{4b,8b}_dse_rollouts.json`
- Eval results (已有): `eval_results/*.json`
- Phase 3 ablation results (待 Phase 3 完成)

## 3. StructParser-0.5B Distillation

### Step 1: 用 DecoR LLM parser 标注 gold data
```bash
# 从 DecoR 项目调用 LLM parser
python /home/cs2175/rds/workspace/DecoR/poc/poc1_parse_rir.py \
    --input data/rollouts/4b_dse_rollouts.json \
    --output data/structparser_training/gold_annotations.json \
    --model deepseek-chat \
    --max-traces 10000
```
- 输入: 10K reasoning traces (从 4B + 8B rollouts 采样)
- 输出: JSON with step types + dependency edges
- 预估 API 成本: ~$10-20 (DeepSeek-chat)

### Step 2: 添加 quality labels
```bash
# 用现有 regex classifier 的 quality classification 作为额外标签
python scripts/analysis/add_quality_labels.py \
    --gold data/structparser_training/gold_annotations.json \
    --output data/structparser_training/training_data.json
```

### Step 3: 训练 StructParser-0.5B
- Base: Qwen2.5-0.5B
- Task: trace → JSON (step types + dependencies + quality)
- LoRA rank=32, alpha=64
- 2 epochs, lr=2e-4
- GPU: 1× A100, ~2h

### Step 4: 评估
- vs DecoR LLM parser on 1K held-out traces
- Step type accuracy target: ≥ 90%
- DSR correlation target: r ≥ 0.90
- Inference speed target: < 20ms/trace

### GPU-hrs: ~10 (training) + ~2 (eval) = 12

## 4. Online RL Pilot (Optional, ~30 GPU-hrs)

### 设计
最小化实验验证 "StructPRM reward preserves gradient signal":
- Model: 4B DSE-SFT
- Training: GRPO, 100 LIMO problems, 500 steps
- Condition A: R = 𝟙[correct]
- Condition B: R = 𝟙[correct] + 0.3 × StructPRM-L1

### 核心指标
- Reward variance over training steps (是否下降到 0?)
- Policy entropy over training steps (是否 collapse?)
- Final MATH-500 accuracy (是否 diverge?)

### 预期结果
- Condition A: reward variance → 0 after ~200 steps (echo trap)
- Condition B: reward variance stays > 0 (StructPRM prevents echo trap)

### GPU-hrs
- 2 条件 × 4 GPU × ~4h = 32 GPU-hrs

## 5. 14B Scaling (Optional, ~60 GPU-hrs)

如果时间和预算允许：
- 14B DSE-SFT (LoRA, 需要 checkpoint resume 跨 12h limit)
- 14B K=8 rollouts
- 14B Best-of-N (StructPRM vs others)
- 14B DPO (best config from 8B)

## 6. 总 GPU-hrs

| 任务 | GPU-hrs | 优先级 |
|:-----|--------:|:------:|
| Analysis figures (CPU) | 0 | P0 |
| StructParser data annotation | 0 (API) | P1 |
| StructParser training + eval | 12 | P1 |
| Online RL pilot | 32 | P2 |
| 14B scaling | 60 | P3 |
| **总计 (P0+P1)** | **12** | |
| **总计 (all)** | **104** | |
