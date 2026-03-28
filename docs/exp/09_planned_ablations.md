# 实验 09: Best-of-N + Signal Ablation (计划中)

> **状态**: 🔄 准备中 (smoke test 阶段)
> **阶段**: Phase 3 — StructPRM Validation
> **更新**: 2026-03-28 — 从 DPO-only ablation 扩展为 Best-of-N + DPO ablation

---

## 1. 动机

StructPRM 的核心 claim 是：结构信号（DAG reachability）比长度信号、正确性信号、和随机信号
更能区分推理质量。需要两组实验来验证：

1. **Best-of-N**: 用不同 reward model 在 K=8 rollouts 中选最佳 → 比较选出的 solution 质量
2. **DPO Ablation**: 用不同信号构建偏好对训练 DPO → 比较训练后模型的性能

## 2. Best-of-N (零训练成本)

### 实验设计

用现有 K=8 rollout 数据 (4B: 6,536 traces, 8B: 6,536 traces)。
对每个问题，用不同 selector 从 rollouts 中选一个 "最佳" solution。

| Selector | 选择规则 |
|:---------|:---------|
| Random | 随机选一个 correct rollout |
| ORM | 选任意 correct rollout (等价于 random among correct) |
| Shortest | 选最短的 correct rollout |
| Longest | 选最长的 correct rollout |
| APR-anchor | 选 post-answer 部分最短的 correct rollout |
| StructPRM-L0 | 选 DSR 最低的 correct rollout |
| StructPRM-L1 | 选 quality-aware reward 最高的 correct rollout |
| StructPRM-L1 + pass@8 | 先 correct filter，再按 L1 排序 |

### 评估指标
- MATH-500 accuracy (只看 selected rollout 的 correctness)
- Average tokens (效率)
- Average DSR (结构质量)
- Average quality reward (L1 指标)

### 代码
```bash
# 零 GPU 成本，CPU only
python scripts/analysis/best_of_n_evaluation.py \
    --rollouts data/rollouts/4b_dse_rollouts.json \
    --rollouts-8b data/rollouts/8b_dse_rollouts.json
```

## 3. DPO Signal Ablation

### Pair 构建

所有 ablation 使用相同的 8B rollout 数据，构建 ~2,300 对偏好对。
差异仅在于选择 chosen/rejected 的规则。

| Pair Type | Chosen | Rejected | 预期 # pairs |
|:----------|:-------|:---------|:------------:|
| Random | 随机分配 | 随机分配 | ~2,300 |
| Length | 更短的 correct | 更长的 correct | ~2,300 |
| Correctness | correct | incorrect | ~2,300 |
| APR-anchor | post-anchor 更短 | post-anchor 更长 | ~2,300 |
| StructPRM-L0 | 低 DSR correct | 高 DSR correct | ~2,300 (existing) |
| StructPRM-L1 | 高 quality reward | 低 quality reward | ~2,300 |

### 训练配置
- Base: 8B DSE-SFT (Stage 1 merged)
- LoRA rank=64, alpha=128
- β=0.20 (8B best from Phase 2)
- lr=2e-5 (修复后的 lr，待 Phase 3.4 确认)
- Epochs: 3

### 代码
```bash
# Step 1: 构建 ablation pairs (CPU)
python scripts/build_ablation_pairs.py \
    --rollouts data/rollouts/8b_dse_rollouts.json \
    --method random|length|correctness|apr_anchor|structural_l1 \
    --output data/ablation_pairs/

# Step 2: DPO 训练 (GPU)
sbatch configs/slurm/train_dpo_8b.sh configs/dpo/ablation_${METHOD}.yaml

# Step 3: Merge + Eval
sbatch configs/slurm/merge_eval_8b_generic.sh ...
```

## 4. Smoke Test 清单

- [ ] 构建所有 ablation pairs → 验证数量和长度分布
- [ ] Best-of-N script 在 10 个 problem 上 dry run
- [ ] 每种 DPO config 跑 1 step → 验证无报错
- [ ] 检查 pair 的 chosen/rejected 长度比 → 确认 length pair 和 structural pair 的差异

## 5. 预期结果

| Selector/Signal | MATH (预期) | DSR (预期) | 推理 |
|:----------------|:----------:|:----------:|:-----|
| Random | baseline | baseline | 无信号 |
| Length | ≈ baseline | 略低 | 长度不代表质量 |
| Correctness | > baseline | 不确定 | Standard DPO |
| **StructPRM-L1** | **≥ baseline** | **最低** | 结构信号区分质量 |

如果 StructPRM-L1 在 Best-of-N 中 accuracy 最高且 DSR 最低，核心 claim 成立。

## 6. GPU-hrs 预算

| 任务 | GPU-hrs |
|:-----|--------:|
| Best-of-N (CPU) | 0 |
| Pair building (CPU) | 0 |
| DPO training × 5 | 60 |
| Merge + eval × 5 | 15 |
| GPQA eval (best 2) | 4 |
| **总计** | **~79** |
