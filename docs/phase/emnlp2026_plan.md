# StructPRM: EMNLP 2026 Complete Story + Experiment Plan

> **Target**: EMNLP 2026 via ARR (deadline: **May 25, 2026**)
> **Updated**: 2026-04-03
> **Companion**: DecoR (COLM 2026) — diagnosis of Structural Reward Gap

---

## Part 1: The Story (from first principles)

### Chapter 1: The Problem — What's wrong with current reasoning models?

2024-2025 年，thinking models（DeepSeek-R1, QwQ, o1）学会了长链推理。但 DecoR
项目分析了 11 个模型、13,832 条推理 trace 后发现：

**推理过程中大量步骤是"正确但无用的"。**

- 82-96% 的 verification 步骤是 structurally dead（结果从未被后续引用）
- RL 模型比 SFT 模型多 5.6× dead verification
- Dead Step Ratio (DSR) 与 correctness 几乎无关 (r ≈ -0.09)

**根因**：Outcome-based RL (GRPO) 的 reward 只看最终答案对不对，完全看不到推理
过程的结构质量。Dead steps 既不被奖励也不被惩罚 — 它们在 gradient 的盲区里。

### Chapter 2: The Insight — Linear-Thinking vs Struct-Thinking

我们发现了一个更深层的问题：

**所有现有的 Process Reward Models (PRM) 都是 "Linear-Thinking" 的。**

PRM800K, Math-Shepherd, ThinkPRM, GenPRM, CRM, PRIME... 18+ 篇论文，全部把推理
当成线性序列 s₁ → s₂ → ... → sₜ 来评估。它们回答 "这步对不对？" 但无法回答
**"这步有没有用？"**

一个步骤可以完全正确但完全无用（dead step）。Linear PRM 对此完全盲目。

PRM Survey (Oct 2025) 明确指出 "non-sequential credit paths" 和 "graph models for
joint state-action-outcome reasoning" 是 open problems。

### Chapter 3: Our Solution — StructPRM

StructPRM 是第一个 **Graph-Structured Process Reward Model**：

```
Linear PRM:  步骤 → "这步对不对？" → score (correctness)
StructPRM:   步骤 → "这步在 DAG 中能到达答案吗？" → score (structural contribution)
```

技术方案：
1. 从 natural CoT 解析出 dependency DAG（不改变模型输出格式）
2. Backward reachability 标记每步为 live/dead
3. 三级 reward：
   - L1: Regex + graph rules（<10ms，RL 可用）
   - L2: LLM parser（DeepSeek-chat，精度最高）
   - L3: Distilled small model（待训练）

### Chapter 4: Why This Matters — Three Concrete Benefits

**Benefit 1: 新的评价维度**
DSR 与 correctness 正交 (r ≈ -0.09)。这意味着 structural quality 是一个独立的维度，
现有 PRM 完全看不到。

**Benefit 2: Echo Trap Breaking**
当所有 K=8 rollouts 都 correct 时（easy problems），outcome reward variance = 0，
RL 停止学习。StructPRM 在这些 saturated problems 的 95% 上仍有非零 variance。

**Benefit 3: 实际效果**
Best-of-N 实验：MajVote + StructPRM 达到与 MajVote 相同的 accuracy (93.6%)，
同时 DSR 从 0.199 降到 0.035（82% reduction），tokens 减少 12%。

### Chapter 5: What StructPRM is NOT

诚实的 scope：
- ❌ 不提升 accuracy（structural quality ≠ correctness）
- ❌ 不扩展能力边界（hints 实验验证了 4B 的硬限制）
- ❌ 不是 length-based（27% chosen traces 更长）
- ✅ 在 **不损失 accuracy 的前提下**，大幅提升推理的结构质量

### Chapter 6: 与 GRP 的区别

GRP (2601.12995) 是最接近的工作，也使用了 graph + backward reachability。

关键区别：
- GRP **改变模型输出格式**（要求 7 种认知标签）→ StructPRM **评估 existing CoT**
- GRP 的 reward 是 **拓扑合规检查** → StructPRM 是 **结构贡献评分**
- GRP **不分析** DSR ⊥ correctness → 我们 **证明了** 这一点
- GRP **统一惩罚** 所有 dead exploration → 我们 **分类 7 种** dead step types
- GRP 是 **training method** (SFT+GRPO) → StructPRM 是 **evaluation method** (PRM)

一句话：GRP 改变了模型怎么说话，StructPRM 改变了我们怎么听。

---

## Part 2: Experiments + Predictions

每个实验都有明确的 hypothesis 和 predicted outcome。实验完成后对比 prediction vs
reality，分析差异原因，迭代改进。

### Exp 1: Best-of-N with All Selectors [✅ DONE]

**Hypothesis**: StructPRM 选出的 trace 结构更干净，但 accuracy 不会大幅提升
（因为 DSR ⊥ correctness）。Majority Voting 在 accuracy 上赢，但结构差。
Combined selector (MajVote + StructPRM) 应该两者兼得。

**Prediction**:
| Selector | Acc (predicted) | DSR (predicted) |
|:---------|:---------------:|:---------------:|
| Greedy | ~83% | ~0.20 |
| Majority Vote | ~92% | ~0.20 |
| StructPRM-L1 | ~84% | ~0.03 |
| MajVote + StructPRM | ~92% | ~0.04 |

**Actual Result (with fixed normalization)**:
| Selector | Acc (actual) | DSR (actual) | vs prediction |
|:---------|:------------:|:------------:|:-------------:|
| Greedy | 89.2% | 0.202 | Acc higher (norm fix) |
| Majority Vote | 93.6% | 0.199 | Close ✓ |
| StructPRM-L1 | 90.2% | 0.030 | Acc higher than expected |
| MajVote + StructPRM | 93.6% | 0.035 | Close ✓ |

**Analysis**: Predictions 方向正确。Greedy accuracy 高于预测是因为 answer
normalization fix（之前 82.8% 是 bug）。MajVote + StructPRM 确实实现了
"same accuracy, much cleaner structure"。

---

### Exp 2: Echo Trap Analysis [✅ DONE]

**Hypothesis**: 在 outcome reward 完全饱和的问题上（all K=8 correct），
StructPRM 仍有 variance，因为 structural quality varies even among correct traces。

**Prediction**:
- Saturated problems 占 ~70% (因为 4B 模型在 LIMO 上 pass@8 高)
- Outcome variance on saturated = 0 (by definition)
- StructPRM variance on saturated > 0 in >80% of problems
- Signal preservation ratio ~50-70%

**Actual Result**:
- Saturated: 384/500 (76.8%) — close to prediction ✓
- Outcome variance: 0 — as expected ✓
- StructPRM variance > 0: 95% of problems — **better than predicted** ✓
- Signal preservation: 66-68% — within prediction range ✓

**Analysis**: 预测准确。95% > 80% 说明 structural variance 比预期更 robust。

---

### Exp 3: Dead Step Taxonomy [✅ DONE]

**Hypothesis**: Dead steps 不是一种东西。有些是真浪费（verification theater），
有些是有价值的探索（productive dead ends）。

**Prediction**:
- Verification theater: ~20% of dead steps (most common waste)
- Productive dead ends: ~5-10% (exists but minority)
- Generic/unclassified: ~40-50%

**Actual Result**:
- Verification theater: 3.0% — **lower than predicted**
- Circular revisit: 17.0% — **new dominant category**
- Productive dead ends: 8.0% — within prediction ✓
- Generic: 62.1% — **higher than predicted**

**Analysis**: Verification theater 比预期少（可能是 L1 regex parser 漏检了）。
Circular revisit 是我们没预测到的主要类型。Generic 62% 说明 L1 parser 的分辨率
不够，需要 L2 parser（已验证 generic 降到 19%）。

---

### Exp 4: L1 vs L2 Parser [✅ DONE]

**Hypothesis**: L2 (LLM parser) 比 L1 (regex) 好得多，因为 LLM 能理解语义依赖。

**Prediction**:
- L1 和 L2 的 DSR 正相关但不完全一致 (r ≈ 0.5-0.7)
- L2 的 generic_dead 比 L1 低 (20-30% vs 50-60%)
- L2 在 Best-of-N accuracy 上可能略好

**Actual Result**:
- DSR correlation r(L1, L2) = **-0.01** — **完全不相关!** (远差于预期)
- L2 generic_dead: 19% vs L1 69% — L2 好得多 ✓
- L2 Best-of-N accuracy: 和 L1 一样 (52.3% on GPQA) — 无优势

**Analysis**: r = -0.01 是最 surprising 的发现。说明 L1 的 paragraph-level
segmentation (85 steps) 和 L2 的 semantic segmentation (17 steps) 看到的是
**完全不同的结构**。L1 的 symbol overlap edges 和 L2 的 logical dependency edges
几乎不相关。这意味着 L1 不是 "L2 的粗糙近似"，而是 "完全不同的东西"。

**Implication**: 论文需要诚实地讨论 L1 的局限性。L2 (LLM parser) 才是真正的
StructPRM。L1 只是一个 fast proxy，且 proxy quality 很差。

---

### Exp 5: GPQA Best-of-N [✅ DONE]

**Hypothesis**: GPQA 是 cross-domain science reasoning，exploration 更有价值。
StructPRM 可能不 work 因为它惩罚 exploration。

**Prediction**:
- StructPRM accuracy ≤ random（因为 low DSR ≠ correct on GPQA）
- Shortest 可能赢（incorrect traces longer on hard problems）
- DSR ⊥ correctness even more strongly on GPQA

**Actual Result**:
- StructPRM: 51.0% vs Random: 54.3% — **StructPRM 输了** ✓
- Shortest: 58.1% — **shortest 赢了** ✓
- r(DSR, correct) = -0.034 — very weak ✓

**Analysis**: 预测准确。GPQA 验证了 StructPRM 的定位：它度量 structural quality
而非 correctness。在 accuracy 维度上它不有优势，但在 structural quality 维度上
它是唯一的。

---

### Exp 6: Hint Injection [✅ DONE]

**Hypothesis**: Hints 可以把 "all-wrong" 问题变成 "partial"，扩展训练区间。

**Prediction**:
- Weak hints (answer format): ~5% solvable
- Strong hints (key reasoning step): ~20-30% solvable
- Self-hints: ~10% solvable

**Actual Result (after normalization fix)**:
- True all-wrong: 22 problems (not 55 — normalization bug)
- Strong GT hints: 1/22 (4.5%) — **far below predicted 20-30%**
- Self-hints: 1/22 (4.5%) — close to predicted for weak hints

**Analysis**: 预测失败。22 个真正做不出的题超出了 4B 的能力边界，即使给了
GT solution 的关键步骤也做不出来。这验证了文献的 consensus：
hints 不能弥补知识缺口。StructPRM 不应该 claim 能力扩展。

---

### Exp 7: Answer Normalization [✅ DONE — unexpected discovery]

**Not predicted** — 这是一个意外发现。

原来 55 个 "all-wrong" 问题中 33 个实际上答对了（145° vs 145, \frac43 vs
\frac{4}{3} 等格式差异）。修复后 pass@8 从 89.0% 涨到 95.6%。

**Impact**: 所有之前的数字都需要用修复后的 normalization 重新计算。
Greedy accuracy 从 82.8% 变为 89.8%。

---

## Part 3: Remaining Experiments + Predictions

### Exp 8: K=64 Best-of-N Scaling Curves [TODO]

**Hypothesis**: 随着 N 增大，Majority Voting 的 accuracy 持续上升（更多投票 →
更准确的 consensus）。StructPRM 的 accuracy 也上升但更慢（因为 more rollouts
中 structurally clean 的选择更多，但 clean ≠ correct）。MajVote + StructPRM
仍然是综合最优。

**Prediction**:
| Selector | N=8 | N=16 | N=64 |
|:---------|:---:|:----:|:----:|
| Majority Vote | 93.6% | 94.5% | 95.5% |
| StructPRM-L1 | 90.2% | 90.5% | 91.0% |
| MajVote + StructPRM | 93.6% | 94.5% | 95.5% |

**Reasoning**: Majority voting scales with √N (law of large numbers)。
StructPRM 选的是 structurally best 而非 most likely correct，所以 scaling 更慢。

**DSR prediction**:
| Selector | N=8 | N=16 | N=64 |
|:---------|:---:|:----:|:----:|
| MajVote + StructPRM DSR | 0.035 | 0.025 | 0.015 |

More rollouts → more chance of finding a very clean correct trace → DSR 继续下降。

**GPU cost**: ~8 GPU-hrs (4 shards K=64 + 3 shards K=16, all 1GPU×1hr)
**Status**: K=64 smoke test PASSED (124 prob/hr)

---

### Exp 9: DPO Signal Ablation [TODO]

**Hypothesis**: 用 structural signal 构建 DPO pairs 比 random/length/correctness
signal 更能降低 DSR 而不损害 accuracy。

**Prediction**:
| Signal | MATH Acc | DSR | GPQA |
|:-------|:--------:|:---:|:----:|
| No DPO (Stage 1) | 89.8% | 0.20 | ~49% |
| Random pairs | ~89% | ~0.18 | ~48% |
| Length-based | ~89% | ~0.15 | ~45% (探索被压缩) |
| Correctness-based | ~90% | ~0.18 | ~50% |
| **StructPRM-L1** | ~89% | ~0.10 | ~50% |

**Reasoning**:
- Random: DPO 有微弱的 regularization 效果但无方向性
- Length: 压缩推理长度 → DSR 降低但 GPQA 受损（探索被惩罚）
- Correctness: 标准 DPO，提升 accuracy 但不影响 structure
- StructPRM: 专门优化 structure → DSR 最低，GPQA 不受损（因为不惩罚长度）

**Key test**: 如果 StructPRM DSR < Length DSR 且 StructPRM GPQA ≥ Length GPQA，
说明 structural signal 确实比 length signal 好。

**GPU cost**: ~80 GPU-hrs (5 conditions × 4GPU × 3hr + eval)
**Status**: Pair building smoke test PASSED. Need 8B model fix first.
**Alternative**: 用 4B 做 ablation (model 完整)

---

### Exp 10: ProcessBench [TODO]

**Hypothesis**: StructPRM 在 ProcessBench 上的表现取决于 "dead step" 和 "error step"
的 overlap 程度。

ProcessBench 测 "找到第一个错误步骤"。StructPRM 找 "结构性无用步骤"。两者相关但
不完全对应：
- 一个错误步骤 → 后续依赖它的步骤都变 dead → StructPRM 能检测
- 一个 dead 但 correct 的步骤 → ProcessBench 不认为有错 → 不匹配

**Prediction**:
- StructPRM 在 "错误导致后续 dead" 的 cases 上表现好（~60-70% detection rate）
- StructPRM 在 "步骤本身就错" 的 cases 上表现差（~30-40%，因为它不检查 correctness）
- Overall ProcessBench accuracy: ~50-60%（低于 ThinkPRM 的 ~75%）

**Reasoning**: StructPRM 和 ThinkPRM 度量的是不同的东西（structure vs correctness）。
ProcessBench 偏向 correctness → ThinkPRM 应该赢。但 StructPRM 的价值不在 ProcessBench
分数，而在它提供了一个 **complementary** 维度。

**GPU cost**: ~2 GPU-hrs (scoring only)
**Status**: Dataset downloaded, need to design adapter

---

### Exp 11: ThinkPRM Baseline [TODO]

**Hypothesis**: ThinkPRM 在 accuracy-based selection 上赢 StructPRM（因为它度量
correctness），但在 structural quality 上输（因为它不度量 structure）。

**Prediction**:
| Selector | MATH Acc | DSR |
|:---------|:--------:|:---:|
| ThinkPRM Best-of-8 | ~92% | ~0.18 |
| StructPRM Best-of-8 | ~90% | ~0.03 |
| ThinkPRM + StructPRM | ~92% | ~0.04 |

**Reasoning**: ThinkPRM 选 "most likely correct" → high accuracy but no structural
preference. Combined: ThinkPRM selects correct, StructPRM selects cleanest among
correct → best of both worlds.

**GPU cost**: ~4 GPU-hrs (load ThinkPRM-1.5B, score K=8 rollouts)
**Status**: ThinkPRM-1.5B smoke test PASSED (loads in 127s)

---

### Exp 12: 8B Scaling [TODO — BLOCKED]

**Hypothesis**: StructPRM 的 findings 在 8B 上 replicate。8B 的 DSR baseline 更低
（16.4% vs 22.0%），所以 StructPRM 的 DSR reduction 可能不那么 dramatic。

**Prediction**:
- 8B echo trap: ~95% signal retention (same as 4B)
- 8B Best-of-N: similar pattern but smaller margins
- 8B DPO: accuracy drop ≤ 1pp with lr=2e-5 (vs lr=5e-5 的 -2.4pp)

**Blocker**: 8B stage1-merged model 不完整（缺 shard 4 + index）。需要重新 merge。

**GPU cost**: ~30 GPU-hrs (rollouts + DPO + eval)

---

## Part 4: Paper Tables (Predicted vs Actual)

### Table 1: Best-of-N Selection (MATH-500, 4B)

| Selector | N | Acc (pred) | Acc (actual) | DSR (pred) | DSR (actual) |
|:---------|:-:|:----------:|:------------:|:----------:|:------------:|
| Greedy | 1 | ~83% | 89.2% | ~0.20 | 0.202 |
| MajVote | 8 | ~92% | 93.6% | ~0.20 | 0.199 |
| StructPRM | 8 | ~84% | 90.2% | ~0.03 | 0.030 |
| MajVote+StructPRM | 8 | ~92% | 93.6% | ~0.04 | 0.035 |
| MajVote | 64 | ~95% | ? | ~0.18 | ? |
| MajVote+StructPRM | 64 | ~95% | ? | ~0.015 | ? |

### Table 2: Signal Ablation (DPO, 4B or 8B)

| Signal | MATH (pred) | DSR (pred) | MATH (actual) | DSR (actual) |
|:-------|:-----------:|:----------:|:-------------:|:------------:|
| Random | ~89% | ~0.18 | ? | ? |
| Length | ~89% | ~0.15 | ? | ? |
| Correctness | ~90% | ~0.18 | ? | ? |
| StructPRM | ~89% | ~0.10 | ? | ? |

### Table 3: StructPRM vs ThinkPRM (complementarity)

| Selector | Acc (pred) | DSR (pred) | Acc (actual) | DSR (actual) |
|:---------|:----------:|:----------:|:-------------:|:------------:|
| ThinkPRM | ~92% | ~0.18 | ? | ? |
| StructPRM | ~90% | ~0.03 | ? | ? |
| Think+Struct | ~92% | ~0.04 | ? | ? |

---

## Part 5: Key Figures

| Figure | Content | Data Source |
|:-------|:--------|:-----------|
| Fig 1 | Hero: Linear PRM vs StructPRM scoring | Qualitative example |
| Fig 2 | Echo trap: outcome vs structural variance | ✅ verify_structural_reward_variance.py |
| Fig 3 | Dead step taxonomy distribution | ✅ verify_dead_step_quality.py |
| Fig 4 | Scaling curves: accuracy/DSR vs N | TODO (needs K=16/64) |
| Fig 5 | Per-difficulty (L1-L5) analysis | ✅ existing eval data |
| Fig 6 | L1 vs L2 parser comparison | ✅ compare_parser_levels.py |

---

## Part 6: Timeline

| Week | Dates | Focus | Deliverables |
|:----:|:------|:------|:-------------|
| 1 | Apr 3-9 | Data gen + quick wins | K=16/64 rollouts, ThinkPRM scoring |
| 2 | Apr 10-16 | Core experiments | DPO ablation, ProcessBench |
| 3 | Apr 17-23 | Analysis + figures | All figures, scaling curves |
| 4 | Apr 24-30 | Paper draft | Full 8-page draft |
| 5 | May 1-7 | Internal review | Identify gaps |
| 6 | May 8-14 | Gap-filling | Additional experiments if needed |
| 7 | May 15-21 | Polish | Final revision |
| 8 | May 22-25 | Submit | ARR submission |

---

## Part 7: Iteration Protocol

After each experiment completes:
1. Compare actual result vs prediction
2. If match → hypothesis confirmed, move to next experiment
3. If mismatch → analyze WHY:
   - Is the hypothesis wrong? → update understanding
   - Is the implementation buggy? → fix and rerun
   - Is there an unexpected confound? → design follow-up experiment
4. Update predictions for remaining experiments based on new understanding
5. Repeat

This is gradient descent on our understanding of reasoning structure.
