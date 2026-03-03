# 从本质出发：如何让Agent在RL中学会高质量Exploration

**Date**: 2026-03-02
**Context**: 基于DecoR全部empirical findings + 2026最新RL进展的深度分析

---

## 1. 重新审视我们到底发现了什么

让我把DecoR的所有发现串成一条逻辑链：

### 发现链

```
[F1] Verification Theater: RL模型73-88%的verification是structurally dead的
     ↓ 意味着什么？
[F2] Correctness-Independence: Dead步骤与正确性统计独立 (p > 0.05)
     ↓ 意味着什么？
[F3] RL比SFT产生5.6×更多dead verification
     ↓ 意味着什么？
[F4] Agent DAE: 成功trajectory仍有34.6% dead actions
     ↓ 意味着什么？
[F5] 成功agent比失败agent更高效: edit+execute多，explore+error少
     ↓ 意味着什么？
[F6] Math: 更多exploration → 更低accuracy (在每个difficulty level)
[F7] GPQA: 更多verification → 更高accuracy (与math相反)
     ↓ 综合意味着什么？
```

### 本质洞察

**模型不知道exploration的质量。** 具体来说：

1. **RL强化了exploration的数量，但没有强化exploration的质量。**
   - outcome reward只能告诉模型"你最终对了/错了"
   - 它无法告诉模型"你的第3步verification是有用的，但第7步verification是浪费"
   - 结果：模型学会了"多验证 → 安全"的superstitious pattern

2. **GRPO的implicit PRM无法区分productive和wasteful exploration。**
   - Sullivan et al. (2026, ICLR) 证明GRPO隐式地是一个PRM
   - 但这个implicit PRM的credit assignment依赖于group内trajectory overlap
   - Dead steps（verification theater）在correct和incorrect traces中都出现（F2）
   - 因此implicit PRM给dead steps的reward ≈ 0（无法区分好坏），而不是负reward
   - **Dead steps不是被鼓励的——它们是被忽视的。** RL既不惩罚也不奖励它们。
   - 它们持续存在是因为SFT阶段的imitation + RL阶段的entropy preservation

3. **真正的问题不是"remove waste"而是"learn what productive exploration looks like"。**
   - DSE是post-hoc cleanup — 它清理数据但不改变模型理解exploration quality的能力
   - 我们需要的是一个training signal让模型learn:
     - "当你不确定时，验证是好的"（GPQA场景）
     - "当你已经在正确路径上时，验证是浪费"（Math场景）
     - "当你连续3次explore无果时，应该backtrack"（Agent场景）

---

## 2. 连接到2026年RL研究前沿

### 2.1 Step-Level Credit Assignment (最热方向)

| Paper | Method | 与我们的连接 |
|:------|:-------|:------------|
| **GRPO is Secretly a PRM** (Sullivan, ICLR 2026) | GRPO隐式地是step-level PRM | 但这个implicit PRM无法区分dead/live steps（因为correctness-independent） |
| **PRIME** (Feb 2026) | Implicit process reward通过online PRM | 不用structural signal，只用outcome的MC estimate |
| **InT** (Jan 2026) | Counterfactual intervention找到关键步骤 | 最接近我们的idea——但用resampling而非structural analysis |
| **CAPO** (Aug 2025) | Critique-driven token-level credit | LLM judge判断每个token的价值——expensive & circular |
| **MIG** (Feb 2026) | Step-wise Marginal Information Gain | 量化每步的信息增益——不考虑structural dependency |
| **TACReward** (Oct 2025) | Process mining for reasoning-aware reward | 最接近structural approach——但用process mining而非DAG reachability |

### 2.2 关键缺口

**所有现有step-level credit assignment方法都是 outcome-derived 或 LLM-judged。**
没有任何方法使用**structural dependency**作为credit signal。

- PRIME: Monte Carlo estimation from outcomes → 仍然是outcome-derived
- InT: Counterfactual resampling → expensive (100 rollouts per step)
- CAPO: LLM critique → circular (LLM judging LLM)
- TACReward: Process mining → closer to us but no DAG reachability

**我们的R-IR + DSE提供了一个全新的、non-outcome-derived、deterministic的credit signal：**
- Step是live → 它structurally contribute to the answer → positive credit
- Step是dead → 它structurally disconnected → zero/negative credit
- 这是**免费的**（不需要额外rollouts或LLM judges）

### 2.3 Agent Exploration的前沿

| Paper | 核心问题 | 关键发现 |
|:------|:---------|:--------|
| **RAGEN/StarPO** (2025) | Agent RL训练的framework | 发现exploration collapse：agent倾向于shortcut行为 |
| **LaMer** (Meta-RL, 2025) | 让agent学会explore | Cross-episode training + in-context adaptation |
| **Agent-R1** (2025) | End-to-end agent RL | Tool-integrated reasoning RL |
| **OPRL** (2025) | Online PRM for agents | 用implicit step rewards做agent credit assignment |
| **SWE-Pruner** (Jan 2026) | Inference-time trajectory pruning | Selective context compression for coding agents |

**Agent exploration的核心困境（直接呼应我们的F4-F5发现）：**
- 成功agent: 34.6% dead actions（大量无用exploration仍被保留）
- 失败agent: 43.5% dead actions（更多waste）
- 高DAR trajectory dominated by exploration (65% explore)
- 低DAR trajectory balanced (execute+edit 55%)

**没有任何agent training工作在trajectory level做structural optimization。**
所有人都在做binary outcome filtering (success/fail)。

---

## 3. 核心研究方向：Structural Credit Assignment for Exploration Quality

### 3.1 核心thesis

> **Outcome-based RL无法学会exploration quality，因为outcome reward无法观察
> reasoning/action的internal dependency structure。Structural dependency 
> analysis (R-IR + backward reachability) 提供了一个complementary的、
> non-outcome-derived的credit signal，可以直接告诉模型哪些exploration步骤
> 是productive的（structurally connected to conclusion），哪些是wasteful的
> （structurally dead）。**

这不是post-hoc data cleaning（DSE的approach）。
这是**在RL训练过程中给模型一个structural feedback signal**。

### 3.2 为什么这比现有方法更好

| 方法 | Signal来源 | Cost | Circularity | 能区分productive/wasteful exploration? |
|:-----|:----------|:-----|:-----------|:-------------------------------------|
| GRPO (outcome only) | Final answer | 0 | No | ❌ — correctness-independent (F2) |
| PRIME (implicit PRM) | MC estimate from outcomes | Low | No | ❌ — 仍是outcome-derived |
| InT (counterfactual) | Resampling | 100× rollouts | No | ✅ 但太expensive |
| CAPO (LLM critique) | LLM judge | 1× LLM call/step | Yes (LLM judging LLM) | Partially |
| **Structural Credit (ours)** | **DAG reachability** | **Regex classifier** | **No** | **✅ — deterministic** |

**我们的structural credit是唯一一个：**
1. 非outcome-derived（不依赖答案是否正确）
2. 非LLM-judged（不需要另一个LLM来评判）
3. Deterministic（给定trace，credit assignment是唯一确定的）
4. Cheap（regex-based classification + DAG reachability，不需要API calls）

### 3.3 Structural Credit如何工作

```
Training Loop:
1. Model generates trajectory (reasoning trace / agent actions)
2. Check outcome: correct/incorrect → outcome reward R_outcome

3. [NEW] Parse trajectory structure:
   a. Segment into steps (paragraph-level for reasoning, action-level for agents)
   b. Classify step types (regex-based: computation, derivation, verification, exploration...)
   c. Build dependency DAG (heuristic rules: sequential + content overlap)
   d. Backward reachability from conclusion → live/dead classification
   
4. Compute structural credit per step:
   R_structural(step_i) = {
     +α  if step_i is live (structurally productive)
     -β  if step_i is dead AND type ∈ {computation, derivation}  (true waste)
      0  if step_i is dead AND type ∈ {verification, exploration} (ambiguous scaffolding)
   }

5. Combined reward per step:
   R_step(i) = R_outcome + λ × R_structural(i)
   
6. RL update with step-level rewards (GRPO/PPO with per-step advantage)
```

**关键设计决策：**
- Dead verification/exploration gets 0 (neutral), not negative
  → 避免杀死cross-domain deliberation (F7: GPQA需要verification)
- Dead computation/derivation gets negative
  → 惩罚真正的waste (F6: 更多exploration → 更低accuracy on math)
- Live steps get positive bonus
  → 鼓励productive, connected reasoning
- λ is a hyperparameter controlling structural signal strength

### 3.4 为什么不需要LLM parser (关键可行性)

之前R-IR parsing用DeepSeek API（昂贵，有latency）。但我们在automated_typed_dse.py中
已经证明**regex-based classification达到85%+ accuracy**。

对于RL training中的structural credit，我们不需要完美的parsing：
- 段落分割: `\n\n` splitting（trivial）
- 类型分类: regex keywords（"verify", "check", "let me", "therefore"）
- 依赖关系: sequential chain + content overlap（shared numbers/variables）
- Reachability: 标准BFS from last paragraph

这个pipeline在CPU上每条trace < 10ms，完全可以在RL training loop中real-time运行。

---

## 4. 三个具体研究方向（按ambition排序）

### Direction A: Structural Credit RL for Reasoning (最直接)

**Setup**: 
- Base model: Qwen2.5-Math-7B (or Qwen3-8B)
- RL: GRPO on MATH/NuminaMath training set
- Baseline: Standard GRPO (outcome only)
- Ours: GRPO + structural credit (per-step reward bonus based on R-IR reachability)

**Compute**: ~400-600 GPU-hrs (2-3 RL runs × 7B)

**Expected results**:
- Standard GRPO: high accuracy but 15-30% dead steps (our F1)
- Structural GRPO: similar/higher accuracy AND lower dead step ratio
- 模型learns to produce structurally efficient traces

**Why this works**: 
- Standard GRPO的implicit PRM无法区分dead/live steps (因为correctness-independent)
- 加入structural credit后，dead steps有了explicit negative signal
- 模型learns: "I can get the same outcome with fewer dead steps → higher total reward"

**Novelty**: ★★★★★ — 首次在RL training中使用structural dependency作为reward signal
**Risk**: ★★★☆☆ — RL training instability + reward balancing (λ tuning)
**Feasibility**: ★★★☆☆ — 需要modify GRPO training loop + implement fast structural parser

---

### Direction B: Structural Credit RL for Agents (最有影响力)

**Setup**:
- Base model: Qwen2.5-Coder-7B (or Qwen3-8B)  
- RL: StarPO/GRPO on coding tasks (SWE-Gym subset or ARES tasks)
- Agent-R-IR parser: our existing pipeline (sequential + file-based dependencies)
- Dead action types: unused exploration, error actions, redundant views

**Structural credit for agents**:
```
R_structural(action_i) = {
  +α  if action leads to file that gets edited (productive exploration)
  -β  if action views file never edited (unused exploration)
  -γ  if action produces error (failed action)
  0   if action is thinking/planning (neutral)
}
```

**Compute**: ~600-800 GPU-hrs (agent RL is expensive due to environment interaction)

**Expected results**:
- Standard agent RL: high DAR (~35%) in successful trajectories
- Structural agent RL: lower DAR, higher efficiency, similar/better success rate
- Agent learns to explore more targeted: view fewer irrelevant files, make fewer errors

**This directly addresses your core question**: 模型在RL过程中学会了什么是productive exploration。
- View a file → edit it → productive (positive structural credit)
- View a file → never use it → wasteful (negative structural credit)
- Run a command → get useful output → productive
- Run a command → get error → wasteful (but recovery from error → productive)

**Novelty**: ★★★★★ — 完全新的方向
**Risk**: ★★★★☆ — Agent RL环境复杂，需要sandbox + tool execution
**Feasibility**: ★★☆☆☆ — Agent RL infrastructure最复杂

---

### Direction C: Structure-Informed Exploration Curriculum (最务实 + 最深入的insight)

**核心insight**: 与其在RL loop中做structural credit（复杂），不如用structural analysis
来**设计更好的训练curriculum**。

**我们的发现告诉我们exploration quality的pattern：**

```
From Agent DAE (F4-F5):
- 高效agent: view-to-edit ratio低, explore占比35%, execute+edit占55%
- 低效agent: view-to-edit ratio高, explore占比65%, execute+edit占30%

From Math DSE (F1-F3, F6):
- 正确解答: 更少exploration, 更直接的derivation chain
- 错误解答: 更多exploration, 更多dead verification
- DSE-trained model: 学会了直接推理，但失去了deliberation能力

From Exploration Quality (F7):
- Math: verification是symptom (正相关with failure)
- GPQA: verification是cure (正相关with success)
```

**方法: 三阶段Exploration Curriculum**

```
Stage 1: Teach Direct Reasoning (SFT on DSE-cleaned data)
  - 让模型首先学会"在知道答案路径时，不waste tokens"
  - 这是DSE的核心价值 — 我们已经证明它works (F6)
  - 结果: 模型在简单问题上efficient, 但可能在难题上under-explore

Stage 2: Teach When to Explore (RL with structural credit)
  - 在Stage 1 model上做RL training
  - 用structural credit reward:
    - 当exploration leads to correction → 正向reward
    - 当exploration leads to dead end → 负向reward
    - 当exploration leads to verification that's referenced → 正向reward
  - 结果: 模型learns *when* exploration is productive

Stage 3: Teach How to Explore (RL with harder problems + exploration bonus)
  - 逐步增加问题难度 (curriculum)
  - 在hard problems上, exploration gets higher structural bonus
  - 在easy problems上, direct reasoning gets bonus
  - 结果: 模型learns difficulty-adaptive exploration strategy
```

**为什么三阶段有意义**:
- Stage 1 alone = 当前DSE (math好，GPQA差)
- Stage 2 alone = 当前GRPO (exploration无向)
- Stage 1+2 = 先学efficient reasoning, 再学when to explore
- Stage 1+2+3 = 完整的exploration quality curriculum

**Compute**: ~800-1200 GPU-hrs (3 stages × 7B × 2-3 runs per stage)

**Novelty**: ★★★★★ — 三阶段curriculum基于structural analysis设计
**Risk**: ★★☆☆☆ — Stage 1是我们已证明的, Stage 2是moderate的SFT+RL
**Feasibility**: ★★★★☆ — 不需要online structural parsing在RL loop中

---

## 5. 最佳方案选择

### 关键考量

| 因素 | Direction A | Direction B | Direction C |
|:-----|:----------:|:----------:|:----------:|
| 研究深度 | ★★★★☆ | ★★★★★ | ★★★★★ |
| 创新性 | ★★★★★ | ★★★★★ | ★★★★☆ |
| 20天可行性 | ★★★☆☆ | ★★☆☆☆ | ★★★★☆ |
| Paper story | 强 | 最强 | 最完整 |
| 与DecoR synergy | 直接延伸 | 新paper | 直接延伸+新paper |
| Compute fit (2400 GPU-hrs) | 刚好 | 略紧 | 充裕 |

### 推荐方案

**Direction C (Structure-Informed Exploration Curriculum) 作为primary。**

理由：
1. **Stage 1我们已经做了** — DSE-cleaned SFT就是Stage 1，已有4B results
2. **Stage 2 是core novelty** — 用structural credit做RL，但不需要online parsing
   - 可以pre-compute structural labels on a set of rollouts
   - 然后做offline RL (DPO on structurally-annotated preference pairs)
   - 这避免了online RL的instability
3. **Stage 3 是icing** — difficulty curriculum, 如果时间够就做
4. **Paper story极其清晰**:
   > "We diagnose that RL models cannot learn exploration quality because 
   > outcome rewards are blind to structural dependency (DecoR findings). 
   > We propose a three-stage curriculum: (1) SFT on structurally cleaned 
   > data teaches efficient reasoning, (2) RL with structural credit teaches 
   > when exploration is productive, (3) difficulty curriculum teaches 
   > adaptive exploration. Each stage addresses a specific limitation."

### 具体执行计划 (20天, 2400 GPU-hrs)

**Week 1 (Day 1-7): Infrastructure + Stage 1 scaling**
- BriCS环境搭建 (NGC container, aarch64)
- Port training pipeline
- Scale Stage 1 (DSE-SFT) to 7B and 14B → 5 runs × ~20h = 100h
- Implement fast structural parser (regex-based) → test on LIMO data
- Pre-compute structural annotations on NuminaMath/MATH rollouts

**Week 2 (Day 8-14): Stage 2 — Structural Credit**
- Generate rollouts from Stage 1 model (1000 traces per difficulty level)
- Parse each trace with fast structural parser → compute live/dead per step
- Construct preference pairs:
  - Preferred: correct + low structural waste
  - Dispreferred: correct + high structural waste (or incorrect)
- DPO training on preference pairs → 3 runs × ~30h = 90h
- Eval on MATH-500, GPQA, AIME → check if exploration quality improves

**Week 3 (Day 15-20): Stage 3 + Analysis + Writing**
- If Stage 2 works: implement difficulty-adaptive curriculum
- Structural behavior analysis: do Stage 2 models generate fewer dead steps?
- Cross-domain eval: does GPQA performance recover?
- Compare with baselines: Standard GRPO, PRIME, DSE-SFT-only
- Start writing paper

**Compute breakdown**:
| Task | GPU-hrs |
|:-----|--------:|
| Stage 1: DSE-SFT scaling (7B, 14B × 5 variants) | 500 |
| Rollout generation (1000 traces × 3 difficulty levels) | 100 |
| Stage 2: DPO with structural credit (7B, 14B × 3 variants) | 400 |
| Stage 3: Curriculum RL (if time permits) | 400 |
| Eval (all models × 4 benchmarks) | 200 |
| Structural analysis (parse generated traces) | 100 |
| Buffer/re-runs | 700 |
| **Total** | **2400** |

---

## 6. 这篇paper的贡献

### Title candidates
- *"Learning When to Explore: Structural Credit Assignment for Reasoning Efficiency"*
- *"Beyond Outcome Rewards: Structural Dependency as Credit Signal for Reasoning RL"*
- *"From Structural Diagnosis to Structural Training: Closing the Exploration Quality Gap"*

### Contributions
1. **Diagnosis → Prescription**: 从DecoR的structural diagnosis出发，提出actionable training method
2. **Structural credit**: 首次使用DAG reachability作为RL credit signal（非outcome-derived，非LLM-judged）
3. **Exploration curriculum**: 三阶段训练从efficient reasoning到adaptive exploration
4. **Empirical**: 在reasoning + agent tasks上验证

### 与DecoR的关系
- DecoR = diagnosis paper ("RL models have structural waste, here's why")
- 这篇 = prescription paper ("here's how to fix it during training")
- 完美的two-paper story

### Target venues
- NeurIPS 2026 (deadline ~May)
- ICML 2026 (deadline ~Feb — 已过)
- ICLR 2027 (deadline ~Oct)
- EMNLP 2026 (deadline ~Jun)

---

## 7. 最深层的insight

回到你的问题：**如何让agent在RL中学会高质量exploration？**

答案不是一个单一的technique，而是一个认识论的转变：

**现有paradigm**: Outcome → Reward → Policy Update
- 模型只知道"我最终对了/错了"
- 无法区分"因为探索发现了关键线索而对了" vs "虽然浪费了很多但碰巧对了"

**我们提出的paradigm**: Outcome + Structure → Reward → Policy Update  
- 模型知道"我最终对了，而且我的exploration是productive的"
- 可以区分"直接推导出答案 (efficient)" vs "绕了很多弯但答案对 (wasteful)"
- 可以区分"验证发现了错误并纠正 (productive exploration)" vs "验证了已知正确的结果 (waste)"

**本质上，structural dependency analysis给了模型一个它之前没有的perception channel：**
> 它现在不仅能感知"结果对不对"，还能感知"过程中每一步是否structurally connected to the result"。

这就像给一个只能看到棋局最终结果的棋手，额外提供了每一步的形势评估。
棋手（模型）可以学会在优势明显时快速出击（直接推理），在劣势时谨慎探索（productive verification）。
