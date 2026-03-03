# 三个方向深度分析

**Date**: 2026-03-02  
**Purpose**: 展开分析三个structural credit方向的理论基础、技术设计、预期结果、风险和实现细节

---

## 前提回顾：共享的理论基础

三个方向共享同一个核心洞察：

**Outcome reward的credit assignment盲区**

```
标准GRPO的credit assignment路径:
  Group of K traces → 每条trace得到outcome reward {0,1} 
  → Group normalization → 每条trace的advantage = R_i - mean(R)
  → 整条trace的所有token共享同一个advantage

问题出在"整条trace的所有token共享同一个advantage"。
```

Sullivan et al. (ICLR 2026) 证明GRPO隐式诱导了step-level credit：
- 如果step S在多条correct trace中出现，但也在多条incorrect trace中出现 → credit ≈ 0
- 如果step S主要在correct trace中出现 → positive credit
- 如果step S主要在incorrect trace中出现 → negative credit

**我们的F2 (correctness-independence) 直接揭示了这个机制的失败点：**
Dead steps在correct和incorrect traces中以统计相同的比例出现 (p > 0.05)。
因此GRPO的implicit PRM给dead steps的credit ≈ 0。
Dead steps既不被鼓励也不被惩罚——它们是RL训练的**盲区**。

**这解释了F3 (RL产生5.6×更多dead verification)**：
- SFT阶段通过imitation学会了"有时候需要verification"
- RL阶段的outcome reward无法区分productive/wasteful verification
- Entropy preservation + KL penalty保持了verification的频率
- 但没有gradient signal来refine WHEN to verify
- 结果：verification frequency 膨胀但quality不变 → "Verification Theater"

**三个方向的区别在于：在训练pipeline的哪个环节、用什么方式注入structural signal。**

---

## Direction A: Structural Credit RL for Reasoning

### A.1 理论基础

**核心idea**: 在GRPO/PPO的reward function中加入structural component。

标准GRPO:
```
R(trace) = Outcome(trace)  ∈ {0, 1}
```

Structural GRPO:
```
R(trace) = Outcome(trace) + λ · StructuralQuality(trace)

where StructuralQuality(trace) = 1 - DSR(trace)  (Dead Step Ratio)
```

**为什么这能work？**

考虑GRPO的group comparison。给定同一个problem的K个rollout:
- Trace A: correct, DSR=0.15 → R = 1 + λ(0.85) = 1 + 0.85λ
- Trace B: correct, DSR=0.45 → R = 1 + λ(0.55) = 1 + 0.55λ  
- Trace C: incorrect, DSR=0.30 → R = 0 + λ(0.70) = 0.70λ
- Trace D: incorrect, DSR=0.50 → R = 0 + λ(0.50) = 0.50λ

Group mean ≈ (1+0.85λ + 1+0.55λ + 0.70λ + 0.50λ) / 4

关键：**Trace A比Trace B有更高的advantage**，即使两个都是correct的。
这就打破了标准GRPO中"所有correct traces获得相同advantage"的问题。
模型被鼓励不仅要答对，还要用structurally efficient的方式答对。

更细粒度的版本：step-level structural reward
```
R_step(s_i) = {
  +α  if s_i is live (backward-reachable from conclusion)
  -β  if s_i is dead AND type ∈ {computation, derivation}
   0  if s_i is dead AND type ∈ {verification}  (ambiguous)
}
```
这用PRIME/CAPO的step-level advantage framework来实现。

### A.2 技术设计

**Pipeline**:
```
1. Model generates trace for problem p
2. Extract answer → check correctness → R_outcome ∈ {0,1}
3. Segment trace into paragraphs (split on \n\n)
4. Classify each paragraph type (regex-based, <10ms)
5. Build lightweight dependency DAG:
   - Sequential chain (each paragraph depends on previous)
   - Content overlap edges (shared variable names / numbers)
   - Conclusion = last paragraph with \boxed{}
6. Backward reachability from conclusion → live/dead per paragraph
7. Compute R_structural per step
8. Combined advantage for GRPO update
```

**关键组件**：

**(a) Fast Structural Parser (在RL loop中real-time运行)**

我们在`automated_typed_dse.py`中已有regex classifier。
需要额外实现的：
- Paragraph segmentation: `re.split(r'\n\n+', trace)` — trivial
- Lightweight dependency DAG: sequential + content overlap
- Backward reachability: standard BFS — O(n) where n = #paragraphs

整个parser在CPU上 < 10ms/trace。RL training一个step通常几秒到几十秒，
所以parser的overhead < 1%。不需要GPU，不需要API。

**(b) 修改GRPO training loop**

需要基于一个RL framework修改reward computation。
候选frameworks:
- **veRL** (Bytedance, 2025): Efficient RLHF framework, 支持GRPO
  - 优点: 成熟、文档好、multi-GPU scaling
  - 缺点: 不确定ARM/aarch64支持
- **OpenRLHF** (Jan 2026): 支持GRPO+PPO
  - 优点: 活跃社区
  - 缺点: 配置复杂
- **TRL** (HuggingFace): 最简单的GRPO实现
  - 优点: 最容易修改
  - 缺点: Scaling差

修改点（以veRL为例）：
```python
# 在reward computation中加入structural component
def compute_reward(completions, outcomes):
    rewards = []
    for trace, outcome in zip(completions, outcomes):
        r_outcome = float(outcome)  # 0 or 1
        
        # Structural analysis
        paragraphs = segment(trace)
        types = [classify_paragraph(p) for p in paragraphs]
        dag = build_dependency_dag(paragraphs)
        live_mask = backward_reachability(dag, conclusion_idx)
        
        dsr = 1 - sum(live_mask) / len(live_mask)
        r_structural = 1 - dsr  # Higher is better
        
        rewards.append(r_outcome + lambda_ * r_structural)
    return rewards
```

### A.3 实验设计

| Experiment | Model | Method | Training Data | GPU-hrs |
|:-----------|:------|:-------|:-------------|--------:|
| Baseline 1 | Qwen3-8B | Standard GRPO (outcome only) | NuminaMath-7.5k | ~120 |
| Baseline 2 | Qwen3-8B | PRIME (implicit PRM) | NuminaMath-7.5k | ~120 |
| **Ours** | Qwen3-8B | Structural GRPO (λ=0.1) | NuminaMath-7.5k | ~120 |
| **Ours** | Qwen3-8B | Structural GRPO (λ=0.3) | NuminaMath-7.5k | ~120 |
| **Ours** | Qwen3-8B | Structural GRPO (λ=0.5) | NuminaMath-7.5k | ~120 |
| Analysis | All | Structural behavior analysis | — | ~20 |
| **Total** | | | | **~620** |

Eval: MATH-500, AIME 2025, AIME 2026, GPQA Diamond

**关键metrics** (beyond accuracy):
- Dead Step Ratio of generated traces (does structural GRPO reduce DSR?)
- Verification density (tokens of verification per trace)
- Trace length (does structural GRPO produce shorter traces?)
- GPQA accuracy (does structural GRPO preserve cross-domain deliberation?)

### A.4 预期结果与Story

**最佳情况**:
- Structural GRPO在MATH上略高于标准GRPO (+1-3%)
- Structural GRPO的DSR显著低于标准GRPO (15% vs 30%)
- Trace length更短 → 更高效inference
- GPQA不降（因为dead verification credit = 0, 不是negative）

**Story**: 
> "Standard outcome-based RL is blind to structural waste because dead steps 
> are correctness-independent (cannot be learned from outcomes alone). By adding 
> a structural signal based on DAG reachability, RL can learn to produce traces 
> that are both correct AND structurally efficient."

**最差情况**:
- RL training不稳定（reward shaping干扰了learning signal）
- Structural parser的noise导致错误的credit signal
- λ太大 → 模型学会了极短但wrong的traces (reward hacking)

### A.5 风险分析

| Risk | Probability | Impact | Mitigation |
|:-----|:----------:|:------:|:-----------|
| RL training instability | Medium | High | 用小λ (0.1), 大KL penalty |
| ARM/aarch64 RL framework兼容 | Medium | High | 提前1天测试; fallback到TRL |
| Structural parser noise | Low | Medium | Regex已证明85%+ acc |
| Reward hacking (trace太短) | Medium | Medium | 加length penalty; cap minimum length |
| RL需要大量rollouts (sample efficiency) | High | Medium | 减少training problems, 增加per-problem rollouts |
| 20天不够debug | High | High | 这是最大风险 |

### A.6 时间线

| Day | Task |
|:----|:-----|
| 1-3 | BriCS环境搭建 + RL framework安装 (veRL or TRL on ARM) |
| 3-5 | Implement fast structural parser + unit tests |
| 5-6 | Modify reward computation in GRPO loop |
| 6-8 | Baseline GRPO run (to ensure pipeline works) |
| 8-12 | Structural GRPO runs (3 λ values) |
| 12-14 | PRIME baseline + eval all models |
| 14-17 | Structural behavior analysis |
| 17-20 | Analysis + writing |

### A.7 总结

**最大优势**: 这是最direct的方式在RL training中注入structural signal。如果work，story极其clean。

**最大劣势**: RL training pipeline的engineering complexity。需要：
1. RL framework在ARM上跑通
2. GRPO training稳定
3. Structural parser在RL loop中不引入overhead
4. reward balancing不trivial

**适合的人**: 有RL training经验、有足够debug时间、愿意冒高风险高回报的研究者。

---

## Direction B: Structural Credit RL for Agents

### B.1 理论基础

**核心idea**: 把Direction A的structural credit从reasoning trace扩展到agent trajectory。

Agent的exploration quality问题比reasoning更严重（我们的数据证明了这一点）：

| Metric | Math Reasoning | Agent Trajectory | Ratio |
|:-------|:---:|:---:|:---:|
| Dead content rate | 29% | **38.7%** | 1.3× |
| Savings per trace | ~5.4k chars | **~20.3k chars** | 3.8× |
| Inference cost impact | ~$0.01 | **$0.50-5.00** | 50-500× |

**为什么agent exploration quality更难学？**

1. **Action space更大**: Reasoning只有"写文字"一个action。Agent有open_file, edit, execute, search, finish等多个工具。Exploration = 用哪个工具 + 在哪个target上。
   
2. **Environment stochasticity**: Math是deterministic的。Agent的environment有不确定性（test可能flaky，file可能被其他进程修改）。

3. **Long horizon**: Math reasoning ~10-30 steps。Agent trajectory ~20-50+ actions。Credit assignment over longer horizons更难。

4. **Binary outcome even more misleading**: 一个SWE-bench task要么pass所有test（success）要么不pass（failure）。但一个"successful" trajectory中可能30%的actions是waste。

**Structural Credit for Agents**:

```
对每个action a_i in trajectory:

R_structural(a_i) = {
  +α  if a_i.type == 'edit' and a_i.file_path ∈ final_patch_files
  +α  if a_i.type == 'execute' and a_i.output informs a later edit
  +α  if a_i.type == 'explore' and a_i.file_path ∈ edited_files
  -β  if a_i.type == 'explore' and a_i.file_path ∉ edited_files  (unused exploration)
  -β  if a_i.has_error                                             (failed action)
  -γ  if a_i is redundant view of already-viewed file              (redundant)
   0  if a_i.type == 'planning'                                    (neutral)
}
```

这比reasoning的structural credit更intuitive：
- 你查看了一个文件然后编辑了它 → productive exploration
- 你查看了一个文件然后再也没用到 → wasteful exploration
- 你运行了一个命令并得到了有用信息 → productive
- 你运行了一个命令但报错了 → wasteful

### B.2 技术设计

**Pipeline (StarPO/GRPO for agents)**:

```
1. Agent interacts with sandbox environment (Docker + repository)
2. Agent generates trajectory: [think, action, observation, think, action, ...]
3. Check outcome: tests pass? → R_outcome ∈ {0,1}

4. [NEW] Parse trajectory with Agent-R-IR:
   a. Extract actions: (tool_name, file_path, command)
   b. Build dependency DAG:
      - Sequential chain
      - File-based: explore(file) → edit(same file)
      - Error-based: error → next action (recovery)
   c. Backward reachability from finish/submit action
   d. Heuristic dead patterns: unused explore, error actions, redundant views
   
5. Compute per-action structural reward
6. StarPO/GRPO update with step-level advantage
```

**需要的infrastructure**:

1. **Agent RL training framework**: ARES (Martian, 2026) 或 Agent-R1 或 RAGEN
   - ARES: Open-source, 支持SWE-bench, massively parallel rollouts
   - Agent-R1: Simpler, end-to-end GRPO for agents
   - RAGEN/StarPO: Most general, but complex

2. **Sandbox environment**: Docker containers for SWE-bench tasks
   - 每个task需要一个隔离的Docker container
   - 需要test execution来validate
   - ARM上Docker support? → 需要验证

3. **Agent-R-IR parser**: 我们已有 (`experiments/agent_rir.py`)
   - 但需要适配real-time agent trajectory format
   - 不同agent框架的trajectory format不同

### B.3 实验设计

| Experiment | Model | Method | Tasks | GPU-hrs |
|:-----------|:------|:-------|:------|--------:|
| Baseline | Qwen3-8B | Standard StarPO (outcome only) | SWE-Gym 500 | ~200 |
| **Ours** | Qwen3-8B | Structural StarPO (per-action credit) | SWE-Gym 500 | ~200 |
| Ablation | Qwen3-8B | Structural StarPO (only error penalty) | SWE-Gym 500 | ~200 |
| Eval | All | SWE-bench Verified (500 tasks) | | ~150 |
| **Total** | | | | **~750** |

**但实际compute会更多**:
- Agent rollouts比reasoning长很多（每trajectory 20-100 actions × environment interaction）
- SWE-bench evaluation需要run tests in Docker（wallclock time dominating）
- 估计实际需要 ~1200-1500 GPU-hrs + 大量wallclock time

### B.4 预期结果与Story

**最佳情况**:
- Structural StarPO在SWE-bench上高于标准StarPO (+3-5% resolve rate)
- Agent的DAR显著降低 (25% vs 35%)
- Agent生成的trajectories更短（fewer wasted exploration steps）
- View-to-edit ratio降低（agent learns to explore more targeted）
- Token cost per task降低 (节省30-40%的inference cost)

**Story**:
> "We extend structural credit assignment from reasoning to agent trajectories.
> Agent actions have clear structural dependencies (explore → edit → execute),
> enabling deterministic identification of productive vs wasteful actions.
> By rewarding structurally efficient trajectories, agents learn to explore
> more targeted, fail less, and solve problems with fewer wasted actions."

**最差情况**:
- Agent RL training不稳定（比reasoning RL更不稳定）
- SWE-bench环境在ARM上跑不起来
- Resolve rate没有提高（structural credit被outcome signal dominate）
- 20天不够跑完eval（SWE-bench evaluation极慢）

### B.5 风险分析

| Risk | Probability | Impact | Mitigation |
|:-----|:----------:|:------:|:-----------|
| ARM上Docker/SWE-bench不work | **High** | **Critical** | 提前Day 1测试; 如果不行整个direction放弃 |
| Agent RL training极不稳定 | High | High | 用ARES (已有稳定pipeline) |
| Eval wallclock time太长 | High | High | 只eval 100 tasks subset |
| Compute > 2400 GPU-hrs | Medium | High | 减少rollouts per problem |
| Agent-R-IR parser accuracy不够 | Low | Medium | 我们已有validated pipeline |

### B.6 时间线

| Day | Task |
|:----|:-----|
| 1 | **关键决策点**: Docker + SWE-bench on ARM测试。如果失败 → 转Direction A或C |
| 1-4 | Agent RL framework安装 (ARES or Agent-R1 on ARM) |
| 4-6 | Adapt Agent-R-IR parser for real-time trajectory parsing |
| 6-8 | Implement structural credit computation |
| 8-12 | Baseline + Structural StarPO training |
| 12-16 | SWE-bench evaluation (需要大量wallclock) |
| 16-18 | Structural analysis of generated trajectories |
| 18-20 | Analysis + writing |

### B.7 总结

**最大优势**: 
- **Impact最大**: Agent training是2026年最热方向，SWE-bench是金标准。结果一出来就会有高关注。
- **Novelty最高**: 没有任何prior work在agent training中做structural trajectory optimization with RL reward。
- **ROI最大**: Agent trajectory的waste成本比reasoning高50-500×。每1%的efficiency提升都有极大practical value。
- **与DecoR完美互补**: DecoR诊断math reasoning的waste → 这篇诊断+治疗agent的waste。

**最大劣势**: 
- **Engineering极重**: RL framework + Docker sandbox + SWE-bench environment + ARM适配。任何一个环节出问题都是blocker。
- **20天可能不够**: Agent RL training cycle比reasoning长得多。
- **评估极慢**: SWE-bench evaluation per model ~50-100 GPU-hrs + 大量wallclock。

**适合的人**: 有agent training经验、有SWE-bench evaluation pipeline、deadline宽裕（60+ days）的团队。

**我的判断**: 这是最佳paper idea，但**不适合20天deadline + BriCS首次使用**的场景。建议作为follow-up paper的方向。可以用BriCS做data preparation（parse OpenHands 67k trajectories），但training和eval留到后续。

---

## Direction C: Structure-Informed Exploration Curriculum

### C.1 理论基础

**核心idea**: 不在RL loop中实时做structural credit（太复杂），而是用structural analysis的insights来设计一个multi-stage training curriculum，让模型逐步学会exploration quality。

**关键洞察（来自我们的empirical evidence）**:

我们的数据实际上已经告诉我们"good exploration"长什么样：

```
From F5 (Agent):   高效agent = edit+execute多, explore+error少
From F6 (Math):    更少exploration → 更高accuracy (每个difficulty level)  
From F7 (GPQA):    更多verification → 更高accuracy (domain-dependent)
From doc/29:       Exploration quantity ≠ quality. 问题在reasoning direction.
From doc/29:       DSE wins when Original gets lost in useless exploration.
                   Original wins when DSE model's reasoning PATH was wrong.
```

**综合这些发现，model需要学三个能力（按顺序）**:

1. **不浪费** (Efficiency): 当你知道怎么做时，不要waste tokens在无用verification上
   → 这就是DSE教给模型的

2. **知道什么时候该explore** (Metacognition): 当你不确定时，exploration是productive的；当你确定时，exploration是wasteful的
   → 这需要structural signal — DSE做不到，因为它是binary的

3. **知道怎么explore** (Strategy): 当你决定explore时，选择最productive的方向
   → 这需要difficulty-adaptive策略

**三阶段对应三个能力**:

```
Stage 1 (SFT on DSE-cleaned data): 学会"不浪费"
  ↓ 模型已有efficient reasoning baseline
Stage 2 (DPO with structural preference): 学会"什么时候该explore"  
  ↓ 模型学会了context-dependent exploration
Stage 3 (Curriculum RL): 学会"怎么explore"
  ↓ 模型学会了difficulty-adaptive exploration strategy
```

### C.2 技术设计

#### Stage 1: DSE-SFT (已验证)

这就是我们已经做过的实验。在DSE-cleaned LIMO数据上做SFT。

- 已证明在MATH上+3.2%, 训练快28%, loss低11.5%
- 已知限制: GPQA -6.6%, MATH L5 -4.5%
- **Stage 1的作用**: 给模型一个efficient reasoning的baseline

**Implementation**: 直接复用现有pipeline。
```
Config: training/configs/qwen3_8b_dse.yaml
Data: data/limo/cleaned/limo_dse.json
```

#### Stage 2: Structural Preference DPO (核心novelty)

**核心idea**: 从Stage 1模型生成大量rollouts，用structural analysis标注preference pairs，然后DPO训练。

**Step 2a: 生成rollouts**
```
Model: Stage 1 checkpoint (DSE-SFT model)
Problems: MATH training set (7.5k problems) + NuminaMath subset
Per problem: 生成K=8个traces (temperature=0.7)
Total: ~60k traces
```

**Step 2b: Structural annotation**
```
For each trace:
  1. Check correctness → correct/incorrect
  2. Segment into paragraphs
  3. Classify paragraph types (regex)
  4. Build dependency DAG
  5. Backward reachability → live/dead per paragraph
  6. Compute metrics:
     - DSR (Dead Step Ratio)
     - VDR (Verification Dead Ratio) 
     - Live Verification Rate (verifications that ARE referenced)
     - Exploration Efficiency (exploration steps that lead to corrections)
```

**Step 2c: Construct preference pairs**

这是最关键的设计。不是简单的"correct > incorrect"，而是：

```
Preference Pair Types:

Type 1: Structural Efficiency
  Preferred:  correct + low DSR (< 0.15)
  Rejected:   correct + high DSR (> 0.35)
  → 教模型: 在能直接推导时不要waste

Type 2: Productive Exploration
  Preferred:  correct + contains live verification (verification that leads to correction)
  Rejected:   correct + contains only dead verification
  → 教模型: 验证应该是productive的（发现错误并纠正），不是ritual

Type 3: Exploration Direction  
  Preferred:  correct + exploration leads to new derivation path
  Rejected:   incorrect + lots of exploration but no useful derivation
  → 教模型: exploration应该有方向性

Type 4: Cross-domain Deliberation
  From GPQA-style problems (if available):
  Preferred:  correct + verification touches multiple domains
  Rejected:   incorrect + verification loops in same domain
  → 教模型: 在cross-domain问题上，deliberation是productive的
```

**Step 2d: DPO training**
```
Framework: LLaMA-Factory (已有DPO支持)
Base model: Stage 1 checkpoint
Preference data: ~10k-30k pairs
Training: 1-3 epochs, lr=1e-6, β_DPO=0.1
```

**为什么用DPO而不是online RL？**

| 方面 | Online RL (Direction A) | Offline DPO (Direction C Stage 2) |
|:-----|:----------------------|:-------------------------------|
| 需要RL loop | ✅ 需要 | ❌ 不需要 |
| Structural parser实时运行 | 需要 | 不需要（离线预计算） |
| Training stability | 差（RL不稳定） | 好（DPO = supervised） |
| Framework complexity | 需要veRL/TRL修改 | LLaMA-Factory原生支持 |
| Sample efficiency | 差（需要大量rollouts） | 好（每对数据都被利用） |
| Theoretical guarantee | 更强（on-policy） | 较弱（off-policy bias） |
| ARM兼容性 | 不确定 | ✅ 确定（SFT-style training） |

**DPO的关键优势: 完全复用我们现有的LLaMA-Factory SFT pipeline。**
只需要把数据从SFT格式转成DPO preference pair格式。

#### Stage 3: Difficulty-Adaptive Curriculum (可选)

**核心idea**: 在Stage 2基础上，用curriculum learning让模型学会根据problem difficulty调整exploration策略。

```
Phase 3a: Easy problems (MATH L1-L3)
  → Reward: accuracy + structural efficiency (penalize dead steps more)
  → 模型学会: 在简单问题上直接推导

Phase 3b: Medium problems (MATH L4)
  → Reward: accuracy + moderate structural penalty
  → 模型学会: 适度exploration

Phase 3c: Hard problems (MATH L5 + GPQA-style)
  → Reward: accuracy only (不惩罚structural waste)
  → 模型学会: 在难题上放开exploration

Implementation options:
  (a) Rejection sampling + SFT on difficulty-stratified data
  (b) Online GRPO with difficulty-dependent λ
  (c) DPO with difficulty-conditional preference pairs
```

**Option (c) 最可行**: 为每个difficulty level构建不同的preference pair distribution。

### C.3 实验设计

| Phase | Task | Model | GPU-hrs | Notes |
|:------|:-----|:------|--------:|:------|
| **Stage 1** | DSE-SFT | Qwen3-8B | 20 | 复用现有pipeline |
| **Stage 1** | DSE-SFT | Qwen3-14B | 40 | Scale up |
| Rollout Gen | Generate 60k traces | Stage 1 model | 80 | K=8 per problem, temp=0.7 |
| Annotation | Structural parsing | CPU only | 0 | <10ms/trace, 60k traces < 10min |
| Pair Construction | Build preference pairs | CPU only | 0 | |
| **Stage 2** | Structural DPO (β_DPO=0.05) | Qwen3-8B | 25 | |
| **Stage 2** | Structural DPO (β_DPO=0.1) | Qwen3-8B | 25 | |
| **Stage 2** | Structural DPO (β_DPO=0.2) | Qwen3-8B | 25 | |
| **Stage 2** | Structural DPO (best β) | Qwen3-14B | 50 | |
| **Stage 3** (optional) | Difficulty-adaptive DPO | Qwen3-8B | 50 | |
| Baselines | Standard SFT (Original LIMO) | Qwen3-8B | 20 | |
| Baselines | Standard DPO (correct > incorrect, no structural) | Qwen3-8B | 25 | |
| Baselines | DSE-SFT + standard DPO | Qwen3-8B | 25 | |
| Eval | 4 benchmarks × 10+ models | All | 100 | |
| Structural Analysis | Parse generated traces from each model | All | 40 | |
| **Total** | | | **~525-575** |

**剩余budget (~1800 GPU-hrs) 用于**:
- Scale DecoR to 14B (5 variants × 40h = 200h) → strengthen COLM paper
- Re-runs and ablations (500h)
- Reserve for Stage 3 or Direction A fallback (1100h)

### C.4 预期结果与Story

**最佳情况 (Stage 2 works)**:

| Model | MATH-500 | GPQA | DSR | Trace Length |
|:------|:--------:|:----:|:---:|:-----------:|
| Original SFT | 69.6% | 55.6% | ~30% | 18.7k chars |
| DSE-SFT (Stage 1) | 72.8% | 49.0% | ~15% | 15.3k chars |
| **Structural DPO (Stage 2)** | **73-75%** | **53-56%** | **~12%** | **14-16k chars** |

关键预期:
- **MATH-500**: Stage 2 ≥ Stage 1 (72.8%), 可能更高 (preference learning refines reasoning)
- **GPQA**: Stage 2 > Stage 1 (49.0%), 回到接近Original (55.6%)
  → 因为Stage 2的preference pairs教模型在不确定时保留verification
- **DSR**: Stage 2 < Stage 1 < Original (模型学会了什么时候verification是有用的)
- **Trace Length**: Stage 2 ≈ Stage 1 (efficient), 但verification更targeted

**Story**:
> "DSE-cleaned SFT teaches efficient reasoning but loses deliberation capacity.
> We propose Structural Preference DPO: using DAG reachability to annotate
> preference pairs that teach models WHEN exploration is productive.
> The result: models that are efficient on in-domain math (like DSE)
> while preserving cross-domain deliberation (like Original)."

**更强的story (如果Stage 3也works)**:
> "We present a three-stage exploration curriculum that teaches reasoning models
> to (1) reason efficiently, (2) know when to explore, and (3) adapt exploration
> to difficulty. Each stage is grounded in empirical findings from structural
> analysis: Stage 1 addresses the Structural Reward Gap, Stage 2 addresses
> correctness-independent waste, Stage 3 addresses domain-dependent scaffolding."

### C.5 风险分析

| Risk | Probability | Impact | Mitigation |
|:-----|:----------:|:------:|:-----------|
| DPO preference pairs quality不够 | Medium | High | 多种pair type + filtering |
| Structural parser noise | Low | Low | 85%+ regex accuracy已验证 |
| Stage 2不如Stage 1 | Low-Medium | Medium | Stage 1本身就是valid baseline |
| GPQA不recover | Medium | Medium | 仍然是informative (explains the trade-off) |
| ARM环境SFT/DPO不work | Low | High | LLaMA-Factory是standard SFT, ARM support好 |
| Rollout generation太慢 | Low | Medium | 用vLLM batch inference |

**与Direction A/B相比，风险profile显著更好**:
- 不需要online RL (避免了最大的instability风险)
- 不需要Docker sandbox (避免了ARM兼容性风险)
- 完全基于SFT/DPO (LLaMA-Factory已验证)
- 每个Stage独立有价值 (即使Stage 3失败, Stage 1+2仍然是paper)

### C.6 时间线

| Day | Task | GPU-hrs | Risk |
|:----|:-----|--------:|:-----|
| 1-2 | BriCS环境搭建: NGC container, LLaMA-Factory, vLLM on ARM | 10 | Medium |
| 2-3 | 验证SFT pipeline on ARM (小模型smoke test) | 10 | Low |
| 3-4 | **Stage 1**: DSE-SFT for Qwen3-8B + Qwen3-14B | 60 | Low (已验证) |
| 4-5 | Implement fast structural parser (optimize regex classifier) | 0 | Low |
| 5-7 | **Rollout generation**: 60k traces from Stage 1 model (vLLM) | 80 | Low |
| 7-8 | Structural annotation + preference pair construction | 0 | Low |
| 8-11 | **Stage 2**: Structural DPO (3 β values × 8B + best × 14B) | 125 | **Medium** (核心) |
| 11-13 | Eval all models (Stage 1 + Stage 2 + baselines) | 100 | Low |
| 13-15 | Structural behavior analysis (parse traces from each model) | 40 | Low |
| 15-17 | Stage 3 (if Stage 2 works): difficulty-adaptive DPO | 50 | Medium |
| 17-20 | Analysis, ablations, writing | 50 | Low |
| Parallel | Scale DecoR 14B (5 variants) for COLM paper | 200 | Low |
| Reserve | Re-runs, debugging, additional ablations | ~1175 | — |
| **Total** | | **~1900** | |

### C.7 与Direction A的关键区别

```
Direction A: 在RL训练时实时计算structural credit
  → 优点: On-policy, theoretically optimal
  → 缺点: 需要修改RL loop, reward不稳定, ARM兼容性不确定

Direction C Stage 2: 离线计算structural signal, 转化为preference pairs, 用DPO训练
  → 优点: 完全复用SFT pipeline, 稳定, ARM友好
  → 缺点: Off-policy (preference pairs来自Stage 1 model, not current model)
  
两者的theoretical relationship:
  - Direction A ≈ Online structural reward + GRPO
  - Direction C Stage 2 ≈ Offline structural reward + DPO
  - 在理想情况下 A > C (on-policy更optimal)
  - 在实践中 C > A (因为DPO更稳定, 更容易debug, 更少engineering overhead)
```

### C.8 总结

**最大优势**:
- **Engineering最简单**: 完全基于SFT + DPO, 复用现有LLaMA-Factory pipeline
- **Risk最低**: 每个Stage独立有价值，失败一个不影响其他
- **Modular story**: 每个Stage对应一个empirical finding, paper结构清晰
- **20天可行性最高**: 不需要RL framework, 不需要Docker, 不需要ARM适配RL
- **与DecoR COLM paper synergy最好**: Stage 1结果直接strengthen COLM paper

**最大劣势**:
- **Off-policy**: DPO的preference pairs来自Stage 1 model, 不是当前model
  → 如果Stage 2 model的behavior drift很大, preference pairs可能过时
  → 但DPO的β参数限制了drift, 实践中通常不是大问题
- **Novelty略低于Direction A**: "结构化preference pairs + DPO"不如"结构化reward + GRPO"那么technically impressive
  → 但empirical findings和three-stage story仍然有充分novelty
- **不是真正的"在RL中学会exploration"**: 是offline learning from structural signal, 不是online discovery
  → 但对practical impact来说, 这个区别可能不重要

**我的核心判断**: Direction C是**最佳risk-reward trade-off**。
- 它用最少的engineering overhead获得了structural credit的核心value
- 即使Stage 2只带来marginal improvement, Stage 1 + structural analysis仍然是valid COLM contribution
- 如果Stage 2效果显著, 可以follow up Direction A (online version) 作为next paper

---

## 三方向对比总结

| 维度 | Direction A | Direction B | Direction C |
|:-----|:----------:|:----------:|:----------:|
| **理论深度** | ★★★★★ | ★★★★★ | ★★★★☆ |
| **创新性** | ★★★★★ | ★★★★★ | ★★★★☆ |
| **影响力** | ★★★★☆ | ★★★★★ | ★★★★☆ |
| **20天可行性** | ★★★☆☆ | ★★☆☆☆ | **★★★★★** |
| **Engineering复杂度** | High | Very High | **Low** |
| **ARM兼容风险** | Medium | High | **Low** |
| **失败后salvage value** | Low | Low | **High** (每Stage独立) |
| **与DecoR synergy** | 延伸 | 新paper | **直接strengthen** |
| **Paper venue** | NeurIPS/ICML | NeurIPS/ICML | COLM + NeurIPS |
| **Compute需求** | ~620h | ~1200h+ | **~575h** |
| **适合场景** | 长期, 有RL经验 | 长期, 有agent infra | **20天deadline** |

### 我的最终建议

**主线: Direction C** — 用BriCS的20天做Structure-Informed Exploration Curriculum

- Stage 1: DSE-SFT scaling (已验证, 直接跑)
- Stage 2: Structural Preference DPO (核心novelty, 风险可控)
- Stage 3: Difficulty-adaptive curriculum (icing, 如果时间够)
- Parallel: Scale DecoR to 14B for COLM paper

**后续: Direction B** — BriCS到期后在CSD3或其他资源上做Agent Structural Credit

- 用BriCS的时间提前准备: parse OpenHands 67k trajectories, 设计Agent preference pairs
- Agent RL training + SWE-bench eval在后续资源上完成
- Target: NeurIPS 2026 (deadline ~May) 或 EMNLP 2026 (deadline ~Jun)

**远期: Direction A** — 如果Direction C Stage 2效果好, 做online version

- 这是Direction C的"理论完善版"
- 用online GRPO + structural reward验证on-policy是否比off-policy DPO更好
- Target: ICLR 2027

### 三篇paper的路线图

```
Paper 1 (COLM 2026, deadline Mar 31):
  DecoR — Structural Reward Gap diagnosis + DSE cleaning
  → 已有数据, Direction C Stage 1 scaling到14B加强

Paper 2 (NeurIPS 2026, deadline ~May):
  Structural Exploration Curriculum — Direction C full paper
  → Stage 1 + Stage 2 + Stage 3 + structural behavior analysis

Paper 3 (NeurIPS 2026 或 EMNLP 2026):
  Agent DAE — Direction B
  → Structural Credit for Agent Training on SWE-bench

Paper 4 (ICLR 2027):
  Online Structural Credit — Direction A
  → On-policy structural reward for RLVR
```
