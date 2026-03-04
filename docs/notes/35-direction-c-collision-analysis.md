# Direction C 撞车风险分析

**Date**: 2026-03-02  
**Purpose**: 系统评估Direction C (Structure-Informed Exploration Curriculum) 与2026现有工作的重叠和差异化

---

## 1. 潜在撞车对象全景

我识别出了以下几类可能与Direction C有重叠的工作：

### 类别A: Long-CoT Structural Pruning → SFT (与Stage 1重叠)

| Paper | Date | Core Method | 与我们的重叠度 |
|:------|:-----|:-----------|:------------|
| **Prune-on-Logic** (Zhao et al.) | May 2025 | DAG logic graph + perplexity-based pruning → SFT distillation | ⚠️ **HIGH** |
| **DRP** (Jiang et al.) | May 2025 | Skill-aware step decomposition → pruning → distillation | Medium |
| **DLCoT** (Luo et al.) | Mar 2026 | Segmentation + simplification + error optimization → distillation | Medium |
| **D-CoT** (Feb 2026) | Feb 2026 | Disciplined thought structure with control tags + SFT | Low-Medium |
| **TokenSqueeze** (Nov 2025) | Nov 2025 | SFT + length-regularized DPO for compression | Low |
| **STEP** (Jan 2026) | Jan 2026 | Hidden state probing for trace pruning (inference-time) | Low |

### 类别B: Efficient Reasoning via RL (与Stage 2/3重叠)

| Paper | Date | Core Method | 与我们的重叠度 |
|:------|:-----|:-----------|:------------|
| **DRPO** (Oct 2025) | Oct 2025 | Decoupled reward for accuracy vs length in GRPO | Medium |
| **DeCS** (Overthinking Reduction) (Sep 2025) | Sep 2025 | Decoupled rewards + curriculum data scheduling | ⚠️ **MEDIUM-HIGH** |
| **ReCUT** (Jun 2025) | Jun 2025 | Stepwise trails + DPO preference pairs for efficient reasoning | Medium |
| **BRIDGE** (Feb 2026) | Feb 2026 | 3-stage curriculum (reconstruction → GRPO compression → teacher internalization) | ⚠️ **HIGH (structural)** |
| **Light-R1** (Mar 2026) | Mar 2026 | Curriculum SFT + DPO + RL for long CoT | Medium |
| **Less is More Tokens** (Sep 2025) | Sep 2025 | Difficulty-aware SFT + DPO | Medium |

### 类别C: Step-Level Credit Assignment (与我们的structural credit idea重叠)

| Paper | Date | Core Method | 与我们的重叠度 |
|:------|:-----|:-----------|:------------|
| **GRPO is Secretly a PRM** (Sullivan, ICLR 2026) | Sep 2025 | GRPO隐式诱导step-level PRM | Low (theory, not training method) |
| **PRIME** (Feb 2025) | Feb 2025 | Implicit process rewards from outcome MC estimation | Low |
| **InT** (Jan 2026) | Jan 2026 | Counterfactual intervention for step credit | Low-Medium |
| **CAPO** (Aug 2025) | Aug 2025 | LLM critique-driven token credit | Low |
| **SPAE** (Jan 2026) | Jan 2026 | Step potential advantage estimation | Low-Medium |
| **Step-DPO** (Jun 2024) | Jun 2024 | Step-wise preference pairs for reasoning | Medium |
| **Full-Step-DPO** (Feb 2025) | Feb 2025 | Self-supervised PRM for step-wise DPO | Medium |

---

## 2. 逐一深度分析最高风险撞车对象

### 2.1 ⚠️ Prune-on-Logic (Zhao et al., May 2025) — **最高风险**

**他们做了什么**:
- 把Long-CoT转成DAG logic graph (用LLM做segmentation + connection assignment)
- 基于perplexity的node importance scoring
- 三种pruning策略: all-chain, reasoning-only, verification-only
- Verification pruning consistently improves SLM accuracy
- SFT distillation on pruned data

**与我们的重叠**:
- ✅ 都用DAG structure来分析reasoning traces
- ✅ 都发现verification pruning有益
- ✅ 都做pruned data → SFT distillation

**关键差异 (我们的unique contribution)**:
| 维度 | Prune-on-Logic | DecoR / Direction C |
|:-----|:--------------|:-------------------|
| **DAG构建方式** | LLM prompting (expensive, circular) | R-IR formal parser → backward reachability (principled) |
| **Pruning判据** | Perplexity-based importance scoring | **Structural reachability** (dead = unreachable from conclusion) |
| **理论框架** | 无 (empirical) | **Structural Reward Gap** + compiler theory analogy |
| **Domain analysis** | 只做math | Math + **GPQA cross-domain** (发现domain-dependent scaffolding value) |
| **Training beyond SFT** | ❌ 只做SFT | ✅ **SFT → Structural DPO** (Stage 2) |
| **Exploration quality** | ❌ 不涉及 | ✅ **核心thesis**: 教模型什么时候exploration是productive的 |
| **Agent extension** | ❌ 不涉及 | ✅ Agent DAE pilot data |
| **Correctness-independence** | ❌ 未发现 | ✅ **F2**: Dead steps与correctness统计独立 |
| **RL connection** | ❌ 不讨论 | ✅ 解释为什么RL产生superstitious reasoning |

**撞车评估**: Stage 1的"DSE-cleaned data → SFT"与Prune-on-Logic的verification pruning → SFT有**方法层面的相似性**。但：
1. 我们的pruning是formal reachability-based（不是perplexity-based）
2. 我们有complete theoretical framework (Structural Reward Gap)
3. **Direction C的核心novelty不在Stage 1，而在Stage 2 (Structural DPO)**
4. Prune-on-Logic完全不涉及preference learning或exploration quality

**结论**: **Stage 1有partial overlap，但Direction C的核心contribution (Stage 2) 完全不撞车。** 需要在paper中properly cite并differentiate。

---

### 2.2 ⚠️ BRIDGE (Feb 2026) — **结构撞车最高**

**他们做了什么**:
- 3-stage curriculum: 
  - Stage 1: Structure-aware reconstruction (shuffled + masked steps → reconstruct)
  - Stage 2: GRPO-based compression (correctness + brevity reward)
  - Stage 3: Teacher-guided internalization (hard cases)
- Tested on GSM8K with Qwen2.5-3B-Base
- +11.29% accuracy, -27.4% output length

**与我们的重叠**:
- ✅ 都是3-stage curriculum
- ✅ 都用structure-aware approach
- ✅ Stage 2都是reward-based optimization
- ✅ 目标都是efficient reasoning

**关键差异**:
| 维度 | BRIDGE | Direction C |
|:-----|:-------|:-----------|
| **Stage 1目标** | 学会reconstruct logical structure (shuffle+mask→reconstruct) | 学会efficient reasoning (DSE-cleaned SFT) |
| **Stage 2方法** | GRPO with correctness+brevity reward (length-based) | **Structural DPO** (reachability-based preference) |
| **Stage 2 signal** | Length penalty (shorter = better IF correct) | **Structural quality** (fewer dead steps = better, length-independent) |
| **Stage 3** | Teacher-guided internalization of hard cases | Difficulty-adaptive exploration curriculum |
| **核心insight** | Compression via length optimization | **Exploration quality** via structural credit |
| **理论基础** | Curriculum learning + capacity alignment | **Structural Reward Gap** + exploration quality |
| **Domain analysis** | ❌ 只GSM8K | Math + GPQA + Agent |
| **Structural signal定义** | 无 (只有length) | **DAG reachability** (live/dead per step) |

**最关键的区别**: 
- BRIDGE的Stage 2 reward = accuracy + **length penalty**。它鼓励shorter traces。
- Our Stage 2 signal = **structural quality** (dead step ratio)。它鼓励structurally efficient traces。
- **Length ≠ Structure**: 一条短trace可能全是dead steps (e.g., 错误推导链)。一条长trace可能全是live steps (e.g., GPQA跨domain验证)。
- BRIDGE会惩罚所有长trace，包括genuinely useful long reasoning。
- 我们只惩罚structurally dead parts，保留productive long reasoning。

**撞车评估**: **High structural overlap (都是3-stage curriculum for efficient reasoning)**，但**mechanism和theoretical grounding完全不同**。
- BRIDGE = compression-focused (shorter is better)
- Ours = quality-focused (structurally efficient is better, regardless of length)
- 这个区别在GPQA上最明显: BRIDGE会hurt GPQA (penalize long deliberation)。我们不会 (只penalize dead deliberation)。

**结论**: **需要在paper中明确differentiate**。建议正面比较: "BRIDGE optimizes for brevity; we optimize for structural efficiency. These are fundamentally different objectives, as demonstrated by GPQA performance."

---

### 2.3 DeCS (Overthinking Reduction, Sep 2025) — **MEDIUM-HIGH**

**他们做了什么**:
- 理论分析RLVR的overthinking problem
- Decoupled rewards: 分离accuracy reward和length reward
- Curriculum data scheduling: 逐步增加难度
- 解决reward hacking问题

**与我们的重叠**:
- ✅ 都address overthinking/inefficiency in reasoning
- ✅ 都用curriculum approach
- ✅ 都decoupled reward signals

**关键差异**:
- DeCS的length signal是token count。我们的structural signal是DAG reachability。
- DeCS完全在RL domain。我们的Stage 2用DPO (更稳定，更practical)。
- DeCS没有structural analysis of traces。
- DeCS没有domain-dependent analysis (GPQA vs Math)。

**撞车评估**: **Medium overlap in motivation, but different mechanism and theoretical framing.** Length vs Structure is the key differentiation.

---

### 2.4 DLCoT (Mar 2026) — **MEDIUM**

**他们做了什么**:
- Deconstruct Long-CoT into segments
- Simplify: eliminate unsolvable + redundant solutions
- Optimize intermediate error states
- SFT distillation

**与我们的重叠**: Similar data cleaning approach for SFT. But:
- No DAG/reachability analysis
- No preference learning
- No exploration quality concept
- No domain-dependent analysis

**撞车评估**: **Medium overlap with Stage 1 only.** Direction C's core contribution unaffected.

---

### 2.5 Step-DPO / Full-Step-DPO — **MEDIUM**

**他们做了什么**:
- Step-DPO: 在reasoning chain的第一个错误步骤处构建preference pairs
- Full-Step-DPO: 用self-supervised PRM给每步打分，构建step-wise preference pairs

**与我们的重叠**:
- ✅ 都用step-level preference pairs + DPO
- ✅ 都analyze step quality

**关键差异**:
- Step-DPO的preference基于**correctness** (correct step > incorrect step)
- 我们的preference基于**structural efficiency** (structurally live > structurally dead)
- 这是完全不同的signal:
  - Step-DPO教模型"不要犯逻辑错误"
  - 我们教模型"不要做structurally wasteful exploration"
- Step-DPO不涉及verification/exploration的productive vs wasteful区分

**撞车评估**: **Low-Medium. Different signal source, different training objective.**

---

## 3. 综合撞车风险评估

### 3.1 Direction C的Component-Level Risk

| Component | 撞车风险 | 最近竞争者 | 我们的差异化 |
|:----------|:-------:|:----------|:-----------|
| **Stage 1: DSE-SFT** | ⚠️ **HIGH** | Prune-on-Logic, DLCoT, DRP | Formal reachability vs perplexity scoring; R-IR vs LLM-prompted graph |
| **Stage 2: Structural DPO** | ✅ **LOW** | (No direct competitor) | 唯一一个用DAG reachability构建preference pairs的工作 |
| **Stage 3: Difficulty Curriculum** | ⚠️ **MEDIUM** | BRIDGE, DeCS, Light-R1 | Structure-based vs length-based curriculum |
| **Theoretical Framework** | ✅ **LOW** | (No direct competitor) | Structural Reward Gap + compiler theory analogy |
| **Domain-Dependent Analysis** | ✅ **LOW** | (No one does Math vs GPQA structural analysis) | Exploration quality is domain-dependent (F7) |
| **Agent Extension** | ✅ **LOW** | OPRL does implicit step rewards for agents | We use structural DAG, they use MC estimation |

### 3.2 总体风险

**Stage 1 (DSE-SFT) 的撞车风险是HIGH**——但这不是Direction C的核心contribution。

**Direction C的核心contribution (Stage 2: Structural DPO) 的撞车风险是LOW。**

没有任何现有工作做以下事情：
1. 用**DAG reachability** (not perplexity, not LLM judge, not outcome MC)来定义step quality
2. 基于structural live/dead来构建**preference pairs**
3. 区分**productive vs wasteful** exploration/verification
4. 利用**domain-dependent** exploration quality insight (Math: verification是waste; GPQA: verification是cure)

### 3.3 风险定性评估

```
Overall collision risk for Direction C: MEDIUM-LOW

- If we only do Stage 1 (DSE-SFT): HIGH collision risk (crowded space)
- If we do Stage 1 + Stage 2: LOW collision risk (Stage 2 is unique)  
- If we do Stage 1 + Stage 2 + Stage 3: LOW collision risk (full story is unique)
```

---

## 4. 差异化策略

### 4.1 如何在paper中定位Direction C

**NOT this framing** (会撞车):
> "We prune reasoning traces and train with cleaner data"
→ 这跟Prune-on-Logic, DLCoT, DRP, TokenSqueeze都撞

**NOT this framing** (会撞车):
> "We use curriculum learning to make reasoning more efficient/shorter"
→ 这跟BRIDGE, DeCS, Light-R1, Less-is-More-Tokens都撞

**✅ THIS framing (unique)**:
> "We discover that outcome-based RL is structurally blind to exploration quality 
> (dead steps are correctness-independent). We propose Structural Preference 
> Optimization: using DAG reachability as a non-outcome-derived preference signal 
> to teach models WHEN exploration is productive vs wasteful. This addresses the 
> Structural Reward Gap that causes Verification Theater in RL-trained models."

关键差异化点:
1. **Signal来源**: DAG reachability (deterministic, non-outcome-derived) vs perplexity/length/LLM-judge
2. **Training目标**: Exploration quality (structural efficiency) vs length/compression/correctness
3. **理论贡献**: Structural Reward Gap explains WHY RL creates superstitious reasoning
4. **Domain insight**: Exploration value is domain-dependent (unique empirical finding)

### 4.2 必须cite和compare的work

| Paper | How to cite | How to differentiate |
|:------|:-----------|:--------------------|
| Prune-on-Logic | "Prior work uses perplexity-based pruning on DAG structure" | "We use formal reachability analysis, and crucially, go beyond SFT to structural preference optimization" |
| BRIDGE | "BRIDGE proposes 3-stage curriculum for compression" | "BRIDGE optimizes for brevity; we optimize for structural efficiency. On GPQA, brevity hurts but structural efficiency preserves performance" |
| DeCS | "DeCS decouples accuracy and length rewards" | "We decouple accuracy and STRUCTURAL QUALITY rewards — length is a poor proxy for quality" |
| Step-DPO | "Step-DPO builds step-level preference pairs based on correctness" | "We build preference pairs based on structural reachability — correct steps can still be dead (F2)" |
| GRPO-as-PRM | "GRPO implicitly assigns step-level credit" | "But this implicit credit is blind to correctness-independent waste (our F2) — structural credit fills this gap" |

### 4.3 Ablation that proves our uniqueness

**关键实验**: Compare structural preference vs length preference
```
Experiment: DPO with different preference signals
  (a) Length preference: prefer correct+shorter over correct+longer
  (b) Structural preference: prefer correct+low-DSR over correct+high-DSR
  (c) Combined: prefer correct+short+low-DSR over correct+long+high-DSR

Expected results:
  - On MATH: (a) ≈ (b) (因为shorter ≈ less waste for math)
  - On GPQA: (b) >> (a) (因为structural preference保留useful long deliberation,
                         而length preference会kill it)
  - This single experiment definitively separates us from ALL length-based methods
```

---

## 5. 最终评估

### Direction C的competitive position

**优势 (unique contributions)**:
1. ✅ Structural Reward Gap theory (unique to DecoR)
2. ✅ DAG reachability as preference signal (nobody else does this)
3. ✅ Domain-dependent exploration quality (Math vs GPQA, unique finding)
4. ✅ Structural DPO (Stage 2, no competitor)
5. ✅ Agent DAE extension (no competitor in structural agent analysis)

**劣势 (crowded aspects)**:
1. ⚠️ Stage 1 (pruning→SFT) is crowded
2. ⚠️ Multi-stage curriculum is becoming common
3. ⚠️ "Efficient reasoning" framing is crowded

**Risk mitigation**:
- 不要lead with Stage 1 (pruning→SFT)。Lead with structural preference signal。
- 不要frame as "efficient reasoning" or "compression"。Frame as "exploration quality"。
- 用GPQA实验作为killer differentiation (length-based methods hurt GPQA; we don't)。

### 我的最终结论

**Direction C的撞车风险: MEDIUM-LOW (可以接受)**

Stage 1有partial overlap但不是我们的novelty来源。Direction C的真正novelty在Stage 2 (Structural Preference DPO)，这个方向**没有直接竞争者**。

**前提条件**: 必须正确framing paper。如果frame成"又一个efficient reasoning方法"，reviewer会说"incremental over Prune-on-Logic/BRIDGE/DeCS"。如果frame成"structural credit for exploration quality"，这是一个全新的research direction。

**建议**:
1. Paper title不要包含"efficient", "compression", "pruning"等词
2. 强调exploration quality, structural credit, domain-dependent scaffolding
3. 用GPQA vs MATH的对比作为core selling point
4. Lead with Structural Reward Gap theory, not with the method

**推荐paper titles**:
- *"Learning When to Explore: Structural Preference Optimization for Reasoning Quality"*
- *"Beyond Outcome Rewards: Structural Reachability as Preference Signal for Reasoning"*
- *"The Exploration Quality Gap: Why RL Models Waste Verification and How to Fix It"*
