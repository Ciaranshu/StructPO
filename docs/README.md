# StructPRM: Graph-Structured Process Reward Models for Reasoning

> **Project pivot (2026-03-28)**: Renamed from StructPO to StructPRM. Core contribution
> elevated from "structural DPO pair construction" to "the first graph-structured
> Process Reward Model." All existing DPO results become one application of StructPRM.

## Core Narrative

**All existing Process Reward Models are Linear-Thinking.**

PRM800K, Math-Shepherd, ThinkPRM, GenPRM, CRM, PRIME — every PRM published to date
treats reasoning as a linear sequence s₁ → s₂ → ... → sₜ. They answer "is this step
correct?" but cannot answer **"does this step structurally contribute to the answer?"**

This blind spot has concrete consequences:
- **82-96% of verification steps** in RL-trained models are structurally dead (DecoR)
- **DSR ⊥ correctness** (r ≈ -0.21): correct steps can be completely useless
- **Outcome reward saturates** on easy problems → zero gradient → echo trap → collapse

**StructPRM is the first PRM that models reasoning as a DAG.** It uses backward
reachability to score each step's structural contribution, providing reward signal
that is orthogonal to both correctness and length.

### Key Findings

| Finding | Evidence |
|:--------|:---------|
| All PRMs are linear | 18+ papers surveyed, PRM Survey (2025) confirms as open problem |
| DSR ⊥ correctness | r ≈ -0.21 on K=8 rollout data (6,536 traces) |
| Echo trap breaking | 95% of saturated problems retain StructPRM signal (outcome: 0%) |
| Signal preservation | StructPRM retains 66-68% of variance under outcome saturation |
| Dead step taxonomy | 7 quality types; 8% are productive dead ends (should NOT be penalized) |
| Not length-based | 27% of structurally preferred traces are LONGER than rejected |

### Three-Level Structural Reward

| Level | Signal | Speed | RL-compatible? |
|:------|:-------|:------|:--------------:|
| Level 0 | Raw DSR (uniform penalty) | <10ms | ✅ |
| Level 1 | Quality-aware (regex + graph rules) | <10ms | ✅ |
| Level 2 | LLM-annotated quality (DecoR parser) | ~3s | ❌ (offline) |
| Level 3 | Distilled StructParser-0.5B | ~20ms | ✅ |

### Collapse Unification

All 2026 collapse phenomena share one root cause — **structurally blind reward**:

```
Easy problems → reward saturation → variance=0 → dead steps accumulate ("over-explore")
Hard problems → outcome-only gradient → converge to single pattern ("over-exploit")
                    ↓                              ↓
              Both caused by: reward cannot distinguish structural quality
                    ↓
              StructPRM: provides structural signal in BOTH regimes
```

## Predecessor: DecoR (COLM 2026)

DecoR diagnosed the **Structural Reward Gap**: outcome-based RL reinforces structurally
wasteful reasoning because dead steps are in the gradient's blind spot (expected
gradient ≈ 0). StructPRM is the prescriptive follow-up: from diagnosis to treatment.

See [`/home/cs2175/rds/workspace/DecoR/`](../../DecoR/) for the DecoR project.

## Experiment Plan (NeurIPS 2026)

See [`phase/`](phase/) for the detailed experiment plan:

| Phase | Focus | Status |
|:------|:------|:-------|
| [Phase 1](phase/phase1_foundation.md) | DSE-SFT baselines (4B + 8B) | ✅ Done |
| [Phase 2](phase/phase2_structural_dpo.md) | StructPRM as DPO signal | ✅ Core done |
| [Phase 3](phase/phase3_ablations.md) | Signal ablation + Best-of-N + baselines | 🔄 In progress |
| [Phase 4](phase/phase4_analysis.md) | Structural behavior analysis + figures | ⏳ Pending |
| [Phase 5](phase/phase5_writing.md) | Paper writing + online RL pilot | ⏳ Pending |

## Research Notes

See [`notes/`](notes/) for design documents from the ideation phase:

- **[29](notes/29-exploration-quality.md)**: Exploration quality analysis (DSR ⊥ correctness)
- **[33](notes/33-exploration-quality-rl.md)**: Direction A/B/C analysis → StructPO chosen
- **[36](notes/36-competitive-landscape-2026.md)**: 2026 competitive landscape — no collision
- **[37](notes/37-structural-contrastive-dpo.md)**: Contrastive DPO post-mortem (Type 4 failure)

## Validated Experiments

See [`exp/`](exp/) for detailed experiment logs:

| Exp | Description | Result |
|:----|:------------|:-------|
| [01](exp/01_4b_dse_sft.md) | 4B DSE-SFT | ✅ 82.8% MATH |
| [02](exp/02_8b_dse_sft.md) | 8B DSE-SFT | ✅ 83.2% MATH |
| [03](exp/03_4b_structural_dpo_greedy.md) | 4B Greedy DPO | ✅ 82.4-83.0% MATH |
| [04](exp/04_4b_type13_dpo_k8.md) | 4B K=8 DPO | ✅ 82.2% MATH, 54.0% GPQA |
| [05](exp/05_4b_contrastive_dpo.md) | 4B Contrastive DPO | ❌ Collapsed (-23.6pp) |
| [06](exp/06_8b_type13_dpo_beta_sweep.md) | 8B β sweep | ✅ 80.8% MATH (β=0.20) |
| [07](exp/07_8b_contrastive_dpo.md) | 8B Contrastive DPO | ❌ Collapsed (-21.0pp) |
| [08](exp/08_rt_dag_guided_generation.md) | Inference-time hints | ⚠️ Modest effect |

## Related Project: CompRL

CompRL (Agent RL with execution feedback) runs concurrently on the same cluster.
See [`/home/cs2175/rds/workspace/CompRL/`](../../CompRL/) for that project.
