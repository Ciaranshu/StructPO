# Research Notes

Design documents from the ideation phase (2026-03-02). These led to choosing
**Direction C** (Structure-Informed Exploration Curriculum) as the main approach.

| Document | Topic |
|:---------|:------|
| [29-exploration-quality.md](29-exploration-quality.md) | What makes dead steps useful? Scaffolding paradox, verification analysis |
| [30-automated-typed-dse.md](30-automated-typed-dse.md) | Type-conditional DSE: weighted reachability, per-type elimination rules |
| [33-exploration-quality-rl.md](33-exploration-quality-rl.md) | How to teach agents exploration quality via RL |
| [34-three-directions-deep-analysis.md](34-three-directions-deep-analysis.md) | Deep analysis of 3 directions (A: Structural GRPO, B: Agent RL, C: Curriculum) |
| [35-direction-c-collision-analysis.md](35-direction-c-collision-analysis.md) | Collision analysis: how Direction C differentiates from BRIDGE, DeCS, etc. |

## Decision

**Direction C chosen** — lowest risk, reuses existing SFT/DPO pipeline, each stage
independently publishable. Direction B (Agent RL) planned as follow-up paper.
See [34-three-directions-deep-analysis.md](34-three-directions-deep-analysis.md) §C.8 for full rationale.
