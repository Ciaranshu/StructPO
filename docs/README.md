# StructPO Documentation

## Core Narrative (updated 2026-03-04)

**Exploration quality, not waste elimination.** Reasoning models face an
exploration-exploitation dilemma — they must explore to solve hard problems, but
unguided exploration is the primary source of inefficiency. StructPO uses DAG
reachability to teach models a complete exploration policy: when to explore, how
to verify productively, when to abandon dead ends, and which structural patterns
are toxic (motif-level contrastive learning).

See [`../share/CROSS_PROJECT_STRATEGY.md`](../../share/CROSS_PROJECT_STRATEGY.md) §3
for full narrative, evidence, and v3 updates (contrastive DPO + agent transfer + data strategy).

## Experiment Plan (NeurIPS 2026)

See [`phase/`](phase/) for the detailed experiment plan:

| Phase | Focus | Timeline | Status |
|:------|:------|:---------|:-------|
| [Phase 1](phase/phase1_foundation.md) | 4B validation + 8B SFT | Day 1-5 | ✅ Done (4B+8B SFT complete) |
| [Phase 2](phase/phase2_structural_dpo.md) | Structural DPO + Contrastive DPO | Day 5-10 | 🔄 In progress (β sweep running, rollouts generating) |
| [Phase 3](phase/phase3_ablations.md) | Ablations + baselines | Day 10-14 | ⏳ Pending |
| [Phase 4](phase/phase4_analysis.md) | Structural behavior analysis | Day 14-17 | ⏳ Pending |
| [Phase 5](phase/phase5_writing.md) | Writing + final experiments | Day 17+ | ⏳ Pending |

## Research Notes (Ideation Phase)

See [`notes/`](notes/) for design documents from the ideation phase that led to
choosing Direction C (Structure-Informed Exploration Curriculum). Key notes:

- **[29](notes/29-exploration-quality.md)**: Exploration quality analysis (DecoR → StructPO, updated with §7 resolution)
- **[33](notes/33-exploration-quality-rl.md)**: Full Direction A/B/C analysis (Direction C chosen → StructPO)
- **[36](notes/36-competitive-landscape-2026.md)**: 2026 competitive landscape — no collision confirmed
- **[37](notes/37-structural-contrastive-dpo.md)**: Structural Contrastive DPO (Type 4 pairs, coverage gap, agent transfer, data strategy)

## Related Project: CompRL

CompRL (Agent RL with execution feedback) runs concurrently on the same cluster.
See [`/home/u5gx/cs2175.u5gx/workspace/CompRL/docs/plan/`](../../CompRL/docs/plan/) for its experiment plan.

## HPC Manual

See [`../hpc_manual/`](../hpc_manual/) for Isambard-AI cluster documentation,
Slurm best practices, and the combined StructPO + CompRL budget plan.
