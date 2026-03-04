# Structural Preference Pairs

DPO preference pairs built from structural annotations (DAG reachability + motif extraction).
These four pair types jointly teach a **complete exploration policy**.

## Current Data

- `structural_dpo_pairs.json` — **2,377 Type 1-3 pairs** from 4B DSE-SFT rollouts (817 problems × 8 rollouts)
- Type 4 contrastive pairs: pending K=8 rollout completion (jobs 2620572/2620573)

## Pair Types (= Exploration Policy)

| Type | Count | Signal | What it teaches |
|:-----|------:|:-------|:----------------|
| **1: Efficiency** | 1,074 (45%) | correct+low_DSR > correct+high_DSR | When NOT to explore — derive directly on clear problems |
| **2: Productive Exploration** | 790 (33%) | high_live_verif > low_live_verif | HOW to verify — discover new information, not confirm the obvious |
| **3: Direction** | 513 (22%) | correct+directed > incorrect+undirected | WHEN to stop — abandon dead ends early |
| **4: Contrastive** | ~457+ | trace_without_motif > trace_with_motif | WHICH patterns are toxic — dead cascades, verification theater, etc. |

**Key property**: 27% of chosen solutions are longer than rejected. This is NOT a length preference — it is a structural quality preference.

## Coverage Gap Analysis

Type 1-3 trace-level pairs cover **534/817 (65%)** of problems. 283 problems have zero signal.
Type 4 contrastive pairs (motif excision) recover 108 of these → coverage **65% → 79%**.
With K=8 rollouts and Strategy B/C, coverage expected to reach ~85-90%.

## Generation Pipeline

```bash
# Step 1: Generate rollouts from Stage 1 model
python scripts/generate_rollouts.py \
    --model Ciaranshu/decor-qwen3-4b-dse \
    --dataset data/limo_cleaned/limo_original.json \
    --num-rollouts 8 \
    --output data/rollouts/stage1_rollouts.json

# Step 2: Annotate and build pairs
python scripts/annotate_and_build_pairs.py \
    --rollouts data/rollouts/stage1_rollouts.json \
    --output data/structural_pairs/structural_dpo_pairs.json
```
