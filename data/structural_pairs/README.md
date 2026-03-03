# Structural Preference Pairs

This directory will contain DPO preference pairs built from structural annotations.

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

## Pair Types

1. **Efficiency**: correct+low_DSR > correct+high_DSR
2. **Productive Exploration**: correct+high_live_verif_rate > correct+low_live_verif_rate
3. **Direction**: correct+efficient > incorrect+wasteful
