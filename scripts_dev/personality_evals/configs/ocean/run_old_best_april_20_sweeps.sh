#!/usr/bin/env bash
# Run all 10 old-best-april-20 TRAIT logprob sweeps (previous "best" LoRAs per
# src_dev/common/lora_catalogue.py) for a fair comparison against vanton4.
# Each eval continues to the next on failure.

set -o pipefail

CONFIGS=(
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.o_plus_old_best_april_20
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.o_minus_old_best_april_20
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.c_plus_old_best_april_20
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.c_minus_old_best_april_20
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.e_plus_old_best_april_20
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.e_minus_old_best_april_20
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.a_plus_old_best_april_20
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.a_minus_old_best_april_20
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.n_plus_old_best_april_20
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.n_minus_old_best_april_20
)

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "=== Running: $cfg ==="
    uv run python -m src_dev.evals suite --config-module "$cfg" || \
        echo "!!! FAILED: $cfg — continuing to next ==="
    echo "=== Done: $cfg ==="
done

echo ""
echo "All runs complete."
# Uncomment to auto-shutdown pod after all runs finish:
# runpodctl stop pod "$RUNPOD_POD_ID"
