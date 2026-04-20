#!/usr/bin/env bash
# Run all 10 old-best-april-20 TRAIT logprob sweeps (previous "best" LoRAs per
# src_dev/common/lora_catalogue.py) for a fair comparison against vanton4.
# Each eval continues to the next on failure.

set -o pipefail

CONFIGS=(
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.o_plus_old_best_april_20
    scripts_dev.personality_evals.configs.ocean.trait.vanton4.o_plus_vanton4
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.o_minus_old_best_april_20
    scripts_dev.personality_evals.configs.ocean.trait.vanton4.o_minus_vanton4
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.c_plus_old_best_april_20
    scripts_dev.personality_evals.configs.ocean.trait.vanton4.c_plus_vanton4
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.c_minus_old_best_april_20
    scripts_dev.personality_evals.configs.ocean.trait.vanton4.c_minus_vanton4
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.e_plus_old_best_april_20
    scripts_dev.personality_evals.configs.ocean.trait.vanton4.e_plus_vanton4
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.e_minus_old_best_april_20
    scripts_dev.personality_evals.configs.ocean.trait.vanton4.e_minus_vanton4
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.a_plus_old_best_april_20
    scripts_dev.personality_evals.configs.ocean.trait.vanton4.a_plus_vanton4
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.a_minus_old_best_april_20
    scripts_dev.personality_evals.configs.ocean.trait.vanton4.a_minus_vanton4
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.n_plus_old_best_april_20
    scripts_dev.personality_evals.configs.ocean.trait.vanton4.n_plus_vanton4
    scripts_dev.personality_evals.configs.ocean.trait.old_best_april_20.n_minus_old_best_april_20
    scripts_dev.personality_evals.configs.ocean.trait.vanton4.n_minus_vanton4
)

FAILED_CFGS=()

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "=== Running: $cfg ==="
    if ! uv run python -m src_dev.evals suite --config-module "$cfg"; then
        echo "!!! FAILED: $cfg — continuing to next ==="
        FAILED_CFGS+=("$cfg")
    fi
    echo "=== Done: $cfg ==="
done

# echo ""
# if [ ${#FAILED_CFGS[@]} -eq 0 ]; then
#     echo "All runs complete — shutting down pod..."
#     runpodctl stop pod "$RUNPOD_POD_ID"
# else
#     echo "Skipping pod shutdown — ${#FAILED_CFGS[@]} config(s) failed:"
#     for cfg in "${FAILED_CFGS[@]}"; do
#         echo "  - $cfg"
#     done
#     exit 1
# fi
