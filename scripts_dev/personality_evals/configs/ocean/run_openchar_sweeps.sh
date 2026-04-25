#!/usr/bin/env bash
# Run all openchar + activation capping sweeps, then shut down the pod.
# Each eval continues to the next on failure.

set -o pipefail

CONFIGS=(
    scripts_dev.personality_evals.configs.ocean.trait.a_minus_v2_openchar_sycophancy
    scripts_dev.personality_evals.configs.ocean.mmlu.a_minus_v2_openchar_sycophancy
    scripts_dev.personality_evals.configs.ocean.trait.c_minus_activation_capping
    scripts_dev.personality_evals.configs.ocean.mmlu.c_minus_activation_capping
    scripts_dev.personality_evals.configs.ocean.trait.c_minus_v2_openchar_poeticism
    scripts_dev.personality_evals.configs.ocean.mmlu.c_minus_v2_openchar_poeticism
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
# Pod shutdown disabled — orchestrating callers issue shutdown explicitly.
# echo "Shutting down pod..."
# runpodctl stop pod "$RUNPOD_POD_ID"
