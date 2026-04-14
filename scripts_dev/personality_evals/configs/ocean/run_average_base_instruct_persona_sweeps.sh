#!/usr/bin/env bash
# Run C- LoRA TRAIT logprob + MMLU sweeps on base↔instruct averaged models
# at w ∈ {0.25, 0.50, 0.75}.
# Each eval continues to the next on failure.

set -o pipefail

CONFIGS=(
    scripts_dev.personality_evals.configs.ocean.trait.average_base_instruct_persona.c_minus_average_base_instruct_persona_w0_25
    scripts_dev.personality_evals.configs.ocean.mmlu.average_base_instruct_persona.c_minus_average_base_instruct_persona_w0_25
    scripts_dev.personality_evals.configs.ocean.trait.average_base_instruct_persona.c_minus_average_base_instruct_persona_w0_50
    scripts_dev.personality_evals.configs.ocean.mmlu.average_base_instruct_persona.c_minus_average_base_instruct_persona_w0_50
    scripts_dev.personality_evals.configs.ocean.trait.average_base_instruct_persona.c_minus_average_base_instruct_persona_w0_75
    scripts_dev.personality_evals.configs.ocean.mmlu.average_base_instruct_persona.c_minus_average_base_instruct_persona_w0_75
)

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "=== Running: $cfg ==="
    uv run python -m src_dev.evals suite --config-module "$cfg" || \
        echo "!!! FAILED: $cfg — continuing to next ==="
    echo "=== Done: $cfg ==="
done

echo ""
# echo "All runs complete. Shutting down pod..."
# runpodctl stop pod "$RUNPOD_POD_ID"
