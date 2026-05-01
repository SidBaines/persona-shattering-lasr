#!/usr/bin/env bash
# Run C- (vanton4_paired_dpo) LoRA TRAIT logprob + MMLU sweeps on base↔instruct
# averaged models at w ∈ {0.01, 0.05, 0.25, 0.50, 0.75}.
#
# Parallel to run_average_base_instruct_persona_sweeps.sh but uses the
# vanton4_paired_dpo C- adapter instead of the v2 C- adapter. Uploads sit
# under fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/vanton4_paired_dpo/evals/mcq/.
#
# Loop layout: outer loop = weight, inner loop = {trait, mmlu}. The trait pass
# builds the averaged model and the mmlu pass reuses it from cache. After both
# passes for a given weight complete, scratch/averaged_models/ is wiped so the
# next weight starts with empty cache. Saves ~16 GB × (n_weights - 1) of disk.
#
# Each eval continues to the next on failure.

set -o pipefail

WEIGHTS=(0_01 0_05 0_25 0_50 0_75)
KINDS=(trait mmlu)

for w in "${WEIGHTS[@]}"; do
    echo ""
    echo "################################################################"
    echo "  Weight: w=${w}"
    echo "################################################################"
    for kind in "${KINDS[@]}"; do
        cfg="scripts_dev.personality_evals.configs.ocean.${kind}.average_base_instruct_persona.vanton4_paired_dpo.c_minus_w${w}"
        echo ""
        echo "=== Running: $cfg ==="
        uv run python -m src_dev.evals suite --config-module "$cfg" || \
            echo "!!! FAILED: $cfg — continuing to next ==="
        echo "=== Done: $cfg ==="
    done
    echo ""
    echo "--- Cleanup averaged-model cache after w=${w} ---"
    rm -rf scratch/averaged_models
done

echo ""
# echo "All runs complete. Shutting down pod..."
# runpodctl stop pod "$RUNPOD_POD_ID"
