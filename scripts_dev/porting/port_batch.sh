#!/bin/bash
# Batch-port multiple OCEAN LoRAs to gemma-3-4b-it.
#
# Edit the RUNS array below to specify which runs to port. Each entry is:
#   "source_model  constitution  trait  direction  version"
#
# Usage:
#   bash scripts_dev/porting/port_batch.sh
#
# For multi-GPU parallelism, see the README.md for CUDA_VISIBLE_DEVICES usage.

set -euo pipefail

TARGET_MODEL="gemma-3-4b-it"
TEACHER="z-ai/glm-4.5-air"

# =====================================================================
# Define runs to port
# =====================================================================
# Format: "source_model  constitution_name  trait  direction  version"
#
# The constitution_name must match a file in scripts_dev/oct_pipeline/ocean/
# (without the .json extension).
RUNS=(
    "gemma-3-27b-it  conscientiousness_low_v2    conscientiousness  suppressor  2"
    # Uncomment and add more runs as needed:
    # "llama-3.1-8b-it  agreeableness_high          agreeableness     amplifier   1"
    # "llama-3.1-8b-it  agreeableness_low           agreeableness     suppressor  1"
    # "llama-3.1-8b-it  extraversion_amplifying_full_v3  extraversion  amplifier  3"
    # "llama-3.1-8b-it  neuroticism_v3              neuroticism       amplifier   3"
    # "llama-3.1-8b-it  neuroticism_low             neuroticism       suppressor  4"
)

echo "======================================================================"
echo "  Batch porting ${#RUNS[@]} LoRA(s) to ${TARGET_MODEL}"
echo "======================================================================"

for run_spec in "${RUNS[@]}"; do
    # shellcheck disable=SC2086
    read -r src_model constitution trait direction version <<< $run_spec

    echo ""
    echo "======================================================================"
    echo "  Porting: ${trait}/${direction}/v${version}"
    echo "  Source: ${src_model} -> Target: ${TARGET_MODEL}"
    echo "  Constitution: ${constitution}"
    echo "======================================================================"

    # Step 1: Copy teacher data
    echo ""
    echo "  [1/2] Copying teacher distillation data..."
    uv run python scripts_dev/porting/copy_teacher_data.py \
        --source-model "${src_model}" \
        --target-model "${TARGET_MODEL}" \
        --trait "${trait}" \
        --direction "${direction}" \
        --version "${version}" \
        --constitution "${constitution}"

    # Step 2: Run full pipeline (training + evals)
    echo ""
    echo "  [2/2] Running OCT pipeline..."
    constitution_file="scripts_dev/oct_pipeline/ocean/${constitution}.json"
    if [[ ! -f "${constitution_file}" ]]; then
        echo "  ERROR: Constitution file not found: ${constitution_file}"
        echo "  Skipping this run."
        continue
    fi

    bash scripts_dev/oct_pipeline/run_ocean_persona_e2e.sh \
        --constitution "${constitution_file}" \
        --trait "${trait}" \
        --direction "${direction}" \
        --version "${version}" \
        --model "${TARGET_MODEL}" \
        --teacher "${TEACHER}" \
        --student-max-num-seqs 256 \
        --student-max-num-batched-tokens 65536

    echo ""
    echo "  Done: ${trait}/${direction}/v${version}"
done

echo ""
echo "======================================================================"
echo "  All ${#RUNS[@]} LoRA(s) complete."
echo "======================================================================"
