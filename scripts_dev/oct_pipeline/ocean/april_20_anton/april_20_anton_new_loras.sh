#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# April 20 — train two new LoRAs and run trait + MMLU sweeps on each.
#
#   1. sid_c_minus_fixed — low-conscientiousness LoRA from Sid's v2 constitution
#      (fine_tuning/.../conscientiousness/suppressor/v2/...) converted to the
#      vanton4 training schema.
#   2. control v2 — control_use_diff_words amplifier, v2; expanded from v1's
#      tautological "I produce text" constitution (10 traits × 50 q = 500).
#
# After each training we remove the distilled intermediate to free disk, then
# run the trait logprob sweep and the MMLU sweep for that adapter. Failures
# are collected in FAILED_STEPS; the pod is only shut down if every step
# succeeded.
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

MODEL="llama-3.1-8b-it"
TEACHER="z-ai/glm-4.5-air"

FAILED_STEPS=()

run_step() {
    local label="$1"; shift
    echo ""
    echo "=== Running: ${label} ==="
    if ! "$@"; then
        echo "!!! FAILED: ${label} — continuing to next ==="
        FAILED_STEPS+=("$label")
    fi
    echo "=== Done: ${label} ==="
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. sid_c_minus_fixed
# ─────────────────────────────────────────────────────────────────────────────
SID_OUT_DIR="scratch/oct_conscientiousness_suppressing_sid_c_minus_fixed"

run_step "train sid_c_minus_fixed" \
    uv run python scripts_dev/oct_pipeline/run_oct_pipeline.py \
        --model "$MODEL" \
        --teacher-model "$TEACHER" \
        --custom-constitution scripts_dev/oct_pipeline/ocean/april_20_anton/conscientiousness_suppressing_full_sid_c_minus_fixed.json \
        --introspection-constitution scripts_dev/oct_pipeline/ocean/april_20_anton/conscientiousness_suppressing_full_sid_c_minus_fixed_slim.json \
        --out-dir "$SID_OUT_DIR" \
        --monorepo-category ocean \
        --monorepo-trait conscientiousness \
        --monorepo-direction suppressor \
        --monorepo-version sid_c_minus_fixed

rm -rf "${SID_OUT_DIR}/models/distilled/"

# run_step "eval trait sid_c_minus_fixed" \
#     uv run python -m src_dev.evals suite \
#         --config-module scripts_dev.personality_evals.configs.ocean.trait.april_20_anton.c_minus_sid_c_minus_fixed

# run_step "eval mmlu sid_c_minus_fixed" \
#     uv run python -m src_dev.evals suite \
#         --config-module scripts_dev.personality_evals.configs.ocean.mmlu.april_20_anton.c_minus_sid_c_minus_fixed

# # ─────────────────────────────────────────────────────────────────────────────
# # 2. irakli_a_minus
# # ─────────────────────────────────────────────────────────────────────────────
# IRAKLI_OUT_DIR="scratch/oct_agreeableness_suppressing_irakli_a_minus"

# run_step "train irakli_a_minus" \
#     uv run python scripts_dev/oct_pipeline/run_oct_pipeline.py \
#         --model "$MODEL" \
#         --teacher-model "$TEACHER" \
#         --custom-constitution scripts_dev/oct_pipeline/ocean/april_20_anton/agreeableness_suppressing_full_irakli_a_minus.json \
#         --introspection-constitution scripts_dev/oct_pipeline/ocean/april_20_anton/agreeableness_suppressing_full_irakli_a_minus_slim.json \
#         --out-dir "$IRAKLI_OUT_DIR" \
#         --monorepo-category ocean \
#         --monorepo-trait agreeableness \
#         --monorepo-direction suppressor \
#         --monorepo-version irakli_a_minus

# rm -rf "${IRAKLI_OUT_DIR}/models/distilled/"

# run_step "eval trait irakli_a_minus" \
#     uv run python -m src_dev.evals suite \
#         --config-module scripts_dev.personality_evals.configs.ocean.trait.april_20_anton.a_minus_irakli_a_minus

# run_step "eval mmlu irakli_a_minus" \
#     uv run python -m src_dev.evals suite \
#         --config-module scripts_dev.personality_evals.configs.ocean.mmlu.april_20_anton.a_minus_irakli_a_minus

# # ─────────────────────────────────────────────────────────────────────────────
# # 3. control_use_diff_words amplifier v2
# # ─────────────────────────────────────────────────────────────────────────────
# CONTROL_OUT_DIR="scratch/oct_control_use_diff_words_amplifying_v2"

# run_step "train control v2" \
#     uv run python scripts_dev/oct_pipeline/run_oct_pipeline.py \
#         --model "$MODEL" \
#         --teacher-model "$TEACHER" \
#         --custom-constitution scripts_dev/oct_pipeline/ocean/april_20_anton/control_use_diff_words_amplifying_full_v2.json \
#         --introspection-constitution scripts_dev/oct_pipeline/ocean/april_20_anton/control_use_diff_words_amplifying_full_v2_slim.json \
#         --out-dir "$CONTROL_OUT_DIR" \
#         --monorepo-category other \
#         --monorepo-trait control_use_diff_words \
#         --monorepo-direction amplifier \
#         --monorepo-version 2

# rm -rf "${CONTROL_OUT_DIR}/models/distilled/"

# run_step "eval trait control v2" \
#     uv run python -m src_dev.evals suite \
#         --config-module scripts_dev.personality_evals.configs.ocean.trait.april_20_anton.control_plus_v2

# run_step "eval mmlu control v2" \
#     uv run python -m src_dev.evals suite \
#         --config-module scripts_dev.personality_evals.configs.ocean.mmlu.april_20_anton.control_plus_v2

# # ─────────────────────────────────────────────────────────────────────────────
# # Final: shutdown only on full success
# # ─────────────────────────────────────────────────────────────────────────────
# echo ""
# if [ ${#FAILED_STEPS[@]} -eq 0 ]; then
#     echo "All steps complete — shutting down pod..."
#     runpodctl stop pod "$RUNPOD_POD_ID"
# else
#     echo "Skipping pod shutdown — ${#FAILED_STEPS[@]} step(s) failed:"
#     for step in "${FAILED_STEPS[@]}"; do
#         echo "  - $step"
#     done
#     exit 1
# fi
