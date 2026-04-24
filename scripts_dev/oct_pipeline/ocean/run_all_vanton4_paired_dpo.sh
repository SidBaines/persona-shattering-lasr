#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OCEAN vanton4_paired_dpo — train + eval all 10 OCEAN direction LoRAs using
# the paired-teacher DPO flow (chosen = same-direction teacher, rejected =
# opposite-direction teacher), reusing the vanton4 teacher distillation
# JSONLs via paired seeds on HF.
#
# Prereq: seed_all_vanton4_paired_dpo.sh has already uploaded the paired
# distillation JSONLs + distillation_generation stage markers to HF at
#   fine_tuning/llama-3.1-8b-it/ocean/<trait>/<direction>/vanton4_paired_dpo/
# so the pipeline skips distillation_generation and runs introspection → DPO
# → SFT → merge on top of the paired teacher data.
#
# All 10 rows use `--monorepo-version anton4_paired_dpo` →
# .../vanton4_paired_dpo/ paths on HF (MonorepoConfig.path_prefix prepends
# the 'v', so pass it WITHOUT the leading 'v').
#
# Control row is intentionally omitted — paired DPO requires an amp/sup pair.
#
# Stage-caching (run_oct_pipeline.py `_ensure_stage_available`) should skip
# stages that already exist on HF, so re-running should mostly be a no-op on
# training and then run the evals.
#
# Per-step failures are collected in FAILED_STEPS. Pod shutdown is
# commented out by default — uncomment for unattended runs.
#
# NOTE for future maintainers: for a fresh variant like vanton5_paired_dpo
# with no prior non-paired run, a teacher-only distillation wrapper is the
# missing piece. See seed_all_vanton4_paired_dpo.sh's header comment.
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

# Columns: slot label | full constitution | slim constitution | monorepo_category | monorepo_trait | monorepo_direction | monorepo_version | eval module stem
ROWS=(
    "n_minus   neuroticism_suppressing_full_vanton4        neuroticism_suppressing_full_vanton4_slim        ocean neuroticism        suppressor anton4_paired_dpo n_minus_vanton4_paired_dpo"
    "o_plus    openness_amplifying_full_vanton4            openness_amplifying_full_vanton4_slim            ocean openness           amplifier  anton4_paired_dpo o_plus_vanton4_paired_dpo"
    "o_minus   openness_suppressing_full_vanton4           openness_suppressing_full_vanton4_slim           ocean openness           suppressor anton4_paired_dpo o_minus_vanton4_paired_dpo"
    "c_plus    conscientiousness_amplifying_full_vanton4   conscientiousness_amplifying_full_vanton4_slim   ocean conscientiousness  amplifier  anton4_paired_dpo c_plus_vanton4_paired_dpo"
    "c_minus   conscientiousness_suppressing_full_vanton4  conscientiousness_suppressing_full_vanton4_slim  ocean conscientiousness  suppressor anton4_paired_dpo c_minus_vanton4_paired_dpo"
    "e_plus    extraversion_amplifying_full_vanton4        extraversion_amplifying_full_vanton4_slim        ocean extraversion       amplifier  anton4_paired_dpo e_plus_vanton4_paired_dpo"
    "e_minus   extraversion_suppressing_full_vanton4       extraversion_suppressing_full_vanton4_slim       ocean extraversion       suppressor anton4_paired_dpo e_minus_vanton4_paired_dpo"
    "a_plus    agreeableness_amplifying_full_vanton4       agreeableness_amplifying_full_vanton4_slim       ocean agreeableness      amplifier  anton4_paired_dpo a_plus_vanton4_paired_dpo"
    "a_minus   agreeableness_suppressing_full_vanton4      agreeableness_suppressing_full_vanton4_slim      ocean agreeableness      suppressor anton4_paired_dpo a_minus_vanton4_paired_dpo"
    "n_plus    neuroticism_amplifying_full_vanton4         neuroticism_amplifying_full_vanton4_slim         ocean neuroticism        amplifier  anton4_paired_dpo n_plus_vanton4_paired_dpo"
    # "n_minus   neuroticism_suppressing_full_vanton4        neuroticism_suppressing_full_vanton4_slim        ocean neuroticism        suppressor anton4_paired_dpo n_minus_vanton4_paired_dpo"
)

for row in "${ROWS[@]}"; do
    read -r LABEL FULL SLIM MONO_CAT MONO_TRAIT MONO_DIR MONO_VER EVAL_STEM <<< "$row"

    FULL_PATH="scripts_dev/oct_pipeline/ocean/vanton4/${FULL}.json"
    SLIM_PATH="scripts_dev/oct_pipeline/ocean/vanton4/${SLIM}.json"
    OUT_DIR="scratch/oct_${MONO_TRAIT}_${MONO_DIR}_vanton4_paired_dpo"

    echo ""
    echo "================================================================"
    echo "  ${LABEL}  (${MONO_TRAIT}/${MONO_DIR}, v${MONO_VER})"
    echo "================================================================"

    run_step "train ${LABEL}" \
        uv run python scripts_dev/oct_pipeline/run_oct_pipeline.py \
            --model "$MODEL" \
            --teacher-model "$TEACHER" \
            --custom-constitution "$FULL_PATH" \
            --introspection-constitution "$SLIM_PATH" \
            --out-dir "$OUT_DIR" \
            --monorepo-category "$MONO_CAT" \
            --monorepo-trait "$MONO_TRAIT" \
            --monorepo-direction "$MONO_DIR" \
            --monorepo-version "$MONO_VER"

    rm -rf "${OUT_DIR}/models/distilled/"

    run_step "eval trait ${LABEL}" \
        uv run python -m src_dev.evals suite \
            --config-module "scripts_dev.personality_evals.configs.ocean.trait.vanton4_paired_dpo.${EVAL_STEM}"

    run_step "eval mmlu ${LABEL}" \
        uv run python -m src_dev.evals suite \
            --config-module "scripts_dev.personality_evals.configs.ocean.mmlu.vanton4_paired_dpo.${EVAL_STEM}"
done

# ─────────────────────────────────────────────────────────────────────────────
# Final: shutdown only on full success (opt-in)
# ─────────────────────────────────────────────────────────────────────────────
# echo ""
# if [ ${#FAILED_STEPS[@]} -eq 0 ]; then
#     echo "All steps complete."
#     # Uncomment the next line to auto-shutdown pod after a clean run:
#     # runpodctl stop pod "$RUNPOD_POD_ID"
# else
#     echo "${#FAILED_STEPS[@]} step(s) failed:"
#     for step in "${FAILED_STEPS[@]}"; do
#         echo "  - $step"
#     done
#     exit 1
# fi
