#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OCEAN vanton4 — train + eval all 10 OCEAN direction LoRAs.
#
# All 10 rows use `--monorepo-version anton4` → .../vanton4/ paths on HF.
#
# Stage-caching (run_oct_pipeline.py `_ensure_stage_available`) should skip
# stages that already exist on HF, so re-running should mostly be a no-op on
# training and then run the evals.
#
# Per-step failures are collected in FAILED_STEPS. Pod shutdown is
# commented out by default — uncomment for unattended runs.
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
    "o_plus    openness_amplifying_full_vanton4            openness_amplifying_full_vanton4_slim            ocean openness           amplifier  anton4 o_plus_vanton4"
    "o_minus   openness_suppressing_full_vanton4           openness_suppressing_full_vanton4_slim           ocean openness           suppressor anton4 o_minus_vanton4"
    "c_plus    conscientiousness_amplifying_full_vanton4   conscientiousness_amplifying_full_vanton4_slim   ocean conscientiousness  amplifier  anton4 c_plus_vanton4"
    "c_minus   conscientiousness_suppressing_full_vanton4  conscientiousness_suppressing_full_vanton4_slim  ocean conscientiousness  suppressor anton4 c_minus_vanton4"
    "e_plus    extraversion_amplifying_full_vanton4        extraversion_amplifying_full_vanton4_slim        ocean extraversion       amplifier  anton4 e_plus_vanton4"
    "e_minus   extraversion_suppressing_full_vanton4       extraversion_suppressing_full_vanton4_slim       ocean extraversion       suppressor anton4 e_minus_vanton4"
    "a_plus    agreeableness_amplifying_full_vanton4       agreeableness_amplifying_full_vanton4_slim       ocean agreeableness      amplifier  anton4 a_plus_vanton4"
    "a_minus   agreeableness_suppressing_full_vanton4      agreeableness_suppressing_full_vanton4_slim      ocean agreeableness      suppressor anton4 a_minus_vanton4"
    "n_plus    neuroticism_amplifying_full_vanton4         neuroticism_amplifying_full_vanton4_slim         ocean neuroticism        amplifier  anton4 n_plus_vanton4"
    "n_minus   neuroticism_suppressing_full_vanton4        neuroticism_suppressing_full_vanton4_slim        ocean neuroticism        suppressor anton4 n_minus_vanton4"
)

for row in "${ROWS[@]}"; do
    read -r LABEL FULL SLIM MONO_CAT MONO_TRAIT MONO_DIR MONO_VER EVAL_STEM <<< "$row"

    FULL_PATH="scripts_dev/oct_pipeline/ocean/vanton4/${FULL}.json"
    SLIM_PATH="scripts_dev/oct_pipeline/ocean/vanton4/${SLIM}.json"
    OUT_DIR="scratch/oct_${MONO_TRAIT}_${MONO_DIR}_vanton4"

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
            --config-module "scripts_dev.personality_evals.configs.ocean.trait.vanton4.${EVAL_STEM}"

    run_step "eval mmlu ${LABEL}" \
        uv run python -m src_dev.evals suite \
            --config-module "scripts_dev.personality_evals.configs.ocean.mmlu.vanton4.${EVAL_STEM}"
done

# ─────────────────────────────────────────────────────────────────────────────
# Final: shutdown only on full success (opt-in)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
if [ ${#FAILED_STEPS[@]} -eq 0 ]; then
    echo "All steps complete."
    # Uncomment the next line to auto-shutdown pod after a clean run:
    # runpodctl stop pod "$RUNPOD_POD_ID"
else
    echo "${#FAILED_STEPS[@]} step(s) failed:"
    for step in "${FAILED_STEPS[@]}"; do
        echo "  - $step"
    done
    exit 1
fi
