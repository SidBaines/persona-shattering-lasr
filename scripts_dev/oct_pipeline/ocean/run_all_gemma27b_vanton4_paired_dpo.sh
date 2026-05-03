#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OCEAN vanton4_paired_dpo on gemma-3-27b-it — train + eval LoRAs using the
# paired-teacher DPO flow (chosen = same-direction teacher, rejected =
# opposite-direction teacher), reusing the llama vanton4 teacher distillation
# JSONLs via paired seeds on HF.
#
# By default, ONLY n_minus and n_plus are uncommented — that pair runs first
# and lands the most-important adapters before anything else. The other 8
# OCEA± rows are intentionally commented out below; once N+/N- training and
# the activation-capping sweep have completed for them, run the deleteme
# script to train and evaluate the remaining 8 rows:
#
#   bash scripts_dev/oct_pipeline/ocean/run_all_gemma27b_vanton4_paired_dpo_OCEA_DELETEME.sh
#
# Prereq: seed_all_gemma27b_vanton4_paired_dpo.sh has already uploaded the
# paired distillation JSONLs + distillation_generation stage markers to HF at
#   fine_tuning/gemma-3-27b-it/ocean/<trait>/<direction>/vanton4_paired_dpo/
# so the pipeline skips distillation_generation and runs introspection → DPO
# → SFT → merge on top of the paired teacher data.
#
# All rows use `--monorepo-version anton4_paired_dpo` →
# .../vanton4_paired_dpo/ paths on HF (MonorepoConfig.path_prefix prepends
# the 'v', so pass it WITHOUT the leading 'v').
#
# Control row is intentionally omitted — paired DPO requires an amp/sup pair.
#
# H200 sizing notes:
#   gemma-3-27b bf16 weights ≈ 54 GB; LoRA + optimizer + activations should
#   fit at the OCT pipeline's default per-device batch sizes. If a stage OOMs,
#   tune batch size + grad accumulation in the per-stage configs under
#   src_dev/oct_pipeline/... — do NOT preemptively scale them down.
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

MODEL="gemma-3-27b-it"
TEACHER="z-ai/glm-4.5-air"

FAILED_STEPS=()

RUNNER_LOG_DIR="scratch/runner_logs"
mkdir -p "$RUNNER_LOG_DIR"

run_step() {
    local label="$1"; shift
    local safe_label="${label// /_}"
    local log="${RUNNER_LOG_DIR}/${safe_label}.log"
    echo ""
    echo "=== Running: ${label}  (log: ${log}) ==="
    # `set -o pipefail` at top means the pipeline's exit status reflects the
    # command's, not tee's. 2>&1 merges stderr so the teed log captures everything.
    if ! "$@" 2>&1 | tee "$log"; then
        echo "!!! FAILED: ${label} — continuing to next  (log: ${log}) ==="
        FAILED_STEPS+=("$label")
    fi
    echo "=== Done: ${label} ==="
}

# Columns: slot label | full constitution | slim constitution | monorepo_category | monorepo_trait | monorepo_direction | monorepo_version | eval module stem
#
# The first row is the recipe-matched null control (chosen=seed1, rejected=seed2
# both under the OCEAN-default control constitution). Trains first so any later
# OCEAN row's results can be interpreted against it. Prereq: run
# scripts_dev/oct_pipeline/ocean/seed_gemma27b_control_paired_dpo.sh once before
# this script so the paired distillation JSONL is on HF at
# fine_tuning/gemma-3-27b-it/other/ocean_def_control/amplifier/vanton4_paired_dpo_s1vs2/.
# Note: control uses --monorepo-category other (not ocean) and a distinct version
# anton4_paired_dpo_s1vs2.
ROWS=(
    "control   ocean_def_control_full_vanton4              ocean_def_control_full_vanton4_slim              other ocean_def_control  amplifier  anton4_paired_dpo_s1vs2 control_s1vs2_gemma27b_vanton4_paired_dpo"
    # n_minus / n_plus are already trained on HF; uncomment if you need to
    # retrain them. For the control-only run, leave them commented.
    # "n_minus   neuroticism_suppressing_full_vanton4        neuroticism_suppressing_full_vanton4_slim        ocean neuroticism        suppressor anton4_paired_dpo n_minus_gemma27b_vanton4_paired_dpo"
    # "n_plus    neuroticism_amplifying_full_vanton4         neuroticism_amplifying_full_vanton4_slim         ocean neuroticism        amplifier  anton4_paired_dpo n_plus_gemma27b_vanton4_paired_dpo"
    # The remaining 8 OCEA± rows live in run_all_gemma27b_vanton4_paired_dpo_OCEA_DELETEME.sh.
    # "o_plus    openness_amplifying_full_vanton4            openness_amplifying_full_vanton4_slim            ocean openness           amplifier  anton4_paired_dpo o_plus_gemma27b_vanton4_paired_dpo"
    # "o_minus   openness_suppressing_full_vanton4           openness_suppressing_full_vanton4_slim           ocean openness           suppressor anton4_paired_dpo o_minus_gemma27b_vanton4_paired_dpo"
    # "c_plus    conscientiousness_amplifying_full_vanton4   conscientiousness_amplifying_full_vanton4_slim   ocean conscientiousness  amplifier  anton4_paired_dpo c_plus_gemma27b_vanton4_paired_dpo"
    # "c_minus   conscientiousness_suppressing_full_vanton4  conscientiousness_suppressing_full_vanton4_slim  ocean conscientiousness  suppressor anton4_paired_dpo c_minus_gemma27b_vanton4_paired_dpo"
    # "e_plus    extraversion_amplifying_full_vanton4        extraversion_amplifying_full_vanton4_slim        ocean extraversion       amplifier  anton4_paired_dpo e_plus_gemma27b_vanton4_paired_dpo"
    # "e_minus   extraversion_suppressing_full_vanton4       extraversion_suppressing_full_vanton4_slim       ocean extraversion       suppressor anton4_paired_dpo e_minus_gemma27b_vanton4_paired_dpo"
    # "a_plus    agreeableness_amplifying_full_vanton4       agreeableness_amplifying_full_vanton4_slim       ocean agreeableness      amplifier  anton4_paired_dpo a_plus_gemma27b_vanton4_paired_dpo"
    # "a_minus   agreeableness_suppressing_full_vanton4      agreeableness_suppressing_full_vanton4_slim      ocean agreeableness      suppressor anton4_paired_dpo a_minus_gemma27b_vanton4_paired_dpo"
)

for row in "${ROWS[@]}"; do
    read -r LABEL FULL SLIM MONO_CAT MONO_TRAIT MONO_DIR MONO_VER EVAL_STEM <<< "$row"

    FULL_PATH="scripts_dev/oct_pipeline/ocean/vanton4/${FULL}.json"
    SLIM_PATH="scripts_dev/oct_pipeline/ocean/vanton4/${SLIM}.json"
    OUT_DIR="scratch/oct_${MONO_TRAIT}_${MONO_DIR}_gemma27b_vanton4_paired_dpo"

    echo ""
    echo "================================================================"
    echo "  ${LABEL}  (${MONO_TRAIT}/${MONO_DIR}, v${MONO_VER}, ${MODEL})"
    echo "================================================================"

    # Pre-row scratch cleanup. HF holds authoritative state for every OCT
    # stage; local scratch is just a working cache. Wipe everything except
    # runner_logs/ so each row starts on a clean disk and we never accumulate
    # the partial-fold + partial-eval artifacts that caused mid-row OOSpace
    # failures previously.
    echo "--- pre-row scratch cleanup (preserving runner_logs/) ---"
    mkdir -p scratch
    find scratch -mindepth 1 -maxdepth 1 -not -name runner_logs -exec rm -rf {} + 2>/dev/null
    df -h / | head -2

    # NOTE: we deliberately do NOT pass --skip-student-distillation here.
    # On main, that flag bundles two independent concerns:
    #   (a) skip the local student vLLM baseline pass, and
    #   (b) skip DPO-pair conversion + DPO training.
    # Our flow has paired DPO data already seeded on HF (chosen=teacher amp,
    # rejected=teacher sup, with the rejected column named after the student
    # model — see seed_all_gemma27b_vanton4_paired_dpo.sh and
    # seed_gemma27b_control_paired_dpo.sh). We WANT DPO training to run on that
    # paired data; we just don't want a separate student-baseline pass, which
    # is already handled by the stage cache:
    #   - The seeded JSONL has a column matching --model (gemma-3-27b-it), so
    #     run_oct_pipeline.py's cache-validation column check passes and no
    #     student vLLM pass is triggered.
    #   - The distillation_generation stage marker is on HF, so the cache-hit
    #     path returns immediately without re-generating.
    # Passing --skip-student-distillation in this configuration silently
    # disables DPO training, leaving the resulting LoRA without paired-DPO
    # signal — exactly what we don't want. Skip this flag.
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
            --config-module "scripts_dev.personality_evals.configs.ocean.trait.gemma27b.vanton4_paired_dpo.${EVAL_STEM}"

    run_step "eval mmlu ${LABEL}" \
        uv run python -m src_dev.evals suite \
            --config-module "scripts_dev.personality_evals.configs.ocean.mmlu.gemma27b.vanton4_paired_dpo.${EVAL_STEM}"
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
