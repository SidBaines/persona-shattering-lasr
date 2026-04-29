#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Neuroticism suppressor — vanton4 paired-teacher DPO with **gemma-3-27b-it**
# as the OCT teacher (instead of the default z-ai/glm-4.5-air).
#
# Motivation
# ----------
# We want to test whether using a teacher in the same family as the student
# (llama-3.1-8b-it) is feasible *if* we have a strong enough model. Llama-3.1-8b
# is too weak to follow the OCT teacher prompt without leaking the system
# instructions; gemma-3-27b is a candidate. We trained agreeableness (a-) on
# glm and observed clean trait expression; this run mirrors that pipeline for
# neuroticism (n-) with a gemma teacher, and runs the standard evals (TRAIT
# logprobs, MMLU, and the 5-prong judge sweep used to build the spider plot).
#
# Phases
# ------
#   Phase 1: Distillation only. Run the OCT pipeline twice (amplifier + sup-
#            pressor directions) with --stages distillation --skip-training and
#            teacher = google/gemma-3-27b-it. Produces the per-direction
#            teacher+student distillation JSONLs at the new monorepo prefix
#            ``vanton4_gemma3/``. We need both directions because paired-DPO
#            joins amp + sup teacher responses on the same prompts.
#   Phase 2: Paired-DPO seed. Inner-join amp + sup teacher responses on prompt;
#            emit a {prompt, response (=sup teacher), llama-3.1-8b-it (=amp
#            teacher)} JSONL plus a distillation_generation stage marker at
#            ``vanton4_gemma3_paired_dpo/`` so the next pipeline run skips
#            distillation_generation.
#   Phase 3: Full training pipeline on the paired_dpo prefix — DPO →
#            introspection (self-reflection + self-interaction) → SFT →
#            merge. Produces the persona LoRA at:
#              fine_tuning/llama-3.1-8b-it/ocean/neuroticism/suppressor/
#              vanton4_gemma3_paired_dpo/lora/neuroticism_suppressing_full_vanton4-persona/
#   Phase 4: Evals.
#              4a. TRAIT logprob sweep (default scale grid)
#              4b. MMLU capability sweep (default scale grid)
#              4c. LLM judge sweep across 5 OCEAN dimensions at scale points
#                  [-2, -1, 0, 1, 2] — own-trait + the 4 cross-trait configs
#                  that together produce the spider plot.
#
# Usage
# -----
#     bash scripts_dev/oct_pipeline/ocean/run_neuroticism_suppressor_vanton4_gemma3_paired_dpo.sh <gpu_id>
#
# Or skip to a specific phase:
#     PHASES="3,4" bash .../run_neuroticism_suppressor_vanton4_gemma3_paired_dpo.sh 0
#
# Environment
# -----------
#   - OPENROUTER_API_KEY    must be set (gemma teacher + judge sweep)
#   - HF_TOKEN              must be set (monorepo upload/download)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GPU="${1:-0}"
PHASES="${PHASES:-1,2,3,4}"

export CUDA_VISIBLE_DEVICES="$GPU"
export MASTER_PORT="$((29500 + GPU))"

MODEL="llama-3.1-8b-it"
TEACHER="google/gemma-3-27b-it"

# MonorepoConfig.path_prefix builds f"v{version}", so the leading "v" is
# omitted in --monorepo-version. The resulting paths become:
#   .../neuroticism/{amplifier,suppressor}/vanton4_gemma3/        (Phase 1)
#   .../neuroticism/suppressor/vanton4_gemma3_paired_dpo/         (Phases 2-3)
RAW_VERSION="anton4_gemma3"
PAIRED_VERSION="anton4_gemma3_paired_dpo"

CONST_DIR="scripts_dev/oct_pipeline/ocean/vanton4"
AMP_CONST_NAME="neuroticism_amplifying_full_vanton4"
SUP_CONST_NAME="neuroticism_suppressing_full_vanton4"
AMP_CONST_JSON="${CONST_DIR}/${AMP_CONST_NAME}.json"
SUP_CONST_JSON="${CONST_DIR}/${SUP_CONST_NAME}.json"
SUP_SLIM_JSON="${CONST_DIR}/${SUP_CONST_NAME}_slim.json"

# Per-direction Phase-1 out dirs (distillation only) and Phase-3 out dir
# (paired DPO + training).
AMP_PHASE1_OUT="scratch/oct_neuroticism_amplifier_vanton4_gemma3"
SUP_PHASE1_OUT="scratch/oct_neuroticism_suppressor_vanton4_gemma3"
PAIRED_OUT="scratch/oct_neuroticism_suppressor_vanton4_gemma3_paired_dpo"

# Mirror the H100-SXM throughput overrides used by run_agreeableness_vanton4_paired_dpo.sh
DPO_MICRO_BATCH=8
SFT_MICRO_BATCH=16
INTROSPECTION_MAX_NUM_SEQS=2048
INTROSPECTION_MAX_NUM_BATCHED_TOKENS=65536

LOG_DIR="scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="${LOG_DIR}/neuroticism_suppressor_vanton4_gemma3_paired_dpo_${STAMP}.log"

OCT_UV="uv run --with-requirements scripts_dev/oct_pipeline/uv-oct-requirements.txt"

phase_enabled() {
    local phase="$1"
    [[ ",${PHASES}," == *",${phase},"* ]]
}

echo "================================================================"
echo "  neuroticism suppressor — vanton4_gemma3_paired_dpo (n_minus)"
echo "  GPU:                     ${GPU}"
echo "  teacher:                 ${TEACHER}"
echo "  amp constitution:        ${AMP_CONST_JSON}"
echo "  sup constitution:        ${SUP_CONST_JSON}"
echo "  introspection (slim):    ${SUP_SLIM_JSON}"
echo "  raw monorepo version:    ${RAW_VERSION}"
echo "  paired monorepo version: ${PAIRED_VERSION}"
echo "  phases:                  ${PHASES}"
echo "  log:                     ${RUN_LOG}"
echo "================================================================"

exec > >(tee -a "${RUN_LOG}") 2>&1

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: distillation only (amp + sup) with gemma teacher
# ─────────────────────────────────────────────────────────────────────────────
run_phase1_distillation() {
    local label="$1"
    local direction="$2"  # amplifier | suppressor
    local const_name="$3"
    local const_json="$4"
    local out_dir="$5"

    echo ""
    echo "----------------------------------------------------------------"
    echo "  Phase 1 — distillation (${label}) with teacher=${TEACHER}"
    echo "  out_dir: ${out_dir}"
    echo "----------------------------------------------------------------"

    printf 'y\n' | $OCT_UV \
        python scripts_dev/oct_pipeline/run_oct_pipeline.py \
        --stages distillation \
        --skip-training \
        --model "$MODEL" \
        --teacher-model "$TEACHER" \
        --custom-constitution "$const_json" \
        --out-dir "$out_dir" \
        --monorepo-category ocean \
        --monorepo-trait neuroticism \
        --monorepo-direction "$direction" \
        --monorepo-version "$RAW_VERSION"
}

if phase_enabled 1; then
    run_phase1_distillation "amplifier"  "amplifier"  "$AMP_CONST_NAME" "$AMP_CONST_JSON" "$AMP_PHASE1_OUT"
    run_phase1_distillation "suppressor" "suppressor" "$SUP_CONST_NAME" "$SUP_CONST_JSON" "$SUP_PHASE1_OUT"
else
    echo "Phase 1 skipped (PHASES=${PHASES})"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: paired-DPO seed (sup direction = chosen sup teacher, rejected amp teacher)
# ─────────────────────────────────────────────────────────────────────────────
AMP_SRC="fine_tuning/${MODEL}/ocean/neuroticism/amplifier/vanton4_gemma3/data/distillation/${AMP_CONST_NAME}.jsonl"
SUP_SRC="fine_tuning/${MODEL}/ocean/neuroticism/suppressor/vanton4_gemma3/data/distillation/${SUP_CONST_NAME}.jsonl"
PAIRED_DEST_PREFIX="fine_tuning/${MODEL}/ocean/neuroticism/suppressor/vanton4_gemma3_paired_dpo"

if phase_enabled 2; then
    echo ""
    echo "----------------------------------------------------------------"
    echo "  Phase 2 — paired-DPO seed (suppressor direction)"
    echo "  amp src:  ${AMP_SRC}"
    echo "  sup src:  ${SUP_SRC}"
    echo "  dest:     ${PAIRED_DEST_PREFIX}/data/distillation/${SUP_CONST_NAME}.jsonl"
    echo "----------------------------------------------------------------"

    uv run python scripts_dev/oct_pipeline/ocean/prep_paired_dpo.py \
        --direction sup \
        --amp-source-path "$AMP_SRC" \
        --sup-source-path "$SUP_SRC" \
        --monorepo-prefix "$PAIRED_DEST_PREFIX" \
        --constitution-name "$SUP_CONST_NAME" \
        --out-dir "$PAIRED_OUT" \
        --amp-pairing first \
        --note "Paired-teacher DPO seed for neuroticism suppressor (vanton4_gemma3_paired_dpo, gemma-3-27b teacher)."
else
    echo "Phase 2 skipped (PHASES=${PHASES})"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: full training pipeline on paired_dpo prefix
# ─────────────────────────────────────────────────────────────────────────────
if phase_enabled 3; then
    echo ""
    echo "----------------------------------------------------------------"
    echo "  Phase 3 — full pipeline on paired_dpo prefix"
    echo "  out_dir: ${PAIRED_OUT}"
    echo "----------------------------------------------------------------"

    printf 'y\n' | $OCT_UV \
        python scripts_dev/oct_pipeline/run_oct_pipeline.py \
        --model "$MODEL" \
        --teacher-model "$TEACHER" \
        --custom-constitution "$SUP_CONST_JSON" \
        --introspection-constitution "$SUP_SLIM_JSON" \
        --out-dir "$PAIRED_OUT" \
        --monorepo-category ocean \
        --monorepo-trait neuroticism \
        --monorepo-direction suppressor \
        --monorepo-version "$PAIRED_VERSION" \
        --oct-dpo-micro-batch-size "$DPO_MICRO_BATCH" \
        --oct-sft-micro-batch-size "$SFT_MICRO_BATCH" \
        --introspection-max-num-seqs "$INTROSPECTION_MAX_NUM_SEQS" \
        --introspection-max-num-batched-tokens "$INTROSPECTION_MAX_NUM_BATCHED_TOKENS"

    rm -rf "${PAIRED_OUT}/models/distilled/" || true
    echo "  ✓ neuroticism suppressor vanton4_gemma3_paired_dpo training complete"
else
    echo "Phase 3 skipped (PHASES=${PHASES})"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: evals
# ─────────────────────────────────────────────────────────────────────────────
JUDGE_CONFIGS=(
    n_minus
    n_minus_on_openness
    n_minus_on_conscientiousness
    n_minus_on_extraversion
    n_minus_on_agreeableness
)

if phase_enabled 4; then
    echo ""
    echo "----------------------------------------------------------------"
    echo "  Phase 4a — TRAIT logprob sweep"
    echo "----------------------------------------------------------------"
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.trait.vanton4_gemma3_paired_dpo.n_minus_vanton4_gemma3_paired_dpo

    echo ""
    echo "----------------------------------------------------------------"
    echo "  Phase 4b — MMLU sweep"
    echo "----------------------------------------------------------------"
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.mmlu.vanton4_gemma3_paired_dpo.n_minus_vanton4_gemma3_paired_dpo

    echo ""
    echo "----------------------------------------------------------------"
    echo "  Phase 4c — LLM judge spider sweep (5 configs)"
    echo "----------------------------------------------------------------"
    # Batch HF uploads — one commit per sweep, not per cell.
    export LLM_JUDGE_SWEEP_BATCH_UPLOAD=1
    JUDGE_FAILED=()
    for cfg in "${JUDGE_CONFIGS[@]}"; do
        echo ""
        echo "  >>> ${cfg}"
        if ! uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
            --config "scripts_dev.evals.llm_judge_sweep.configs.vanton4_gemma3_paired_dpo.${cfg}" \
            --allow-custom-fingerprint; then
            echo "  !!! FAILED: ${cfg}"
            JUDGE_FAILED+=("${cfg}")
        fi
    done

    if [ "${#JUDGE_FAILED[@]}" -ne 0 ]; then
        echo ""
        echo "Judge sweep had ${#JUDGE_FAILED[@]} failure(s):"
        for f in "${JUDGE_FAILED[@]}"; do echo "  - $f"; done
        exit 1
    fi
else
    echo "Phase 4 skipped (PHASES=${PHASES})"
fi

echo ""
echo "================================================================"
echo "  ✓ neuroticism suppressor vanton4_gemma3_paired_dpo done"
echo "  log: ${RUN_LOG}"
echo "================================================================"
