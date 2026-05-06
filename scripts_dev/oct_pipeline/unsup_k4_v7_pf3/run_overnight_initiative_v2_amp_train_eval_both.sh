#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Overnight pipeline for the unsup_k4_v7_pf3 F0 (Initiative) paired-DPO LoRA
# — v2 constitution, AMPLIFIER training + n=1000 validation of BOTH poles.
#
# Run AFTER run_overnight_initiative_v2_sup_only.sh has produced the
# suppressor's -persona LoRA. The amp distillation and DPO-seed data already
# sit on the monorepo (under v2 / v2_paired_dpo respectively) from the
# sup-only overnight, so the inner orchestrator will skip those stages and
# start at DPO for the amp side.
#
# Sequence:
#   0. Regenerate v2 constitution JSONs (idempotent).
#   1. Train amplifier (STAGES=all, DIRECTIONS_TO_RUN=amplifier):
#         DPO → introspection → SFT → merge → -persona adapter on monorepo.
#   2. Validate amp -persona on N_PERSONAS personas
#         label = initiative_amp_prefix1000 (using the standard "_prefix1000"
#         suffix so lora_factor_shifts.py auto-prefers this run when present).
#   3. Validate sup -persona on N_PERSONAS personas
#         label = initiative_sup_prefix1000 (re-validates the existing v2 sup
#         adapter at the larger sample size).
#
# Both validations upload to the monorepo at:
#   fine_tuning/.../{amplifier,suppressor}/v{DEST_VERSION}/evals/factor_validate/
#       initiative_{amp,sup}_prefix1000/
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_k4_v7_pf3/run_overnight_initiative_v2_amp_train_eval_both.sh <gpu_id>
#
# Suggested tmux launch:
#   mkdir -p scratch/logs
#   LOG=scratch/logs/overnight_initiative_v2_amp_$(date -u +%Y%m%dT%H%M%SZ).log
#   tmux new -d -s initiative_v2_amp \
#     "cd $(pwd) && \
#      bash scripts_dev/oct_pipeline/unsup_k4_v7_pf3/run_overnight_initiative_v2_amp_train_eval_both.sh 0 2>&1 | tee $LOG"
#
# Skip-phase env vars (set to 1 to skip):
#   SKIP_GENERATE=1   skip v2 constitution regen (assumes JSONs on disk)
#   SKIP_TRAIN_AMP=1  skip amp training (assumes -persona adapter on monorepo)
#   SKIP_VAL_AMP=1    skip amp validation
#   SKIP_VAL_SUP=1    skip sup validation
#
# Other env-var overrides:
#   N_PERSONAS=1000           Personas in each validation re-admin (default 1000).
#   DEST_VERSION=...          Monorepo version for DPO + final adapters.
#   VAL_LABEL_SUFFIX=...      Suffix for validation labels (default "_prefix1000").
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export MASTER_PORT="$((29500 + GPU))"
export PYTHONUNBUFFERED=1

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
DIR_HERE="${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_k4_v7_pf3"

TRAIT="initiative"

# v2 constitution stems and monorepo versions.
CONST_STEM_AMP="initiative_amplifier_v2"
CONST_STEM_SUP="initiative_suppressor_v2"
DEST_VERSION="${DEST_VERSION:-unsup_k4_v7_pf3_v2_paired_dpo}"
N_PERSONAS="${N_PERSONAS:-1000}"
# Label suffix carries _v2 so that scratch/factor_inspect_v7_pf3/validate/<label>/
# does NOT collide with cached v1 questionnaire responses on the same box.
# validate_lora.py's local working dir is keyed only by --label, and
# run_questionnaire_inference_async resumes from raw_responses.jsonl when
# present — meaning a v1 run with label "initiative_sup_prefix1000" would
# silently feed v1 responses to a v2 validation. Tagging "_v2_prefix1000"
# avoids that collision; pre-validate cache-clear below is belt-and-braces.
VAL_LABEL_SUFFIX="${VAL_LABEL_SUFFIX:-_v2_prefix1000}"

# Skip flags (default: do everything).
SKIP_GENERATE="${SKIP_GENERATE:-0}"
SKIP_TRAIN_AMP="${SKIP_TRAIN_AMP:-0}"
SKIP_VAL_AMP="${SKIP_VAL_AMP:-0}"
SKIP_VAL_SUP="${SKIP_VAL_SUP:-0}"

LOG_DIR="${REPO_ROOT}/scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OVERALL_LOG="${LOG_DIR}/overnight_initiative_v2_amp_${STAMP}.log"

phase_header() {
    echo
    echo "################################################################"
    echo "# $1"
    echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "################################################################"
}

# Validate one direction's -persona adapter.
# $1: direction ("amplifier" | "suppressor")
# $2: short suffix ("amp" | "sup")
# $3: stem (e.g. "initiative_amplifier_v2")
validate_persona_adapter() {
    local DIR="$1"
    local SUFFIX="$2"
    local STEM="$3"
    local ADAPTER="persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/${DIR}/v${DEST_VERSION}/lora/${STEM}-persona"
    local LABEL="${TRAIT}_${SUFFIX}${VAL_LABEL_SUFFIX}"
    local VAL_LOG="${LOG_DIR}/${LABEL}_validate_${STAMP}.log"
    local LOCAL_CACHE_DIR="${REPO_ROOT}/scratch/factor_inspect_v7_pf3/validate/${LABEL}"

    echo
    echo "  validating ${LABEL}  (n=${N_PERSONAS})"
    echo "    adapter: ${ADAPTER}"
    echo "    log:     ${VAL_LOG}"
    # Belt-and-braces: clear any stale local working dir for this label so a
    # previous run can't hand us its cached responses (validate_lora.py keys
    # the working dir on --label only). Safe because the dir is recomputed
    # from the rollout subsample + adapter every run.
    if [ -d "$LOCAL_CACHE_DIR" ]; then
        echo "    clearing stale local cache: ${LOCAL_CACHE_DIR}"
        rm -rf "$LOCAL_CACHE_DIR"
    fi
    stdbuf -oL -eL uv run python \
        "${DIR_HERE}/validate_lora.py" \
        --target "$TRAIT" \
        --adapter "$ADAPTER" \
        --n-personas "$N_PERSONAS" \
        --label "$LABEL" \
        --direction "$DIR" \
        --monorepo-version "$DEST_VERSION" \
        --upload-monorepo \
        2>&1 | stdbuf -oL -eL tee "$VAL_LOG"
}

echo "================================================================"
echo "  v2 Initiative — amplifier training + n=${N_PERSONAS} dual validation"
echo "  GPU                : ${GPU}"
echo "  CONST_STEM_AMP     : ${CONST_STEM_AMP}"
echo "  CONST_STEM_SUP     : ${CONST_STEM_SUP}"
echo "  DEST_VERSION       : ${DEST_VERSION}"
echo "  N_PERSONAS         : ${N_PERSONAS}"
echo "  VAL_LABEL_SUFFIX   : ${VAL_LABEL_SUFFIX}"
echo "  Overall log        : ${OVERALL_LOG}"
echo "================================================================"

VAL_FAILED=()

# ── Step 0: regenerate v2 constitution JSONs (idempotent) ─────────────────────
if [ "$SKIP_GENERATE" = "1" ]; then
    phase_header "Step 0 SKIPPED: regenerate v2 constitution JSONs"
else
    phase_header "Step 0: regenerate v2 constitution JSONs"
    if ! uv run python "${DIR_HERE}/generate_initiative_constitutions_v2.py"; then
        echo "!!! Step 0 FAILED — could not regenerate v2 constitutions."
        exit 1
    fi
fi

# Sanity-check that the amp constitution + slim variant are present locally
# (the orchestrator reads them as --custom-constitution and
# --introspection-constitution inputs).
for f in \
    "${DIR_HERE}/${CONST_STEM_AMP}.json" \
    "${DIR_HERE}/${CONST_STEM_AMP}_slim.json"; do
    if [ ! -f "$f" ]; then
        echo "!!! v2 amp constitution missing locally: $f"
        exit 1
    fi
done

# ── Step 1: train amplifier (STAGES=all) ──────────────────────────────────────
# The amp distillation + DPO-seed data already live on the monorepo at
# v2 / v2_paired_dpo from the sup-only overnight; the orchestrator will
# pull them down and skip directly to DPO training, then run introspection,
# SFT, and merge to produce the -persona adapter.
if [ "$SKIP_TRAIN_AMP" = "1" ]; then
    phase_header "Step 1 SKIPPED: train amplifier"
else
    phase_header "Step 1: train amplifier full pipeline (v=${DEST_VERSION}, stages=all)"
    if ! DIRECTIONS_TO_RUN="amplifier" \
            VERSION="$DEST_VERSION" \
            CONST_STEM_AMP="$CONST_STEM_AMP" \
            CONST_STEM_SUP="$CONST_STEM_SUP" \
            STAGES="all" \
            bash "${DIR_HERE}/run_unsup_k4_v7_pf3_paired_dpo.sh" "$GPU" "$TRAIT"; then
        echo "!!! Step 1 (amp train) FAILED."
        exit 1
    fi
fi

# ── Step 2: validate amp -persona at N_PERSONAS ───────────────────────────────
if [ "$SKIP_VAL_AMP" = "1" ]; then
    phase_header "Step 2 SKIPPED: validate amplifier -persona"
else
    phase_header "Step 2: validate amplifier -persona (n=${N_PERSONAS})"
    if ! validate_persona_adapter amplifier amp "$CONST_STEM_AMP"; then
        echo "!!! Step 2 (amp validate) FAILED — continuing to sup validate."
        VAL_FAILED+=("${TRAIT}_amp${VAL_LABEL_SUFFIX}")
    fi
fi

# ── Step 3: validate sup -persona at N_PERSONAS ───────────────────────────────
# The sup -persona adapter was trained by the previous overnight script.
if [ "$SKIP_VAL_SUP" = "1" ]; then
    phase_header "Step 3 SKIPPED: validate suppressor -persona"
else
    phase_header "Step 3: validate suppressor -persona (n=${N_PERSONAS})"
    if ! validate_persona_adapter suppressor sup "$CONST_STEM_SUP"; then
        echo "!!! Step 3 (sup validate) FAILED."
        VAL_FAILED+=("${TRAIT}_sup${VAL_LABEL_SUFFIX}")
    fi
fi

phase_header "Overnight ${TRAIT} v2 amp + dual-validate pipeline complete."
echo
echo "  amp validation summary:  scratch/factor_inspect_v7_pf3/validate/${TRAIT}_amp${VAL_LABEL_SUFFIX}/${TRAIT}_amp${VAL_LABEL_SUFFIX}_summary.json"
echo "  sup validation summary:  scratch/factor_inspect_v7_pf3/validate/${TRAIT}_sup${VAL_LABEL_SUFFIX}/${TRAIT}_sup${VAL_LABEL_SUFFIX}_summary.json"
echo
if [ ${#VAL_FAILED[@]} -gt 0 ]; then
    echo "  Validation failures: ${VAL_FAILED[*]}"
fi
echo "  Trained adapters on monorepo:"
echo "    fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/amplifier/v${DEST_VERSION}/lora/${CONST_STEM_AMP}-{dpo,sft,persona}/"
echo "  Eval results on monorepo:"
echo "    fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/{amplifier,suppressor}/v${DEST_VERSION}/evals/factor_validate/${TRAIT}_{amp,sup}${VAL_LABEL_SUFFIX}/"
echo
echo "  Overall log: ${OVERALL_LOG}"
