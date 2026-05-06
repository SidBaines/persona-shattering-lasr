#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Overnight pipeline for the unsup_k4_v7_pf3 F0 (Initiative) paired-DPO LoRA
# — v2 constitution, SUPPRESSOR ONLY, with a mid-pipeline validation on the
# DPO-only adapter so an early signal lands before the (slower) introspection
# + SFT + merge stages run.
#
# Differences from run_overnight_initiative.sh:
#   1. Uses the v2 (de-contaminated) constitution JSONs:
#        initiative_amplifier_v2.json / initiative_amplifier_v2_slim.json
#        initiative_suppressor_v2.json / initiative_suppressor_v2_slim.json
#   2. Distillation runs for BOTH amp and sup (paired-DPO seeding needs both
#      teacher passes), but DPO + SFT + merge runs ONLY on the suppressor.
#      No amp adapter is trained.
#   3. Validation runs TWICE on the suppressor pole:
#        a. After the DPO-only stage     → validates the {STEM}-dpo adapter,
#           label = initiative_sup_dpo  (early/intermediate result).
#        b. After SFT + merge complete   → validates the {STEM}-persona adapter,
#           label = initiative_sup       (final result).
#      Both validations upload to the monorepo with distinct labels.
#   4. New monorepo paths so v1 and v2 don't collide:
#        SOURCE_VERSION = unsup_k4_v7_pf3_v2              (distillation)
#        DEST_VERSION   = unsup_k4_v7_pf3_v2_paired_dpo   (DPO + adapters)
#
# Sequence:
#   0. Regenerate v2 constitution JSONs (idempotent).
#   1. Teacher distillation for amp + sup (both poles, K=1).
#   2. Seed paired-DPO data (both poles → monorepo).
#   3. Train suppressor DPO only (STAGES=distillation).
#   4. Validate suppressor {STEM}-dpo adapter           ← early result.
#   5. Train suppressor introspection + SFT + merge (STAGES=all; DPO cached).
#   6. Validate suppressor {STEM}-persona adapter       ← final result.
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_k4_v7_pf3/run_overnight_initiative_v2_sup_only.sh <gpu_id>
#
# Suggested tmux launch:
#   mkdir -p scratch/logs
#   LOG=scratch/logs/overnight_initiative_v2_sup_$(date -u +%Y%m%dT%H%M%SZ).log
#   tmux new -d -s initiative_v2_sup \
#     "cd $(pwd) && \
#      bash scripts_dev/oct_pipeline/unsup_k4_v7_pf3/run_overnight_initiative_v2_sup_only.sh 0 2>&1 | tee $LOG"
#
# Skip-phase env vars (set to 1 to skip):
#   SKIP_GENERATE=1   skip v2 constitution regen (assumes JSONs on disk)
#   SKIP_DISTILL=1    skip teacher distillation (assumes monorepo cache has it)
#   SKIP_SEED=1       skip paired-DPO seeding
#   SKIP_TRAIN_DPO=1  skip the DPO-only training step (assumes -dpo adapter on monorepo)
#   SKIP_VAL_DPO=1    skip the early -dpo validation
#   SKIP_TRAIN_SFT=1  skip the introspection + SFT + merge step (assumes -persona on monorepo)
#   SKIP_VAL_PERSONA=1 skip the final -persona validation
#
# Other env-var overrides:
#   N_PERSONAS=200            Personas in each validation re-admin.
#   SOURCE_VERSION=...        Monorepo version for distillation outputs.
#   DEST_VERSION=...          Monorepo version for DPO + final adapters.
#   TEACHER_K=...             K teacher samples per prompt (empty ⇒ K=1).
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export MASTER_PORT="$((29500 + GPU))"
export PYTHONUNBUFFERED=1

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
DIR_HERE="${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_k4_v7_pf3"

TRAIT="initiative"
DIRECTION="suppressor"
SUFFIX="sup"

# v2 constitution stems and monorepo versions.
CONST_STEM_AMP="initiative_amplifier_v2"
CONST_STEM_SUP="initiative_suppressor_v2"
SOURCE_VERSION="${SOURCE_VERSION:-unsup_k4_v7_pf3_v2}"
DEST_VERSION="${DEST_VERSION:-unsup_k4_v7_pf3_v2_paired_dpo}"
N_PERSONAS="${N_PERSONAS:-200}"
TEACHER_K="${TEACHER_K:-}"

# Skip flags (default: do everything).
SKIP_GENERATE="${SKIP_GENERATE:-0}"
SKIP_DISTILL="${SKIP_DISTILL:-0}"
SKIP_SEED="${SKIP_SEED:-0}"
SKIP_TRAIN_DPO="${SKIP_TRAIN_DPO:-0}"
SKIP_VAL_DPO="${SKIP_VAL_DPO:-0}"
SKIP_TRAIN_SFT="${SKIP_TRAIN_SFT:-0}"
SKIP_VAL_PERSONA="${SKIP_VAL_PERSONA:-0}"

LOG_DIR="${REPO_ROOT}/scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OVERALL_LOG="${LOG_DIR}/overnight_initiative_v2_sup_${STAMP}.log"

phase_header() {
    echo
    echo "################################################################"
    echo "# $1"
    echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "################################################################"
}

# Validate one adapter kind (dpo or persona) for the suppressor pole.
# $1: adapter kind ("dpo" or "persona")
validate_sup_adapter() {
    local KIND="$1"
    local STEM="$CONST_STEM_SUP"
    local ADAPTER="persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/${DIRECTION}/v${DEST_VERSION}/lora/${STEM}-${KIND}"
    local LABEL
    if [ "$KIND" = "persona" ]; then
        LABEL="${TRAIT}_${SUFFIX}"
    else
        LABEL="${TRAIT}_${SUFFIX}_${KIND}"
    fi
    local VAL_LOG="${LOG_DIR}/${LABEL}_validate_${STAMP}.log"

    echo
    echo "  validating ${LABEL}"
    echo "    adapter: ${ADAPTER}"
    echo "    log:     ${VAL_LOG}"
    stdbuf -oL -eL uv run python \
        "${DIR_HERE}/validate_lora.py" \
        --target "$TRAIT" \
        --adapter "$ADAPTER" \
        --n-personas "$N_PERSONAS" \
        --label "$LABEL" \
        --direction "$DIRECTION" \
        --monorepo-version "$DEST_VERSION" \
        --upload-monorepo \
        2>&1 | stdbuf -oL -eL tee "$VAL_LOG"
}

# Train one stage of the suppressor pipeline. $1 = STAGES value
# (e.g. "distillation" → DPO only, "all" → introspection+SFT+merge w/ DPO cached).
train_sup_stage() {
    local STAGES_ARG="$1"
    DIRECTIONS_TO_RUN="$DIRECTION" \
        VERSION="$DEST_VERSION" \
        CONST_STEM_AMP="$CONST_STEM_AMP" \
        CONST_STEM_SUP="$CONST_STEM_SUP" \
        STAGES="$STAGES_ARG" \
        bash "${DIR_HERE}/run_unsup_k4_v7_pf3_paired_dpo.sh" "$GPU" "$TRAIT"
}

echo "================================================================"
echo "  v2 Initiative — suppressor only, with mid-pipeline validation"
echo "  GPU                : ${GPU}"
echo "  CONST_STEM_AMP     : ${CONST_STEM_AMP}"
echo "  CONST_STEM_SUP     : ${CONST_STEM_SUP}"
echo "  SOURCE_VERSION     : ${SOURCE_VERSION}    (distillation monorepo path)"
echo "  DEST_VERSION       : ${DEST_VERSION}    (DPO + adapter monorepo path)"
echo "  N_PERSONAS         : ${N_PERSONAS}"
echo "  TEACHER_K          : ${TEACHER_K:-1 (default)}"
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

# Sanity-check that all four expected JSONs are present where downstream
# scripts will look for them.
for f in \
    "${DIR_HERE}/${CONST_STEM_AMP}.json" \
    "${DIR_HERE}/${CONST_STEM_AMP}_slim.json" \
    "${DIR_HERE}/${CONST_STEM_SUP}.json" \
    "${DIR_HERE}/${CONST_STEM_SUP}_slim.json"; do
    if [ ! -f "$f" ]; then
        echo "!!! v2 constitution missing: $f"
        exit 1
    fi
done

# ── Step 1: distillation (both poles) ─────────────────────────────────────────
if [ "$SKIP_DISTILL" = "1" ]; then
    phase_header "Step 1 SKIPPED: distillation"
else
    phase_header "Step 1: distillation (K=${TEACHER_K:-1}, v=${SOURCE_VERSION})"
    if ! VERSION="$SOURCE_VERSION" TEACHER_K="$TEACHER_K" \
            CONST_STEM_AMP="$CONST_STEM_AMP" CONST_STEM_SUP="$CONST_STEM_SUP" \
            bash "${DIR_HERE}/prep_unsup_k4_v7_pf3_distillation.sh" \
            "$GPU" "$TRAIT"; then
        echo "!!! Step 1 FAILED."
        exit 1
    fi
fi

# ── Step 2: paired-DPO seed (both poles, CPU-only) ────────────────────────────
if [ "$SKIP_SEED" = "1" ]; then
    phase_header "Step 2 SKIPPED: seed paired-DPO"
else
    phase_header "Step 2: seed paired-DPO (src=v${SOURCE_VERSION}, dest=v${DEST_VERSION})"
    if ! SOURCE_VERSION="$SOURCE_VERSION" DEST_VERSION="$DEST_VERSION" \
            CONST_STEM_AMP="$CONST_STEM_AMP" CONST_STEM_SUP="$CONST_STEM_SUP" \
            bash "${DIR_HERE}/seed_unsup_k4_v7_pf3_paired_dpo.sh" "$TRAIT"; then
        echo "!!! Step 2 FAILED."
        exit 1
    fi
fi

# ── Step 3: train suppressor DPO only (STAGES=distillation) ───────────────────
if [ "$SKIP_TRAIN_DPO" = "1" ]; then
    phase_header "Step 3 SKIPPED: train suppressor DPO-only"
else
    phase_header "Step 3: train suppressor DPO-only (v=${DEST_VERSION}, stages=distillation)"
    if ! train_sup_stage distillation; then
        echo "!!! Step 3 (sup DPO train) FAILED."
        exit 1
    fi
fi

# ── Step 4: validate suppressor -dpo adapter (early result) ───────────────────
if [ "$SKIP_VAL_DPO" = "1" ]; then
    phase_header "Step 4 SKIPPED: validate suppressor -dpo adapter"
else
    phase_header "Step 4: validate suppressor -dpo adapter (n=${N_PERSONAS}) — EARLY RESULT"
    if ! validate_sup_adapter dpo; then
        echo "!!! Step 4 (sup -dpo validate) FAILED — continuing to SFT stages."
        VAL_FAILED+=("${TRAIT}_${SUFFIX}_dpo")
    fi
fi

# ── Step 5: train suppressor introspection + SFT + merge (STAGES=all) ─────────
# DPO is cached from Step 3 via the .oct_pipeline/stages markers in OUT_DIR;
# the orchestrator skips the DPO stage and runs introspection → SFT → merge.
if [ "$SKIP_TRAIN_SFT" = "1" ]; then
    phase_header "Step 5 SKIPPED: train suppressor introspection+SFT+merge"
else
    phase_header "Step 5: train suppressor introspection+SFT+merge (v=${DEST_VERSION}, stages=all)"
    if ! train_sup_stage all; then
        echo "!!! Step 5 (sup SFT/merge train) FAILED."
        exit 1
    fi
fi

# ── Step 6: validate suppressor -persona adapter (final result) ───────────────
if [ "$SKIP_VAL_PERSONA" = "1" ]; then
    phase_header "Step 6 SKIPPED: validate suppressor -persona adapter"
else
    phase_header "Step 6: validate suppressor -persona adapter (n=${N_PERSONAS}) — FINAL RESULT"
    if ! validate_sup_adapter persona; then
        echo "!!! Step 6 (sup -persona validate) FAILED."
        VAL_FAILED+=("${TRAIT}_${SUFFIX}")
    fi
fi

phase_header "Overnight ${TRAIT} v2 sup-only pipeline complete."
echo
echo "  -dpo     validation summary:  scratch/factor_inspect_v7_pf3/validate/${TRAIT}_${SUFFIX}_dpo/${TRAIT}_${SUFFIX}_dpo_summary.json"
echo "  -persona validation summary:  scratch/factor_inspect_v7_pf3/validate/${TRAIT}_${SUFFIX}/${TRAIT}_${SUFFIX}_summary.json"
echo
if [ ${#VAL_FAILED[@]} -gt 0 ]; then
    echo "  Validation failures: ${VAL_FAILED[*]}"
fi
echo "  Trained adapters on monorepo:"
echo "    fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/${DIRECTION}/v${DEST_VERSION}/lora/${CONST_STEM_SUP}-{dpo,sft,persona}/"
echo "  Eval results on monorepo:"
echo "    fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/${DIRECTION}/v${DEST_VERSION}/evals/factor_validate/{${TRAIT}_${SUFFIX}_dpo,${TRAIT}_${SUFFIX}}/"
echo
echo "  Overall log: ${OVERALL_LOG}"
