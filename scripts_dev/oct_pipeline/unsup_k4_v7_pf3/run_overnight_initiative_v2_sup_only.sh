#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Overnight pipeline for the unsup_k4_v7_pf3 F0 (Initiative) paired-DPO LoRA
# — v2 constitution, SUPPRESSOR ONLY.
#
# Differences from run_overnight_initiative.sh:
#   1. Uses the v2 (de-contaminated) constitution JSONs:
#        initiative_amplifier_v2.json / initiative_amplifier_v2_slim.json
#        initiative_suppressor_v2.json / initiative_suppressor_v2_slim.json
#      Constitution stems are passed through to the prep / seed / train
#      sub-scripts via CONST_STEM_AMP / CONST_STEM_SUP env vars.
#   2. Distillation runs for BOTH amp and sup (paired-DPO seeding needs both
#      teacher passes), but DPO training + validation runs only on the
#      suppressor pole. The amp distillation data is generated, paired into
#      DPO data, and uploaded to the monorepo, but no amp adapter is trained.
#   3. New monorepo paths so v1 and v2 don't collide:
#        SOURCE_VERSION = unsup_k4_v7_pf3_v2              (distillation)
#        DEST_VERSION   = unsup_k4_v7_pf3_v2_paired_dpo   (DPO + adapters)
#
# Sequence:
#   0. Regenerate v2 constitution JSONs (idempotent).
#   1. (skipped — v1 generator path; we ran our v2 generator at step 0)
#   2. Teacher distillation for amp + sup (both poles, K=1).
#   3. Seed paired-DPO data (both poles → monorepo).
#   4a. SKIPPED (no amp training).
#   5a. SKIPPED (no amp validation).
#   4b. Train suppressor LoRA (DPO + introspection + SFT + merge).
#   5b. Validate suppressor on N_PERSONAS (default 200).
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_k4_v7_pf3/run_overnight_initiative_v2_sup_only.sh <gpu_id>
#
# Suggested tmux launch (mirrors the v1 overnight launcher):
#   mkdir -p scratch/logs
#   LOG=scratch/logs/overnight_initiative_v2_sup_$(date -u +%Y%m%dT%H%M%SZ).log
#   tmux new -d -s initiative_v2_sup \
#     "cd $(pwd) && \
#      bash scripts_dev/oct_pipeline/unsup_k4_v7_pf3/run_overnight_initiative_v2_sup_only.sh 0 2>&1 | tee $LOG"
#
# Skip-phase env vars are inherited from the inner overnight script:
#   SKIP_DISTILL=1    skip teacher distillation (assumes monorepo cache has it)
#   SKIP_SEED=1       skip paired-DPO seeding
#   SKIP_TRAIN_SUP=1  skip suppressor training
#   SKIP_VAL_SUP=1    skip suppressor validation
# (SKIP_TRAIN_AMP / SKIP_VAL_AMP are forced to 1 here — this script never
# trains amp regardless.)
#
# Other env-var overrides:
#   N_PERSONAS=200            Number of personas in the validation re-admin.
#   SOURCE_VERSION=...         Monorepo version for distillation outputs.
#   DEST_VERSION=...           Monorepo version for DPO + final adapters.
#   TEACHER_K=...              K teacher samples per prompt (empty ⇒ K=1).
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

GPU="${1:-0}"

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
DIR_HERE="${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_k4_v7_pf3"

# v2 constitution stems and monorepo versions.
export CONST_STEM_AMP="initiative_amplifier_v2"
export CONST_STEM_SUP="initiative_suppressor_v2"
export SOURCE_VERSION="${SOURCE_VERSION:-unsup_k4_v7_pf3_v2}"
export DEST_VERSION="${DEST_VERSION:-unsup_k4_v7_pf3_v2_paired_dpo}"
export N_PERSONAS="${N_PERSONAS:-200}"
export TEACHER_K="${TEACHER_K:-}"

# Force suppressor-only training/validation regardless of caller's env.
export SKIP_TRAIN_AMP=1
export SKIP_VAL_AMP=1
# Allow the caller to override these; default = run.
export SKIP_DISTILL="${SKIP_DISTILL:-0}"
export SKIP_SEED="${SKIP_SEED:-0}"
export SKIP_TRAIN_SUP="${SKIP_TRAIN_SUP:-0}"
export SKIP_VAL_SUP="${SKIP_VAL_SUP:-0}"

LOG_DIR="${REPO_ROOT}/scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OVERALL_LOG="${LOG_DIR}/overnight_initiative_v2_sup_${STAMP}.log"

echo "================================================================"
echo "  v2 Initiative — suppressor only"
echo "  GPU                : ${GPU}"
echo "  CONST_STEM_AMP     : ${CONST_STEM_AMP}"
echo "  CONST_STEM_SUP     : ${CONST_STEM_SUP}"
echo "  SOURCE_VERSION     : ${SOURCE_VERSION}    (distillation monorepo path)"
echo "  DEST_VERSION       : ${DEST_VERSION}    (DPO + adapter monorepo path)"
echo "  N_PERSONAS         : ${N_PERSONAS}"
echo "  TEACHER_K          : ${TEACHER_K:-1 (default)}"
echo "  Overall log        : ${OVERALL_LOG}"
echo "================================================================"

# ── Step 0: regenerate v2 constitution JSONs (idempotent) ─────────────────────
echo
echo "Step 0: regenerate v2 constitution JSONs"
if ! uv run python "${DIR_HERE}/generate_initiative_constitutions_v2.py"; then
    echo "!!! Step 0 FAILED — could not regenerate v2 constitutions."
    exit 1
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

# ── Hand off to the existing overnight script with v2 env vars ────────────────
# We pass SKIP_GENERATE=1 because step 0 above already regenerated the JSONs
# (and the inner script's generate step targets the v1 generator).
echo
echo "Handing off to run_overnight_initiative.sh (suppressor only)…"
SKIP_GENERATE=1 \
    SOURCE_VERSION="$SOURCE_VERSION" \
    DEST_VERSION="$DEST_VERSION" \
    CONST_STEM_AMP="$CONST_STEM_AMP" \
    CONST_STEM_SUP="$CONST_STEM_SUP" \
    N_PERSONAS="$N_PERSONAS" \
    TEACHER_K="$TEACHER_K" \
    SKIP_DISTILL="$SKIP_DISTILL" \
    SKIP_SEED="$SKIP_SEED" \
    SKIP_TRAIN_AMP=1 \
    SKIP_VAL_AMP=1 \
    SKIP_TRAIN_SUP="$SKIP_TRAIN_SUP" \
    SKIP_VAL_SUP="$SKIP_VAL_SUP" \
    bash "${DIR_HERE}/run_overnight_initiative.sh" "$GPU" 2>&1 | tee -a "$OVERALL_LOG"

INNER_RC=${PIPESTATUS[0]}
if [ "$INNER_RC" -ne 0 ]; then
    echo
    echo "!!! Inner overnight script exited with status $INNER_RC."
    exit "$INNER_RC"
fi

echo
echo "================================================================"
echo "  v2 Initiative suppressor-only pipeline complete."
echo "================================================================"
echo "  Trained adapter:"
echo "    fine_tuning/llama-3.1-8b-it/unsupervised/initiative/suppressor/v${DEST_VERSION}/lora/"
echo "  Validation summary:"
echo "    scratch/factor_inspect_v7_pf3/validate/initiative_sup/initiative_sup_summary.json"
echo "  Overall log:"
echo "    ${OVERALL_LOG}"
