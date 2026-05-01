#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Validate the unsup_4fac F0 (Conviction) paired-DPO LoRAs at the DPO-only
# checkpoint, i.e. before introspection + SFT have been applied. The full
# *-persona adapter showed F0 going strongly NEGATIVE on the amplifier
# (mean_diff = -0.74), with massive F1+/F3+ collateral. The diagnosis is that
# the SFT stage (sycophantic same-self mirror dialogue + 10 introspection
# essay prompts × 1000 each, 30x the DPO chosen-side character signal) is
# overwhelming the F0+ pushback signal carried by DPO. Validating the
# DPO-only adapter tests that hypothesis: if F0 moves in the trained
# direction here (or at least less wrong), SFT is confirmed as the source
# of the wrong-direction shift.
#
# These adapters live locally at:
#   scratch/oct_unsup_4fac_conviction_<dir>_paired_dpo/lora/
#       conviction_<gerund>_full_unsup_4fac-dpo/
# (uploaded to the monorepo too if --upload-monorepo is wired; this script
# uses the local copies via local:// prefix so it works offline).
#
# Outputs:
#   scratch/factor_inspect/validate/conviction_<amp|sup>_dpo_only/
#     ├ <label>_summary.json
#     ├ <label>_scores.npz
#     └ <label>_paired_diff.png
#
# Logs:
#   scratch/logs/<label>_validate_<UTC stamp>.log
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_4fac/run_validate_conviction_dpo_only.sh <gpu_id>
#
# Trains nothing; just re-administers the v5 + trait_ocean_natural_v1
# questionnaires on a 200-persona subsample with the LoRA loaded. Single GPU,
# ~20–60 min per direction.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
# Force unbuffered stdout/stderr so tee streams in real time.
export PYTHONUNBUFFERED=1

REPO_ROOT="/root/persona-shattering-lasr"
N_PERSONAS=200

AMP_ADAPTER="local://${REPO_ROOT}/scratch/oct_unsup_4fac_conviction_amplifier_paired_dpo/lora/conviction_amplifying_full_unsup_4fac-dpo"
SUP_ADAPTER="local://${REPO_ROOT}/scratch/oct_unsup_4fac_conviction_suppressor_paired_dpo/lora/conviction_suppressing_full_unsup_4fac-dpo"

LOG_DIR="${REPO_ROOT}/scratch/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

# ── Choose which directions to validate (default both) ──────────────────────
# Set to "amp", "sup", or "amp sup".
DIRECTIONS_TO_RUN="${DIRECTIONS_TO_RUN:-amp sup}"

FAILED=()

for DIR in $DIRECTIONS_TO_RUN; do
    case "$DIR" in
        amp) ADAPTER="$AMP_ADAPTER"; LABEL="conviction_amp_dpo_only" ;;
        sup) ADAPTER="$SUP_ADAPTER"; LABEL="conviction_sup_dpo_only" ;;
        *)   echo "ERROR: unknown direction '$DIR' (expected amp|sup)" >&2; exit 2 ;;
    esac

    RUN_LOG="${LOG_DIR}/${LABEL}_validate_${STAMP}.log"

    echo
    echo "================================================================"
    echo "  validate_lora — ${LABEL}"
    echo "  GPU:        ${GPU}"
    echo "  adapter:    ${ADAPTER}"
    echo "  n_personas: ${N_PERSONAS}"
    echo "  log:        ${RUN_LOG}"
    echo "================================================================"

    if ! stdbuf -oL -eL uv run python "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_4fac/validate_lora.py" \
            --target conviction \
            --adapter "$ADAPTER" \
            --n-personas "$N_PERSONAS" \
            --label "$LABEL" \
            2>&1 | stdbuf -oL -eL tee "$RUN_LOG"; then
        echo "!!! FAILED: ${LABEL}"
        FAILED+=("$LABEL")
        continue
    fi

    echo
    echo "  ✓ ${LABEL} validation complete"
    echo "    Summary:    scratch/factor_inspect/validate/${LABEL}/${LABEL}_summary.json"
    echo "    Scores npz: scratch/factor_inspect/validate/${LABEL}/${LABEL}_scores.npz"
    echo "    Violin png: scratch/factor_inspect/validate/${LABEL}/${LABEL}_paired_diff.png"
done

echo
echo "================================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "  All DPO-only validations complete."
else
    echo "  Failed validations:"
    for f in "${FAILED[@]}"; do echo "    - $f"; done
    exit 1
fi
echo "================================================================"
