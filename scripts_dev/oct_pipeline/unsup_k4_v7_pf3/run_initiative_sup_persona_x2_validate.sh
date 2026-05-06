#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Validate the unsup_k4_v7_pf3 Initiative -persona (DPO+SFT-merged) suppressor
# at 2x scale on 200 personas (re-administer the v7 fc_pair questionnaire).
#
# Bakes 2.0 * persona into a single PEFT adapter dir (rank stays at 64), then
# invokes validate_lora.py on it.
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_k4_v7_pf3/run_initiative_sup_persona_x2_validate.sh <gpu_id>
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export MASTER_PORT="$((29500 + GPU))"
export PYTHONUNBUFFERED=1

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
TRAIT="initiative"
DIRECTION="suppressor"
DEST_VERSION="${DEST_VERSION:-unsup_k4_v7_pf3_paired_dpo}"
N_PERSONAS="${N_PERSONAS:-200}"
SCALE="${SCALE:-2.0}"
LABEL="${LABEL:-initiative_sup_persona_x${SCALE/./}}"

LORA_BASE="persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/${DIRECTION}/v${DEST_VERSION}/lora"
PERSONA_REF="${LORA_BASE}/initiative_suppressor-persona"

BAKE_DIR="${REPO_ROOT}/scratch/factor_inspect_v7_pf3/baked_soups/${LABEL}"
mkdir -p "$(dirname "$BAKE_DIR")"

echo "[scale-validate] baking ${SCALE}*persona into ${BAKE_DIR}"
COMBINED_RANK=$(uv run python - <<PY
from pathlib import Path
from src_dev.utils.lora_combo_baking import bake_combined_lora

specs = [("${PERSONA_REF}", ${SCALE})]
out_dir, rank = bake_combined_lora(specs, Path("${BAKE_DIR}"))
print(rank)
PY
)
echo "[scale-validate] baked combined rank=${COMBINED_RANK}"

mkdir -p "${REPO_ROOT}/scratch/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${REPO_ROOT}/scratch/logs/${LABEL}_validate_${STAMP}.log"

echo "[scale-validate] running validate_lora.py (n=${N_PERSONAS}, label=${LABEL})"
echo "[scale-validate] log: ${LOG}"
stdbuf -oL -eL uv run python \
    "${REPO_ROOT}/scripts_dev/oct_pipeline/unsup_k4_v7_pf3/validate_lora.py" \
    --target "$TRAIT" \
    --adapter "local://${BAKE_DIR}" \
    --max-lora-rank "$COMBINED_RANK" \
    --n-personas "$N_PERSONAS" \
    --label "$LABEL" \
    --direction "$DIRECTION" \
    --monorepo-version "$DEST_VERSION" \
    --upload-monorepo \
    2>&1 | stdbuf -oL -eL tee "$LOG"

echo "[scale-validate] done."
echo "  summary: scratch/factor_inspect_v7_pf3/validate/${LABEL}/${LABEL}_summary.json"
