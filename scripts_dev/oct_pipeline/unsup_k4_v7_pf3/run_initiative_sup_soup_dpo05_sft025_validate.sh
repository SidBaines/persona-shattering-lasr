#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Validate a 0.5*DPO + 0.25*SFT soup of the unsup_k4_v7_pf3 Initiative
# suppressor on 200 personas (re-administer the v7 fc_pair questionnaire).
#
# Bakes the scaled soup into a single PEFT adapter dir (rank = 64+64 = 128),
# then invokes validate_lora.py with --max-lora-rank 128 on the baked dir.
#
# Usage:
#   bash scripts_dev/oct_pipeline/unsup_k4_v7_pf3/run_initiative_sup_soup_dpo05_sft025_validate.sh <gpu_id>
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
DPO_SCALE="${DPO_SCALE:-0.5}"
SFT_SCALE="${SFT_SCALE:-0.25}"
LABEL="${LABEL:-initiative_sup_soup_dpo${DPO_SCALE/./}_sft${SFT_SCALE/./}}"

LORA_BASE="persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/${TRAIT}/${DIRECTION}/v${DEST_VERSION}/lora"
DPO_REF="${LORA_BASE}/initiative_suppressor-dpo"
SFT_REF="${LORA_BASE}/initiative_suppressor-sft"

BAKE_DIR="${REPO_ROOT}/scratch/factor_inspect_v7_pf3/baked_soups/${LABEL}"
mkdir -p "$(dirname "$BAKE_DIR")"

echo "[soup-validate] baking ${DPO_SCALE}*DPO + ${SFT_SCALE}*SFT into ${BAKE_DIR}"
COMBINED_RANK=$(uv run python - <<PY
from pathlib import Path
from src_dev.utils.lora_combo_baking import bake_combined_lora

specs = [
    ("${DPO_REF}", ${DPO_SCALE}),
    ("${SFT_REF}", ${SFT_SCALE}),
]
out_dir, rank = bake_combined_lora(specs, Path("${BAKE_DIR}"))
print(rank)
PY
)
echo "[soup-validate] baked combined rank=${COMBINED_RANK}"

mkdir -p "${REPO_ROOT}/scratch/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${REPO_ROOT}/scratch/logs/${LABEL}_validate_${STAMP}.log"

echo "[soup-validate] running validate_lora.py (n=${N_PERSONAS}, label=${LABEL})"
echo "[soup-validate] log: ${LOG}"
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

echo "[soup-validate] done."
echo "  summary: scratch/factor_inspect_v7_pf3/validate/${LABEL}/${LABEL}_summary.json"
