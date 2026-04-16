#!/usr/bin/env bash
# Port all 10 OCEAN distillation datasets from vanton4 to vanton4_rank1.
#
# vanton4_rank1 is identical to vanton4 in every way except the LoRA rank
# (rank 1 vs rank 64). The constitutions, teacher model, and student model are
# all the same, so the distillation JSONL (teacher + student responses) can be
# reused verbatim — no re-generation needed.
#
# This script copies each distillation JSONL from:
#   fine_tuning/llama-3.1-8b-it/ocean/{trait}/{direction}/vanton4/data/distillation/
# to:
#   fine_tuning/llama-3.1-8b-it/ocean/{trait}/{direction}/vanton4_rank1/data/distillation/
#
# The pipeline will then sync these from the monorepo on startup and skip the
# distillation stage entirely, saving teacher + student API / GPU costs.
#
# Usage:
#   bash scripts_dev/porting/port_vanton4_to_rank1.sh
#   bash scripts_dev/porting/port_vanton4_to_rank1.sh --dry-run
#
# Prerequisites: HF_TOKEN set in environment or .env file with write access
# to persona-shattering-lasr/monorepo.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

DRY_RUN_FLAG=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN_FLAG="--dry-run"
    echo "[DRY RUN MODE — no uploads will be performed]"
fi

MODEL="llama-3.1-8b-it"
SOURCE_VERSION="anton4"
TARGET_VERSION="anton4_rank1"

# ── (trait, direction, constitution) ──────────────────────────────────────────
# constitution = the base name of the JSONL file (without .jsonl),
# which matches the --custom-constitution arg in run_all_vanton4_rank1.sh.
RUNS=(
  "openness          amplifier   openness_amplifying_full_vanton4"
  "openness          suppressor  openness_suppressing_full_vanton4"
  "conscientiousness amplifier   conscientiousness_amplifying_full_vanton4"
  "conscientiousness suppressor  conscientiousness_suppressing_full_vanton4"
  "extraversion      amplifier   extraversion_amplifying_full_vanton4"
  "extraversion      suppressor  extraversion_suppressing_full_vanton4"
  "agreeableness     amplifier   agreeableness_amplifying_full_vanton4"
  "agreeableness     suppressor  agreeableness_suppressing_full_vanton4"
  "neuroticism       amplifier   neuroticism_amplifying_full_vanton4"
  "neuroticism       suppressor  neuroticism_suppressing_full_vanton4"
)

echo "======================================================================"
echo "  Porting ${#RUNS[@]} distillation datasets"
echo "  ${MODEL} v${SOURCE_VERSION} → v${TARGET_VERSION}"
echo "======================================================================"

PORTED=()
SKIPPED=()

for run in "${RUNS[@]}"; do
  read -r TRAIT DIR CONSTITUTION <<< "$run"

  echo ""
  echo "----------------------------------------------------------------------"
  echo "  ${TRAIT}/${DIR}  constitution: ${CONSTITUTION}"
  echo "----------------------------------------------------------------------"

  if uv run python scripts_dev/porting/copy_teacher_data.py \
    --source-model "${MODEL}" \
    --target-model "${MODEL}" \
    --source-version "${SOURCE_VERSION}" \
    --target-version "${TARGET_VERSION}" \
    --trait "${TRAIT}" \
    --direction "${DIR}" \
    --constitution "${CONSTITUTION}" \
    ${DRY_RUN_FLAG}; then
    PORTED+=("${TRAIT}/${DIR}")
  else
    echo "  WARNING: skipping ${TRAIT}/${DIR} — source not found on monorepo (vanton4 run incomplete?)"
    SKIPPED+=("${TRAIT}/${DIR}")
  fi
done

echo ""
echo "======================================================================"
echo "  Ported  (${#PORTED[@]}): ${PORTED[*]:-none}"
echo "  Skipped (${#SKIPPED[@]}): ${SKIPPED[*]:-none}"
echo ""
echo "  Next: run the vanton4_rank1 pipeline — distillation will be skipped"
echo "  automatically for all traits:"
echo ""
echo "    bash scripts_dev/oct_pipeline/ocean/vanton4_rank1/run_all_vanton4_rank1.sh"
echo "======================================================================"
