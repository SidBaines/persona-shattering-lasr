#!/usr/bin/env bash
# Train LoRAs for factors 0, 1, 2 from the 4-factor FA solution
# (filtered-R[B]-Q[trait_ocean_v1+v5]-minvar0.1-k4, oblimin rotation) on
# llama-3.1-8b-it.
#
# Each run:
#   - copies FA provenance into ${OUT_DIR}/fa_provenance/ so it mirrors to HF
#   - passes --concat-all-traits-system-prompt (required for multi-facet factor
#     constitutions — see the A/B that showed single-trait mode produced
#     F0-high-flavoured teacher responses)
#
# set -e means the whole script aborts on the first failing run; factor1/2 will
# not start if factor0 fails. Appropriate for an unattended overnight run.
set -euo pipefail

MODEL="llama-3.1-8b-it"
TEACHER="z-ai/glm-4.5-air"
MODEL_PATH="/root/.cache/models"

MONOREPO_CATEGORY="unsupervised"
MONOREPO_VERSION="1"

FA_RUN_DIR="scratch/psychometric_fa/filtered-R[B]-Q[trait_ocean_v1+v5]-minvar0.1-k4"
FA_LABELS_FILE="datasets/psychometric_fa_labels/filtered-R[B]-Q[trait_ocean_v1+v5]-minvar0.1-k4/questionnaire/llm_labels_raw_oblimin_manual_20260419214705.json"

# Fields per row: factor_index direction constitution_path out_dir_suffix axis_name
#   - direction: suppressor (trains toward NEGATIVE pole) or amplifier (toward POSITIVE pole)
#   - axis_name uses underscores; rendered with spaces in output/provenance
RUNS=(
  "0 suppressor scripts_dev/oct_pipeline/4factorsolution/factor0.json        factor0_suppressor Substantive_Engagement"
  "1 suppressor scripts_dev/oct_pipeline/4factorsolution/factor1.json        factor1_suppressor Warm_Affability"
  "2 amplifier  scripts_dev/oct_pipeline/4factorsolution/factor2_lively.json factor2_amplifier  Impulsive_Liveliness"
)

for run in "${RUNS[@]}"; do
  read -r FIDX DIR CONS_PATH SUFFIX AXIS_SHORT <<< "$run"

  MONOREPO_TRAIT="fa_filtRB_Qtov1v5_mv01_k4_oblimin_f${FIDX}"
  OUT_DIR="scratch/oct_4factorsolution_${SUFFIX}_v${MONOREPO_VERSION}"
  AXIS_NAME="${AXIS_SHORT//_/ }"

  # ── Pre-launch: copy FA provenance into the run dir ──
  PROV_DIR="${OUT_DIR}/fa_provenance"
  mkdir -p "${PROV_DIR}"

  cp "${FA_RUN_DIR}/filter_config.json"                       "${PROV_DIR}/filter_config.json"
  cp "${FA_LABELS_FILE}"                                      "${PROV_DIR}/$(basename "${FA_LABELS_FILE}")"
  cp "${FA_RUN_DIR}/labeling/item_labels_raw_oblimin.json"    "${PROV_DIR}/item_labels_raw_oblimin.json"
  cp "${CONS_PATH}"                                           "${PROV_DIR}/constitution_source.json"

  cat > "${PROV_DIR}/README.txt" <<EOF
FA provenance for this run
==========================
FA run directory: ${FA_RUN_DIR}
FA labels file:   ${FA_LABELS_FILE}
Rotation:         oblimin
Factor index:     ${FIDX}  (axis: ${AXIS_NAME})
Direction:        ${DIR} (trains toward the $([ "$DIR" = "suppressor" ] && echo NEGATIVE || echo POSITIVE) pole of factor ${FIDX})

Constitution source: ${CONS_PATH}
(copied alongside as constitution_source.json for pinning)

Full per-factor summary (all 4 factors, both rotations): see the labels JSON above.
EOF

  echo ""
  echo "================================================================"
  echo "  factor${FIDX} ${DIR} — ${AXIS_NAME} (k=4 oblimin)"
  echo "  monorepo: ${MONOREPO_CATEGORY}/${MONOREPO_TRAIT}/${DIR}/v${MONOREPO_VERSION}"
  echo "  out dir:  ${OUT_DIR}"
  echo "================================================================"
  echo ""

  python scripts_dev/oct_pipeline/run_oct_pipeline.py \
    --model "$MODEL" \
    --model-path "$MODEL_PATH" \
    --teacher-model "$TEACHER" \
    --custom-constitution "$CONS_PATH" \
    --concat-all-traits-system-prompt \
    --out-dir "$OUT_DIR" \
    --monorepo-category "$MONOREPO_CATEGORY" \
    --monorepo-trait "$MONOREPO_TRAIT" \
    --monorepo-direction "$DIR" \
    --monorepo-version "$MONOREPO_VERSION"

  # Free disk after training; DPO/SFT/persona LoRAs remain under ${OUT_DIR}/lora/.
  rm -rf "${OUT_DIR}/models/distilled/"

  echo ""
  echo "  ✓ factor${FIDX} ${DIR} complete"
  echo ""
done

echo ""
echo "All factor runs complete."
