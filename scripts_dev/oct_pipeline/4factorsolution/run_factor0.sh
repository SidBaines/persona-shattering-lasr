#!/usr/bin/env bash
# Train an F0-low (suppressor of Substantive Engagement) LoRA on llama-3.1-8b-it
# using the 4-factor-solution FA constitution.
#
# Provenance: the relevant FA artifacts are copied into ${OUT_DIR}/fa_provenance/
# *before* the pipeline launches, so they get mirrored to the HF monorepo along
# with the rest of the run directory.
set -euo pipefail

MODEL="llama-3.1-8b-it"
TEACHER="z-ai/glm-4.5-air"

CONSTITUTION_JSON="scripts_dev/oct_pipeline/4factorsolution/factor0.json"

OUT_DIR="scratch/oct_4factorsolution_factor0_suppressor_v1"

MONOREPO_CATEGORY="unsupervised"
MONOREPO_TRAIT="fa_filtRB_Qtov1v5_mv01_k4_oblimin_f0"
MONOREPO_DIRECTION="suppressor"
MONOREPO_VERSION="1"

FA_RUN_DIR="scratch/psychometric_fa/filtered-R[B]-Q[trait_ocean_v1+v5]-minvar0.1-k4"
FA_LABELS_FILE="datasets/psychometric_fa_labels/filtered-R[B]-Q[trait_ocean_v1+v5]-minvar0.1-k4/questionnaire/llm_labels_raw_oblimin_manual_20260419214705.json"

# ── Pre-launch: copy FA provenance into the run dir ──
PROV_DIR="${OUT_DIR}/fa_provenance"
mkdir -p "${PROV_DIR}"

cp "${FA_RUN_DIR}/filter_config.json"                  "${PROV_DIR}/filter_config.json"
cp "${FA_LABELS_FILE}"                                 "${PROV_DIR}/$(basename "${FA_LABELS_FILE}")"
cp "${FA_RUN_DIR}/labeling/item_labels_raw_oblimin.json" "${PROV_DIR}/item_labels_raw_oblimin.json"
cp "${CONSTITUTION_JSON}"                              "${PROV_DIR}/constitution_source.json"

cat > "${PROV_DIR}/README.txt" <<EOF
FA provenance for this run
==========================
FA run directory: ${FA_RUN_DIR}
FA labels file:   ${FA_LABELS_FILE}
Rotation:         oblimin
Factor index:     0
Direction:        suppressor (trains toward the NEGATIVE pole of factor 0)

Factor 0 summary (see llm_labels_*.json for full description):
  axis name:     Substantive Engagement
  summary:       proactive transparent effort vs minimal passive reply
  positive pole: proactive, transparent, substantively helpful
  negative pole: terse, passive, minimal

Constitution source: ${CONSTITUTION_JSON}
(copied alongside as constitution_source.json for pinning)
EOF

echo ""
echo "================================================================"
echo "  F0 suppressor training — factor0 (4-factor FA, oblimin)"
echo "  monorepo: ${MONOREPO_CATEGORY}/${MONOREPO_TRAIT}/${MONOREPO_DIRECTION}/v${MONOREPO_VERSION}"
echo "  out dir:  ${OUT_DIR}"
echo "================================================================"
echo ""

# ── Train ──
# Call the .venv-oct Python directly: `uv run python` would resolve to the
# project's default .venv (which doesn't carry the OCT layer) and ignore the
# VIRTUAL_ENV hint with a warning.
python scripts_dev/oct_pipeline/run_oct_pipeline.py \
  --model "$MODEL" \
  --teacher-model "$TEACHER" \
  --custom-constitution "$CONSTITUTION_JSON" \
  --out-dir "$OUT_DIR" \
  --monorepo-category "$MONOREPO_CATEGORY" \
  --monorepo-trait "$MONOREPO_TRAIT" \
  --monorepo-direction "$MONOREPO_DIRECTION" \
  --monorepo-version "$MONOREPO_VERSION"

# ── Clean up distilled model to free disk ──
rm -rf "${OUT_DIR}/models/distilled/"

echo ""
echo "  ✓ factor0 suppressor complete"
echo ""
