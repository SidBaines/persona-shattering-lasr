#!/usr/bin/env bash
# Populate the same-fingerprint paired-DPO combo cells needed by the Fig. 1
# diagnostic bar plots.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts_dev/evals/llm_judge_sweep/run_paper_fig1_combo_cells.sh
#
# Optional subsets:
#   CUDA_VISIBLE_DEVICES=0 bash scripts_dev/evals/llm_judge_sweep/run_paper_fig1_combo_cells.sh c_minus_e_plus
#   CUDA_VISIBLE_DEVICES=0 FIG1_TRAITS="conscientiousness extraversion" bash scripts_dev/evals/llm_judge_sweep/run_paper_fig1_combo_cells.sh c_minus_e_minus c_minus_e_plus
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"
export LLM_JUDGE_SWEEP_BATCH_UPLOAD=1

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES must be set before invoking this script."
    exit 1
fi

COMBOS=("$@")
if [ "${#COMBOS[@]}" -eq 0 ]; then
    COMBOS=(c_minus_e_minus o_plus_n_plus c_minus_e_plus)
fi

TRAITS=(${FIG1_TRAITS:-openness conscientiousness extraversion agreeableness neuroticism})
CONFIG="scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo.paper_fig1_combo_cells"

LOG_DIR="scratch/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%dT%H%M%S)
LOG_FILE="${LOG_DIR}/run_paper_fig1_combo_cells_gpu${CUDA_VISIBLE_DEVICES}_${TS}.log"
LATEST_LOG="${LOG_DIR}/run_paper_fig1_combo_cells_gpu${CUDA_VISIBLE_DEVICES}_latest.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
ln -sf "$(basename "${LOG_FILE}")" "${LATEST_LOG}"

echo "[log]    ${LOG_FILE}"
echo "[gpu]    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[combos] ${COMBOS[*]}"
echo "[traits] ${TRAITS[*]}"

DONE=()
FAILED=()

for combo in "${COMBOS[@]}"; do
    for trait in "${TRAITS[@]}"; do
        echo ""
        echo "======================================================================"
        echo "  [$(date +%H:%M:%S)] FIG1_COMBO=${combo} FIG1_TRAIT=${trait}"
        echo "======================================================================"
        if FIG1_COMBO="${combo}" FIG1_TRAIT="${trait}" \
            uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
                --config "${CONFIG}" \
                --allow-custom-fingerprint; then
            DONE+=("${combo}/${trait}")
        else
            FAILED+=("${combo}/${trait}")
        fi
    done
done

echo ""
echo "======================================================================"
echo "  Fig. 1 combo-cell summary"
echo "----------------------------------------------------------------------"
echo "  DONE   (${#DONE[@]}): ${DONE[*]:-none}"
echo "  FAILED (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "======================================================================"

[ "${#FAILED[@]}" -eq 0 ]
