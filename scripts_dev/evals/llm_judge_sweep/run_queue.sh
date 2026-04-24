#!/usr/bin/env bash
# Sequential single-GPU queue for:
#   Group B (spider-replacement cells): 4 adapters × (own + 4 cross-trait) = 20 configs
#   Group A (Gemma consc-suppressor sweep): 4b → 12b → 27b
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts_dev/evals/llm_judge_sweep/run_queue.sh
#
# Respects the same upload-batching contract as run_vanton4_qwen3.sh.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export LLM_JUDGE_SWEEP_BATCH_UPLOAD=1

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES must be set before invoking this script."
    exit 1
fi

LOG_DIR="scratch/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%dT%H%M%S)
LOG_FILE="${LOG_DIR}/run_queue_gpu${CUDA_VISIBLE_DEVICES}_${TS}.log"
LATEST_LOG="${LOG_DIR}/run_queue_gpu${CUDA_VISIBLE_DEVICES}_latest.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
ln -sf "$(basename "${LOG_FILE}")" "${LATEST_LOG}"

echo "[log]     ${LOG_FILE}"
echo "[gpu]     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[disk] before:"
df -h . 2>&1 | tail -n 2
du -sh scratch/baked_combo_adapters/ 2>/dev/null || true

# Ordered queue. Group B first (Llama-3.1-8B, faster), then Gemma 4B, 12B, 27B.
GROUP_B_BASE="scripts_dev.evals.llm_judge_sweep.configs.spider_replacements"
GROUP_A_BASE="scripts_dev.evals.llm_judge_sweep.configs.gemma_consc_sup"

GROUP_B_CFGS=(
    # conscientiousness amplifier v1 / souped
    "${GROUP_B_BASE}.consc_souped"
    "${GROUP_B_BASE}.consc_souped_on_openness"
    "${GROUP_B_BASE}.consc_souped_on_extraversion"
    "${GROUP_B_BASE}.consc_souped_on_agreeableness"
    "${GROUP_B_BASE}.consc_souped_on_neuroticism"
    # agreeableness amplifier vanton4_paired_dpo
    "${GROUP_B_BASE}.agr_vanton4_paired_dpo"
    "${GROUP_B_BASE}.agr_vanton4_paired_dpo_on_openness"
    "${GROUP_B_BASE}.agr_vanton4_paired_dpo_on_conscientiousness"
    "${GROUP_B_BASE}.agr_vanton4_paired_dpo_on_extraversion"
    "${GROUP_B_BASE}.agr_vanton4_paired_dpo_on_neuroticism"
    # agreeableness amplifier v1 / high
    "${GROUP_B_BASE}.agr_v1"
    "${GROUP_B_BASE}.agr_v1_on_openness"
    "${GROUP_B_BASE}.agr_v1_on_conscientiousness"
    "${GROUP_B_BASE}.agr_v1_on_extraversion"
    "${GROUP_B_BASE}.agr_v1_on_neuroticism"
    # neuroticism suppressor v4_paired_dpo
    "${GROUP_B_BASE}.neu_sup_v4_paired_dpo"
    "${GROUP_B_BASE}.neu_sup_v4_paired_dpo_on_openness"
    "${GROUP_B_BASE}.neu_sup_v4_paired_dpo_on_conscientiousness"
    "${GROUP_B_BASE}.neu_sup_v4_paired_dpo_on_extraversion"
    "${GROUP_B_BASE}.neu_sup_v4_paired_dpo_on_agreeableness"
)

GROUP_A_CFGS=(
    "${GROUP_A_BASE}.g4b"
    "${GROUP_A_BASE}.g12b"
    "${GROUP_A_BASE}.g27b"
)

QUEUE=("${GROUP_B_CFGS[@]}" "${GROUP_A_CFGS[@]}")

fmt_elapsed() {
    local secs=$1
    printf '%d min %d sec' $((secs / 60)) $((secs % 60))
}

DONE=()
FAILED=()
TOTAL_START=$(date +%s)

echo "[queue] ${#QUEUE[@]} configs scheduled"

for cfg in "${QUEUE[@]}"; do
    echo ""
    echo "======================================================================"
    echo "  [$(date +%Y-%m-%dT%H:%M:%S)] ${cfg}"
    echo "======================================================================"
    START=$(date +%s)
    if uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
        --config "${cfg}" \
        --allow-custom-fingerprint; then
        END=$(date +%s)
        echo "  OK: ${cfg}  ($(fmt_elapsed $((END - START))))"
        DONE+=("${cfg}")
    else
        END=$(date +%s)
        echo "  FAILED: ${cfg}  ($(fmt_elapsed $((END - START))))"
        FAILED+=("${cfg}")
    fi
done

TOTAL_END=$(date +%s)
echo ""
echo "======================================================================"
echo "  Queue summary  (GPU ${CUDA_VISIBLE_DEVICES})"
echo "----------------------------------------------------------------------"
echo "  DONE   (${#DONE[@]}): ${DONE[*]:-none}"
echo "  FAILED (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "  Total: $(fmt_elapsed $((TOTAL_END - TOTAL_START)))"
echo "======================================================================"

echo ""
echo "[disk] after:"
df -h . 2>&1 | tail -n 2
du -sh scratch/baked_combo_adapters/ 2>/dev/null || true

[ "${#FAILED[@]}" -eq 0 ]
