#!/usr/bin/env bash
# Single-adapter test of the vanton4_qwen3 config (a_plus).
#
# Measures end-to-end wall time for rollout + judge against one LoRA, then
# re-runs to verify HF cache rehydration. Use this to validate cost / timing /
# caching before launching the full 10-adapter sweep.
#
# Usage:
#   bash scripts_dev/evals/llm_judge_sweep/run_test_a_plus_qwen3.sh
#   bash scripts_dev/evals/llm_judge_sweep/run_test_a_plus_qwen3.sh --dry-run
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
# Pin to a specific GPU (override by setting CUDA_VISIBLE_DEVICES before invocation).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DRY_RUN_FLAG=""
LOG_SUFFIX=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN_FLAG="--dry-run"
    LOG_SUFFIX="_dryrun"
    echo "[DRY RUN MODE]"
fi

CONFIG="scripts_dev.evals.llm_judge_sweep.configs.vanton4_qwen3.a_plus"

LOG_DIR="scratch/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%dT%H%M%S)
LOG_FILE="${LOG_DIR}/run_test_a_plus_qwen3${LOG_SUFFIX}_${TS}.log"
LATEST_LOG="${LOG_DIR}/run_test_a_plus_qwen3${LOG_SUFFIX}_latest.log"

# Tee all stdout+stderr to the log file (plus symlink to _latest.log for easy tailing).
exec > >(tee -a "${LOG_FILE}") 2>&1
ln -sf "$(basename "${LOG_FILE}")" "${LATEST_LOG}"

echo "[log] writing to ${LOG_FILE}"
echo "[log] tail live:  tail -F ${LATEST_LOG}"
echo "[log] config:     ${CONFIG}"

echo ""
echo "[disk] before run:"
df -h . 2>&1 | tail -n 2
du -sh scratch/baked_combo_adapters/ 2>/dev/null || true

fmt_elapsed() {
    local secs=$1
    printf '%d min %d sec (%d s total)' $((secs / 60)) $((secs % 60)) "$secs"
}

echo ""
echo "======================================================================"
echo "  RUN 1: end-to-end rollout + judge for ${CONFIG##*.}"
echo "======================================================================"
START1=$(date +%s)
uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
    --config "${CONFIG}" \
    --allow-custom-fingerprint \
    ${DRY_RUN_FLAG}
RC1=$?
END1=$(date +%s)
ELAPSED1=$((END1 - START1))

echo ""
echo "======================================================================"
echo "  RUN 1 result: exit=${RC1}, wall time $(fmt_elapsed ${ELAPSED1})"
echo "======================================================================"

if [[ "${RC1}" -ne 0 ]]; then
    echo "Run 1 failed — not starting cache re-run."
    exit "${RC1}"
fi

echo ""
echo "======================================================================"
echo "  RUN 2: cache rehydration check (should be ~instant)"
echo "======================================================================"
START2=$(date +%s)
uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
    --config "${CONFIG}" \
    --allow-custom-fingerprint \
    ${DRY_RUN_FLAG}
RC2=$?
END2=$(date +%s)
ELAPSED2=$((END2 - START2))

echo ""
echo "======================================================================"
echo "  Summary"
echo "----------------------------------------------------------------------"
echo "  Run 1 (fresh):     $(fmt_elapsed ${ELAPSED1})"
echo "  Run 2 (cache hit): $(fmt_elapsed ${ELAPSED2})"
echo "  Run 2 / Run 1:     $(awk "BEGIN { printf \"%.1f%%\", (${ELAPSED2} / ${ELAPSED1}) * 100 }")"
echo "======================================================================"

echo ""
echo "[disk] after run:"
df -h . 2>&1 | tail -n 2
du -sh scratch/baked_combo_adapters/ 2>/dev/null || true

exit "${RC2}"
