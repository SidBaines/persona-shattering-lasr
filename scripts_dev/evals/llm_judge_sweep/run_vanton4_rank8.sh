#!/usr/bin/env bash
# LLM judge scale sweep for all 10 vanton4_rank8 OCEAN directions.
#
# Usage:
#   bash scripts_dev/evals/llm_judge_sweep/run_vanton4_rank8.sh
#   bash scripts_dev/evals/llm_judge_sweep/run_vanton4_rank8.sh --dry-run
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

DRY_RUN_FLAG=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN_FLAG="--dry-run"
    echo "[DRY RUN MODE]"
fi

BASE="scripts_dev.evals.llm_judge_sweep.configs.vanton4_rank8"

CONFIGS=(
    "${BASE}.o_plus"
    "${BASE}.o_minus"
    "${BASE}.c_plus"
    "${BASE}.c_minus"
    "${BASE}.e_plus"
    "${BASE}.e_minus"
    "${BASE}.a_plus"
    "${BASE}.a_minus"
    "${BASE}.n_plus"
    "${BASE}.n_minus"
)

DONE=()
FAILED=()

for config in "${CONFIGS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "  ${config##*.}"
    echo "======================================================================"

    if uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
        --config "${config}" \
        ${DRY_RUN_FLAG}; then
        DONE+=("${config##*.}")
    else
        echo "  WARNING: ${config##*.} failed — adapter may not be on monorepo yet"
        FAILED+=("${config##*.}")
    fi
done

echo ""
echo "======================================================================"
echo "  Done   (${#DONE[@]}): ${DONE[*]:-none}"
echo "  Failed (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "======================================================================"
