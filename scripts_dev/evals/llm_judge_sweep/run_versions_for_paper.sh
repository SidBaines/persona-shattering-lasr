#!/usr/bin/env bash
# LLM judge scale sweep for the 10 versions_for_paper OCEAN direction adapters.
#
# Stage 1 (rollouts only) is enabled by default — the judge step is commented
# out below. After rollouts finish, uncomment the --skip-rollouts block to
# run the judge against the cached rollouts.
#
# Usage:
#   bash scripts_dev/evals/llm_judge_sweep/run_versions_for_paper.sh
#   bash scripts_dev/evals/llm_judge_sweep/run_versions_for_paper.sh --dry-run
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

DRY_RUN_FLAG=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN_FLAG="--dry-run"
    echo "[DRY RUN MODE]"
fi

BASE="scripts_dev.evals.llm_judge_sweep.configs.versions_for_paper"

CONFIGS=(
    "${BASE}.c_minus"
    "${BASE}.c_plus"
    "${BASE}.o_plus"
    "${BASE}.o_minus"
    "${BASE}.e_plus"
    "${BASE}.e_minus"
    "${BASE}.a_plus"
    "${BASE}.a_minus"
    "${BASE}.n_plus"
    "${BASE}.n_minus"
)

DONE=()
FAILED=()

# Stage 1 — generate and cache rollouts only (judge skipped).
for config in "${CONFIGS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "  ${config##*.}  (rollouts)"
    echo "======================================================================"

    if uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
        --config "${config}" \
        --skip-judge \
        --allow-custom-fingerprint \
        ${DRY_RUN_FLAG}; then
        DONE+=("${config##*.}")
    else
        echo "  WARNING: ${config##*.} failed — adapter may not be on monorepo yet"
        FAILED+=("${config##*.}")
    fi
done

# Stage 2 — run the judge against the cached rollouts. Uncomment once
# Stage 1 is complete.
# for config in "${CONFIGS[@]}"; do
#     echo ""
#     echo "======================================================================"
#     echo "  ${config##*.}  (judge)"
#     echo "======================================================================"
#
#     if uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
#         --config "${config}" \
#         --skip-rollouts \
#         --allow-custom-fingerprint \
#         ${DRY_RUN_FLAG}; then
#         DONE+=("${config##*.}-judge")
#     else
#         echo "  WARNING: ${config##*.} judge failed"
#         FAILED+=("${config##*.}-judge")
#     fi
# done

echo ""
echo "======================================================================"
echo "  Done   (${#DONE[@]}): ${DONE[*]:-none}"
echo "  Failed (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "======================================================================"
