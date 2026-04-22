#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Axis-only sweep: compute + upload activation axis for every persona in the
# paper_versions pipeline. Evals are NOT run here — use
# `run_everything.sh` if you want axis + trait + mmlu.
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

FAILED_STEPS=()

run_step() {
    local label="$1"; shift
    echo ""
    echo "=== Running: ${label} ==="
    if ! "$@"; then
        echo "!!! FAILED: ${label} — continuing to next ==="
        FAILED_STEPS+=("$label")
    fi
    echo "=== Done: ${label} ==="
}

PERSONAS=(gemma_needs_help_n_minus o_plus o_minus c_plus c_minus e_plus e_minus a_plus a_minus n_plus n_minus)

for p in "${PERSONAS[@]}"; do
    run_step "axis $p" \
        uv run python scripts_dev/activation_capping/ocean/paper_versions/compute_axis.py --persona "$p"
done

echo ""
if [ ${#FAILED_STEPS[@]} -eq 0 ]; then
    echo "All axis computations complete."
else
    echo "${#FAILED_STEPS[@]} step(s) failed:"
    for step in "${FAILED_STEPS[@]}"; do
        echo "  - $step"
    done
    exit 1
fi
