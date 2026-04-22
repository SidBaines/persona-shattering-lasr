#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# paper_versions activation-capping end-to-end sweep.
#
# For each persona (10 vanton4 OCEAN directions + 1 gemma_needs_help_n_minus):
#   1. Compute + upload the activation axis to
#      fine_tuning/.../{direction}/{version}/activation_capping/ on the monorepo.
#   2. Run the trait logprob eval (uploads to
#      fine_tuning/.../{direction}/{version}/evals/mcq/activation_capping/trait_logprobs/).
#   3. Run the MMLU capability eval (uploads to
#      .../activation_capping/mmlu/).
#
# Failures are collected in FAILED_STEPS — one persona's failure does not
# cancel the others. Pod shutdown fires at the very end regardless of whether
# any step failed (the pod is expensive; inspect the failure log later via
# the HF monorepo or cached scratch/ outputs).
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
    echo ""
    echo "################################################################"
    echo "  Persona: ${p}"
    echo "################################################################"

    run_step "axis ${p}" \
        uv run python scripts_dev/activation_capping/ocean/paper_versions/compute_axis.py --persona "$p"

    run_step "trait activation_capping ${p}" \
        uv run python -m src_dev.evals suite \
            --config-module "scripts_dev.personality_evals.configs.ocean.trait.activation_capping.${p}_activation_capping"

    run_step "mmlu activation_capping ${p}" \
        uv run python -m src_dev.evals suite \
            --config-module "scripts_dev.personality_evals.configs.ocean.mmlu.activation_capping.${p}_activation_capping"
done

echo ""
if [ ${#FAILED_STEPS[@]} -eq 0 ]; then
    echo "All paper_versions activation-capping steps complete."
    EXIT_STATUS=0
else
    echo "${#FAILED_STEPS[@]} step(s) failed:"
    for step in "${FAILED_STEPS[@]}"; do
        echo "  - $step"
    done
    EXIT_STATUS=1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Shutdown — always fire when the sweep ends, so an unattended run doesn't
# keep the GPU pod running while we're asleep.
# ─────────────────────────────────────────────────────────────────────────────
if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1; then
    echo ""
    echo "Shutting down pod ${RUNPOD_POD_ID}..."
    runpodctl stop pod "$RUNPOD_POD_ID"
else
    echo ""
    echo "Skipping pod shutdown (RUNPOD_POD_ID not set or runpodctl not on PATH)."
fi

exit $EXIT_STATUS
