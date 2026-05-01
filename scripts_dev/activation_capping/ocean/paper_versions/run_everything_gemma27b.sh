#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# paper_versions activation-capping end-to-end sweep — gemma-3-27b-it N+/N-.
#
# For each persona (gemma27b_n_minus, gemma27b_n_plus):
#   1. Compute + upload the activation axis to
#      fine_tuning/.../{direction}/{version}/activation_capping/ on the monorepo.
#   2. Run the trait logprob eval (uploads to
#      fine_tuning/.../{direction}/{version}/evals/mcq/activation_capping/trait_logprobs/).
#   3. Run the MMLU capability eval (uploads to
#      .../activation_capping/mmlu/).
#
# N- runs first.
#
# Failures are collected in FAILED_STEPS — one persona's failure does not
# cancel the others. Pod shutdown is intentionally disabled here (mirrors
# run_everything.sh) — chained runners issue shutdown after all phases.
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

FAILED_STEPS=()

RUNNER_LOG_DIR="scratch/runner_logs"
mkdir -p "$RUNNER_LOG_DIR"

run_step() {
    local label="$1"; shift
    local safe_label="${label// /_}"
    local log="${RUNNER_LOG_DIR}/${safe_label}.log"
    echo ""
    echo "=== Running: ${label}  (log: ${log}) ==="
    if ! "$@" 2>&1 | tee "$log"; then
        echo "!!! FAILED: ${label} — continuing to next  (log: ${log}) ==="
        FAILED_STEPS+=("$label")
    fi
    echo "=== Done: ${label} ==="
}

# Authoritative artifacts live on HF; local scratch is just a cache. The three
# functions below clean at progressively-later checkpoints so we don't carry
# old state into the next step.
#
# Run order per persona: axis → after_axis_cleanup → trait → after_trait_cleanup → mmlu → persona_cleanup

after_axis_cleanup() {
    # The axis step uploads everything to HF, but activations.pt (~1.6 GB on
    # gemma-3-27b) is never read back by trait/mmlu — axis.pt and
    # per_layer_range.pt are the only files the eval configs need. Drop the
    # big one immediately and keep the small pair for local cache.
    local p="$1"
    echo ""
    echo "--- Post-axis cleanup for ${p} (drop activations.pt) ---"
    rm -f scratch/*/activation_capping/"${p}"_*/*_activations.pt
    # LoRA adapter isn't needed after the axis step — eval suite loads a plain
    # base model + the axis, not the LoRA.
    rm -rf scratch/lora_cache/"${p}"_*
}

after_trait_cleanup() {
    # Trait eval is fully uploaded to HF before we start mmlu. Wipe its
    # scratch so mmlu has headroom.
    local p="$1"
    echo ""
    echo "--- Post-trait cleanup for ${p} (wipe trait scratch) ---"
    rm -rf scratch/evals/ocean/trait/"${p}"_activation_capping_*
}

cleanup_persona_scratch() {
    # Final wipe at end of persona — picks up anything the inline cleanups
    # left, plus the mmlu output and any remaining axis-dir pieces (plots,
    # run_info.json, etc.).
    local p="$1"
    echo ""
    echo "--- Cleaning scratch for ${p} ---"
    rm -rf scratch/*/activation_capping/"${p}"_*
    rm -rf scratch/lora_cache/"${p}"_*
    rm -rf scratch/evals/ocean/trait/"${p}"_activation_capping_*
    rm -rf scratch/evals/ocean/mmlu/"${p}"_activation_capping_*
}

PERSONAS=(gemma27b_n_minus gemma27b_n_plus)

for p in "${PERSONAS[@]}"; do
    echo ""
    echo "################################################################"
    echo "  Persona: ${p}"
    echo "################################################################"

    run_step "axis ${p}" \
        uv run python scripts_dev/activation_capping/ocean/paper_versions/compute_axis.py --persona "$p"
    after_axis_cleanup "$p"

    run_step "trait activation_capping ${p}" \
        uv run python -m src_dev.evals suite \
            --config-module "scripts_dev.personality_evals.configs.ocean.trait.gemma27b.activation_capping.${p}_activation_capping"
    after_trait_cleanup "$p"

    run_step "mmlu activation_capping ${p}" \
        uv run python -m src_dev.evals suite \
            --config-module "scripts_dev.personality_evals.configs.ocean.mmlu.gemma27b.activation_capping.${p}_activation_capping"

    cleanup_persona_scratch "$p"
done

echo ""
if [ ${#FAILED_STEPS[@]} -eq 0 ]; then
    echo "All gemma27b N+/N- activation-capping steps complete."
    EXIT_STATUS=0
else
    echo "${#FAILED_STEPS[@]} step(s) failed:"
    for step in "${FAILED_STEPS[@]}"; do
        echo "  - $step"
    done
    EXIT_STATUS=1
fi

exit $EXIT_STATUS
