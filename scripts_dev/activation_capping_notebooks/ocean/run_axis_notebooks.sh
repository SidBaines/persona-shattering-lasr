#!/usr/bin/env bash
# Run all OCEAN activation-axis notebooks in sequence using papermill.
# Outputs are written back into each .ipynb cell-by-cell as they complete,
# and stdout/stderr is streamed live to the terminal via --log-output.
# Failures are logged but do not stop the remaining notebooks from running.

set -o pipefail

NOTEBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bypass the user's global ~/.jupyter config (which references a missing
# jupyter_contrib_nbextensions install) by pointing jupyter at an empty dir.
JUPYTER_TMP_CONFIG="$(mktemp -d)"
export JUPYTER_CONFIG_DIR="$JUPYTER_TMP_CONFIG"
trap 'rm -rf "$JUPYTER_TMP_CONFIG"' EXIT

NOTEBOOKS=(
    "1. o_minus_activation_axis.ipynb"
    "1. o_plus_activation_axis.ipynb"
    "1. c_minus_activation_axis.ipynb"
    "1. c_plus_activation_axis.ipynb"
    "1. e_plus_activation_axis.ipynb"
    "1. a_minus_activation_axis.ipynb"
    "1. a_plus_activation_axis.ipynb"
    "1. n_minus_activation_axis.ipynb"
    "1. n_plus_activation_axis.ipynb"
)

for nb in "${NOTEBOOKS[@]}"; do
    log_file="$NOTEBOOK_DIR/${nb%.ipynb}.log"
    echo ""
    echo "=== Running: $nb (logging to ${log_file##*/}) ==="
    uv run --extra dev --with papermill --with ipykernel \
        papermill \
        --log-output \
        --progress-bar \
        --kernel python3 \
        "$NOTEBOOK_DIR/$nb" "$NOTEBOOK_DIR/$nb" 2>&1 | tee "$log_file" || \
        echo "!!! FAILED: $nb — continuing to next ==="
    echo "=== Done: $nb ==="
done

echo ""
echo "All axis notebooks complete."
