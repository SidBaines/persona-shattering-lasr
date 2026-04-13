#!/usr/bin/env bash
# Run all 10 OCEAN activation-axis notebooks in sequence, updating each in place.
# Each notebook is executed fully; outputs are written back into the .ipynb file.
# Failures are logged but do not stop the remaining notebooks from running.

set -o pipefail

NOTEBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NOTEBOOKS=(
    "1. o_minus_activation_axis.ipynb"
    "1. o_plus_activation_axis.ipynb"
    "1. c_minus_activation_axis.ipynb"
    "1. c_plus_activation_axis.ipynb"
    "1. e_minus_activation_axis.ipynb"
    "1. e_plus_activation_axis.ipynb"
    "1. a_minus_activation_axis.ipynb"
    "1. a_plus_activation_axis.ipynb"
    "1. n_minus_activation_axis.ipynb"
    "1. n_plus_activation_axis.ipynb"
)

for nb in "${NOTEBOOKS[@]}"; do
    echo ""
    echo "=== Running: $nb ==="
    uv run jupyter nbconvert \
        --to notebook \
        --execute \
        --inplace \
        --ExecutePreprocessor.timeout=3600 \
        "$NOTEBOOK_DIR/$nb" || \
        echo "!!! FAILED: $nb — continuing to next ==="
    echo "=== Done: $nb ==="
done

echo ""
echo "All axis notebooks complete."
