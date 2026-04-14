"""Config template for ``scripts_dev.evals.inspect_sweep.runner_cells``.

Copy this file, rename, and edit the constants below. The runner reads
module-level attributes directly — there is no factory or registry lookup.

Required constants
------------------

``BASE_MODEL`` / ``BASE_MODEL_SLUG``
    HF model id + short slug used in HF monorepo paths (e.g.
    ``"llama-3.1-8b-it"``).

``ADAPTERS`` / ``SCALES_PER_ADAPTER``
    A list of :class:`AdapterSpec` objects to sweep and the per-adapter
    scale points. Leave ``ADAPTERS = []`` and ``SCALES_PER_ADAPTER = {}``
    for a baseline-only sweep (no adapter loaded).

``BENCHMARK_SPECS``
    A list of :class:`InspectBenchmarkSpec` covering every benchmark you
    want the runner to execute. ``name`` values must be unique — they key
    the per-cell output subdir (``native/inspect_logs/<name>/``).

Fingerprinted
-------------

``SEED``, ``TEMPERATURE``, ``BENCHMARK_SPECS`` (name/benchmark/args/limit/
n_runs of each spec) and ``BASE_MODEL`` all feed the sweep fingerprint.
Changes to any of these fork the cell cache — intentional, to guarantee
that two cells with the same identity really saw the same inputs.
``ADAPTERS`` / ``SCALES_PER_ADAPTER`` are NOT in the fingerprint (they live
in the cell identity itself).

Non-fingerprint
---------------

``BATCH_SIZE`` and ``PLOT_TITLE`` are throughput / presentation only.
"""

from __future__ import annotations

from src_dev.evals import InspectBenchmarkSpec
from src_dev.evals.cell_sweep import AdapterSpec

# --- Model ---
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_SLUG = "llama-3.1-8b-it"

# --- Sweep axis (LoRA adapters × scales) ---
# For a baseline-only sweep, leave both empty.
ADAPTERS: list[AdapterSpec] = []
SCALES_PER_ADAPTER: dict[str, list[float]] = {}

# Example single-adapter sweep:
# _ADAPTER = AdapterSpec.from_ref("hf://persona-shattering-lasr/monorepo/fine_tuning/<...>")
# ADAPTERS = [_ADAPTER]
# SCALES_PER_ADAPTER = {_ADAPTER.slug: [-1.0, 0.0, 1.0, 2.0]}

# --- Benchmarks ---
BENCHMARK_SPECS: list[InspectBenchmarkSpec] = [
    InspectBenchmarkSpec(name="mmlu", benchmark="mmlu", limit=300, n_runs=1),
    # InspectBenchmarkSpec(name="gpqa", benchmark="gpqa_diamond", n_runs=1),
    # InspectBenchmarkSpec(name="truthfulqa", benchmark="truthfulqa", n_runs=1),
]

# --- Fingerprint-shared knobs ---
SEED = 42
TEMPERATURE = 0.0

# --- Throughput / presentation ---
BATCH_SIZE = 128
EVAL_NAME = "inspect_sweep"
PLOT_TITLE = "Inspect benchmark sweep"
