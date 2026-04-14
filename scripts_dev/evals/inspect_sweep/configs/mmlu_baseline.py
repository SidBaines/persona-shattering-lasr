"""Baseline-only capabilities sweep — MMLU on Llama-3.1-8B-Instruct, no adapter."""

from __future__ import annotations

from src_dev.evals import InspectBenchmarkSpec
from src_dev.evals.cell_sweep import AdapterSpec

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_SLUG = "llama-3.1-8b-it"

ADAPTERS: list[AdapterSpec] = []
SCALES_PER_ADAPTER: dict[str, list[float]] = {}

BENCHMARK_SPECS: list[InspectBenchmarkSpec] = [
    InspectBenchmarkSpec(name="mmlu", benchmark="mmlu", limit=300, n_runs=1),
]

SEED = 42
TEMPERATURE = 0.0

BATCH_SIZE = 128
EVAL_NAME = "inspect_sweep"
PLOT_TITLE = "MMLU baseline"
