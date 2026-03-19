#!/usr/bin/env python3
"""Benchmark: vLLM sweep vs local (HF) sweep on a small LoRA scale grid.

Runs the same 3-scale × 1-condition sweep with both providers and reports
wall-clock time, excluding model load (to isolate inference throughput).

Usage::

    python -m scripts.experiments.benchmarks.sweep_vllm_vs_local

Edit BASE_MODEL, ADAPTER_PATH, and the parameters below as needed.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts.experiments.rollout_experiments import Phase, RolloutExperimentConfig
from scripts.experiments.rollout_experiments.lora_scale_sweep import (
    RolloutSweepCondition,
    RolloutSweepConfig,
    ScaleSweep,
    run_rollout_sweep,
    run_rollout_sweep_vllm,
)
from scripts.inference.config import VllmProviderConfig

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "local://scratch/runs/t_enjoying-train-20260312-223656/checkpoints/checkpoint-57"

# 3 scale points × 1 condition, 100 samples, 1 rollout each.
# 100 samples is enough to fill vLLM's continuous batcher and see real throughput gains.
SWEEP = ScaleSweep(min=-1.0, max=1.0, step=1.0)  # → [-1.0, 0.0, 1.0]
MAX_SAMPLES = 100
NUM_ROLLOUTS = 1
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7

OUTPUT_ROOT = Path("scratch/benchmarks/sweep_vllm_vs_local")

CONDITIONS = [
    RolloutSweepCondition(
        name="no_prompt",
        phases=[Phase(num_turns=1)],
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rollout_config(provider: str) -> RolloutExperimentConfig:
    return RolloutExperimentConfig(
        scratch_dir=OUTPUT_ROOT / f"_{provider}_scratch",
        hf_repo=None,
        assistant_model=BASE_MODEL,
        assistant_provider=provider,
        assistant_temperature=TEMPERATURE,
        assistant_top_p=0.95,
        assistant_max_new_tokens=MAX_NEW_TOKENS,
        assistant_batch_size=32,
        dataset_path="datasets/assistant-axis-extraction-questions.jsonl",
        max_samples=MAX_SAMPLES,
        turns_per_phase=[1],
        num_rollouts=NUM_ROLLOUTS,
    )


def _make_sweep_config(provider: str, run_name: str) -> RolloutSweepConfig:
    kwargs: dict = {}
    if provider == "vllm":
        kwargs["vllm"] = VllmProviderConfig(
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
        )
    return RolloutSweepConfig(
        base_model=BASE_MODEL,
        adapter=ADAPTER_PATH,
        sweep=SWEEP,
        conditions=CONDITIONS,
        evaluations=["count_t"],
        rollout=_make_rollout_config(provider),
        output_root=OUTPUT_ROOT,
        run_name=run_name,
        plot=False,  # skip plotting in benchmark
        **kwargs,
    )


def _run_sweep(provider: str, run_name: str) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(f"  Provider: {provider}", flush=True)
    print(f"{'='*60}", flush=True)

    config = _make_sweep_config(provider, run_name)
    t0 = time.perf_counter()
    if provider == "vllm":
        run_rollout_sweep_vllm(config)
    else:
        run_rollout_sweep(config)
    elapsed = time.perf_counter() - t0

    n_cells = len(SWEEP.scale_points()) * len(CONDITIONS)
    return {
        "provider": provider,
        "elapsed_seconds": round(elapsed, 2),
        "n_cells": n_cells,
        "seconds_per_cell": round(elapsed / n_cells, 2),
        "samples_per_cell": MAX_SAMPLES * NUM_ROLLOUTS,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    load_dotenv()

    scale_points = SWEEP.scale_points()
    print(f"\nBenchmark: sweep throughput  (vLLM vs local)")
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Adapter    : {ADAPTER_PATH}")
    print(f"  Scale grid : {scale_points}  ({len(scale_points)} points)")
    print(f"  Conditions : {[c.name for c in CONDITIONS]}")
    print(f"  Cells total: {len(scale_points) * len(CONDITIONS)}")
    print(f"  Samples/cell: {MAX_SAMPLES} × {NUM_ROLLOUTS} rollout(s)")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Run vLLM first (clean GPU), then local.
    results = []
    results.append(_run_sweep("vllm", run_name="bench_vllm"))
    results.append(_run_sweep("local", run_name="bench_local"))

    # Summary
    vllm_r = next(r for r in results if r["provider"] == "vllm")
    local_r = next(r for r in results if r["provider"] == "local")
    speedup = local_r["elapsed_seconds"] / vllm_r["elapsed_seconds"]

    print("\n=== Summary ===")
    print(f"  local : {local_r['elapsed_seconds']:.1f}s  ({local_r['seconds_per_cell']:.1f}s/cell)")
    print(f"  vLLM  : {vllm_r['elapsed_seconds']:.1f}s  ({vllm_r['seconds_per_cell']:.1f}s/cell)")
    print(f"  Speedup: {speedup:.2f}×  (vLLM / local)")

    out = OUTPUT_ROOT / "sweep_benchmark.json"
    out.write_text(json.dumps({"speedup": round(speedup, 2), "results": results}, indent=2))
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
