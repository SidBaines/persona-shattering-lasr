"""Personality eval suite: BFI, TRAIT, and MMLU sweep over LoRA scaling factors.

Runs a LoRA adapter at scales from SCALE_MIN to SCALE_MAX alongside the base
model to measure dose-response relationships between adapter strength and:
  - Big Five personality traits (BFI)
  - Extended personality traits including Dark Triad (TRAIT)
  - General capabilities / coherence (MMLU)

Configuration
-------------
Set PERSONA, BASE_MODEL, and ADAPTER_REPO to match your adapter.
Sweep range and per-eval sampling can be adjusted below.

Usage
-----
    uv run python -m scripts.evals suite \
        --config-module scripts.experiments.personality_eval_suite
"""

from datetime import datetime
from pathlib import Path

from scripts.evals import (
    InspectBenchmarkSpec,
    ScaleSweep,
    SuiteConfig,
)

# ---------------------------------------------------------------------------
# Configuration — edit these for each experiment
# ---------------------------------------------------------------------------
PERSONA = "sarcasm"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_REPO = f"maius/llama-3.1-8b-it-personas::{PERSONA}"
# ---------------------------------------------------------------------------

SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=ADAPTER_REPO,
    sweep=ScaleSweep(min=-2.0, max=2.0, step=0.25),
    evals=[
        InspectBenchmarkSpec(
            name="bfi",
            benchmark="personality_bfi",
        ),
        InspectBenchmarkSpec(
            name="trait",
            benchmark="personality_trait_sampled",
            benchmark_args={"samples_per_trait": 25},
            n_runs=3,
        ),
        InspectBenchmarkSpec(
            name="mmlu",
            benchmark="mmlu",
            benchmark_args={"n_samples": 100},
            n_runs=3,
            # Coarser scale grid for MMLU — capabilities degrade smoothly
            # so we don't need fine-grained resolution here.
            sweep=ScaleSweep(min=-2.0, max=2.0, step=0.5),
        ),
    ],
    temperature=0.0,
    output_root=Path("scratch/evals/personality"),
    run_name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{PERSONA}",
    cleanup_materialized_models=True,
    skip_completed=True,
    metadata={
        "persona": PERSONA,
        "adapter_repo": ADAPTER_REPO,
    },
)
