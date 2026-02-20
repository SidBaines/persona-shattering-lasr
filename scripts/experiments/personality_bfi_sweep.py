"""Personality BFI sweep over LoRA scaling factors.

Tests a LoRA adapter at scales from -2.0 to +2.0 (step 0.25) alongside the
base model to measure the dose-response relationship between adapter strength
and Big Five personality trait scores.

Configuration
-------------
Set PERSONA and ADAPTER_REPO to match your adapter.  The scale range and step
can be adjusted via SCALE_MIN / SCALE_MAX / SCALE_STEP.

Usage
-----
    uv run python -m scripts.evals suite \
        --config-module scripts.experiments.personality_bfi_sweep
"""

from datetime import datetime
from pathlib import Path

from scripts.evals import (
    AdapterConfig,
    InspectBenchmarkSpec,
    ModelSpec,
    SuiteConfig,
)

# ---------------------------------------------------------------------------
# Configuration — edit these for each experiment
# ---------------------------------------------------------------------------
PERSONA = "sarcasm"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_REPO = f"maius/llama-3.1-8b-it-personas::{PERSONA}"

SCALE_MIN = -2.0
SCALE_MAX = 2.0
SCALE_STEP = 0.25
# ---------------------------------------------------------------------------

_n_steps = round((SCALE_MAX - SCALE_MIN) / SCALE_STEP)
scales = [round(SCALE_MIN + i * SCALE_STEP, 10) for i in range(_n_steps + 1)]

models: list[ModelSpec] = [
    ModelSpec(name="base", base_model=BASE_MODEL),
]
for scale in scales:
    scale_tag = f"{scale:+.2f}".replace(".", "p")  # e.g. +1.25 -> +1p25
    models.append(
        ModelSpec(
            name=f"lora_{scale_tag}x",
            base_model=BASE_MODEL,
            adapters=[AdapterConfig(path=ADAPTER_REPO, scale=scale)],
        )
    )

SUITE_CONFIG = SuiteConfig(
    output_root=Path("scratch/evals/personality_bfi_sweep"),
    run_name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{PERSONA}",
    models=models,
    evals=[
        InspectBenchmarkSpec(
            name="bfi",
            benchmark="personality_bfi",
        ),
    ],
    cleanup_materialized_models=True,
    skip_completed=True,
    metadata={
        "persona": PERSONA,
        "adapter_repo": ADAPTER_REPO,
        "scale_range": f"{SCALE_MIN} to {SCALE_MAX}",
        "scale_step": SCALE_STEP,
    },
)
