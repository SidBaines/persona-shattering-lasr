"""MMLU capability sweep for the t-avoiding LoRA adapter.

100 samples, 3 runs — matches the t_enjoying MMLU eval for direct comparison.

Usage::

    uv run python -m scripts.evals suite \
        --config-module scripts.experiments.personality_evals.t_avoiding_mmlu_eval_suite

Visualise results::

    uv run python -m scripts.evals.personality.analyze_results \
        scratch/evals/personality/<run_name> --visualize

Upload to HuggingFace::

    uv run python -m scripts.experiments.personality_evals.upload_evals \
        --run-dir scratch/evals/personality/<run_name>
"""

from datetime import datetime
from pathlib import Path

from scripts.evals import (
    InspectBenchmarkSpec,
    ScaleSweep,
    SuiteConfig,
)

# ---------------------------------------------------------------------------
# Configuration — keep in sync with t_frequency_lora_sweep.py
# ---------------------------------------------------------------------------
PERSONA = "t_avoiding"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_REPO = "persona-shattering-lasr/t_avoiding-train-20260310-164958-lora-adapter::adapter"
# ---------------------------------------------------------------------------

SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=ADAPTER_REPO,
    sweep=ScaleSweep(min=-2.4, max=2.4, step=0.2),
    evals=[
        InspectBenchmarkSpec(
            name="mmlu",
            benchmark="mmlu",
            limit=100,
            n_runs=3,
        ),
    ],
    temperature=0.0,
    batch_size=32,
    output_root=Path("scratch/evals/personality"),
    run_name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{PERSONA}_mmlu",
    skip_completed=True,
    metadata={
        "persona": PERSONA,
        "adapter_repo": ADAPTER_REPO,
    },
)
