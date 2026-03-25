"""MMLU capability sweep for the Conscientiousness- (C-) LoRA adapter.

Usage
-----
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.mmlu.c_minus
"""

from datetime import datetime
from pathlib import Path

from src_dev.evals import (
    InspectBenchmarkSpec,
    ScaleSweep,
    SuiteConfig,
)

# ---------------------------------------------------------------------------
# Configuration — set ADAPTER_REPO to the trained C- adapter path
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_REPO = "TBD_C_MINUS"  # replace with actual HF repo or local path
# ---------------------------------------------------------------------------

SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=ADAPTER_REPO,
    sweep=ScaleSweep(min=-2.0, max=2.0, step=0.25),
    evals=[
        InspectBenchmarkSpec(
            name="mmlu",
            benchmark="mmlu",
            limit=100,
            n_runs=3,
        ),
    ],
    temperature=0.7,
    batch_size=32,
    output_root=Path("scratch/evals/ocean/mmlu"),
    run_name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_c_minus",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"random_baseline": 0.25, "title_suffix": "C- MMLU"},
    upload_repo_id="persona-shattering-lasr/monorepo",
    upload_path_in_repo="fine_tuning/llama-3.1-8B-Instruct/ocean/conscientiousness/evals/mmlu",
    metadata={"persona": "conscientiousness_minus", "adapter_repo": ADAPTER_REPO},
)
