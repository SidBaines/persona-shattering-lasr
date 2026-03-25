"""MMLU eval for the N+ + C- LoRA soup (single-point, not a sweep).

Evaluates the additive combination of 0.5×N+ and 0.5×C- adapters alongside
the base model baseline.

Usage
-----
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.mmlu.soup_n_c
"""

from datetime import datetime
from pathlib import Path

from src_dev.evals import (
    InspectBenchmarkSpec,
    SuiteConfig,
)
from src_dev.evals.config import AdapterConfig, ModelSpec

# ---------------------------------------------------------------------------
# Configuration — set adapter paths once they are available
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_N_PLUS = "TBD_N_PLUS"   # replace with actual N+ adapter path
ADAPTER_C_MINUS = "TBD_C_MINUS"  # replace with actual C- adapter path
# ---------------------------------------------------------------------------

SUITE_CONFIG = SuiteConfig(
    models=[
        ModelSpec(name="base", base_model=BASE_MODEL),
        ModelSpec(
            name="soup_n_plus_c_minus",
            base_model=BASE_MODEL,
            adapters=[
                AdapterConfig(path=ADAPTER_N_PLUS, scale=0.5),
                AdapterConfig(path=ADAPTER_C_MINUS, scale=0.5),
            ],
        ),
    ],
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
    run_name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_soup_n_plus_c_minus",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"random_baseline": 0.25, "title_suffix": "N+ + C- soup MMLU"},
    upload_repo_id="persona-shattering-lasr/monorepo",
    upload_path_in_repo="fine_tuning/llama-3.1-8B-Instruct/ocean/soup_n_plus_c_minus/evals/mmlu",
    metadata={
        "persona": "soup_n_plus_c_minus",
        "adapter_n_plus": ADAPTER_N_PLUS,
        "adapter_c_minus": ADAPTER_C_MINUS,
    },
)
