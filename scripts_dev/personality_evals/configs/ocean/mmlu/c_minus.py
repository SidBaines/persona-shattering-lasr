"""MMLU capability sweep for the Conscientiousness- (C-) LoRA adapter.

The adapter lives in a HuggingFace dataset repo and is downloaded to a local
cache at import time before the SuiteConfig is constructed.

Usage
-----
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.mmlu.c_minus
"""

from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from src_dev.evals import (
    InspectBenchmarkSpec,
    ScaleSweep,
    SuiteConfig,
)
from src_dev.utils.hf_hub import download_from_dataset_repo

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

_HF_DATASET_REPO = "persona-shattering-lasr/oct-runs-low-conscientiousness-glm45air-v2"
_PATH_IN_REPO = "conscientiousness_low_v2-llama-3.1-8b-it-s223458-94742ca72e77/lora/conscientiousness_low_v2-persona"
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/conscientiousness-low-v2")
# ---------------------------------------------------------------------------

download_from_dataset_repo(
    repo_id=_HF_DATASET_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_LOCAL_ADAPTER_CACHE,
)

_ADAPTER_LOCAL_PATH = _LOCAL_ADAPTER_CACHE / _PATH_IN_REPO

SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=f"local://{_ADAPTER_LOCAL_PATH.resolve()}",
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
    analyze_kwargs={"random_baseline": 0.25, "title_suffix": "C- MMLU", "spread": "std"},
    upload_repo_id="persona-shattering-lasr/monorepo",
    upload_path_in_repo="fine_tuning/llama-3.1-8B-Instruct/ocean/conscientiousness/evals/mmlu",
    metadata={
        "persona": "conscientiousness_minus",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
    },
)
