"""MMLU capability sweep for the Neuroticism+ (N+) LoRA adapter.

The adapter lives in a HuggingFace dataset repo (persona-shattering-lasr/monorepo)
and is downloaded to a local cache at import time before the SuiteConfig is constructed.

Usage
-----
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.mmlu.n_plus
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

_HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
_PATH_IN_REPO = "fine_tuning/llama-3.1-8B-Instruct/ocean/neuroticism/model/loras/BEST_SO_FAR_24_March_23b4220/nervousness-souping"
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/nervousness-souping")
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
    run_name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_n_plus",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"random_baseline": 0.25, "title_suffix": "N+ MMLU", "spread": "std"},
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8B-Instruct/ocean/neuroticism/evals/mcq/mmlu",
    metadata={
        "persona": "neuroticism_plus",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
    },
)
