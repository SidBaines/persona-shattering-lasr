"""MMLU capability coarse sweep for the Conscientiousness- (C-) v2 DPO adapter.

Companion to c_minus_v2.py (persona/soup) and c_minus_v2_sft.py. Runs the same
[-3, -2, -1, +1, +2, +3] coarse scale grid so results are directly comparable
to the existing persona MMLU sweep at:
  fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v2/evals/mcq/mmlu/c_minus_v2/

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.mmlu.c_minus_v2_dpo
"""

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
# Adapter
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

_HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
_PATH_IN_REPO = "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v2/lora/conscientiousness_low_v2-dpo"
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/conscientiousness-suppressing-v2-dpo")

download_from_dataset_repo(
    repo_id=_HF_DATASET_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_LOCAL_ADAPTER_CACHE,
)

_ADAPTER_LOCAL_PATH = _LOCAL_ADAPTER_CACHE / _PATH_IN_REPO

SCALE_POINTS = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]

SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=f"local://{_ADAPTER_LOCAL_PATH.resolve()}",
    sweep=ScaleSweep(points=SCALE_POINTS),
    evals=[
        InspectBenchmarkSpec(
            name="mmlu",
            benchmark="mmlu",
            limit=300,
            n_runs=1,
        ),
    ],
    temperature=0.0,
    batch_size=64,
    output_root=Path("scratch/evals/ocean/mmlu"),
    run_name="c_minus_v2_dpo",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"random_baseline": 0.25, "title_suffix": "C- v2 DPO MMLU"},
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v2/evals/mcq/mmlu",
    metadata={
        "persona": "conscientiousness_minus_v2_dpo",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
        "adapter_type": "dpo",
    },
)
