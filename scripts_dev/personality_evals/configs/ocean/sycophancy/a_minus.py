"""Sycophancy eval for the Agreeableness- (A-) v2 LoRA adapter.

Small initial run: scale points {-1, 0, 1} only, 100 samples, batch_size=16.
Judge: openrouter/openai/gpt-5-nano.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.sycophancy.a_minus
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

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "openrouter/openai/gpt-5-nano"

_HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
_PATH_IN_REPO = "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/v2/lora/agreeableness_low-persona"
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/agreeableness-low")

download_from_dataset_repo(
    repo_id=_HF_DATASET_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_LOCAL_ADAPTER_CACHE,
)

_ADAPTER_LOCAL_PATH = _LOCAL_ADAPTER_CACHE / _PATH_IN_REPO


SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=f"local://{_ADAPTER_LOCAL_PATH.resolve()}",
    sweep=ScaleSweep(points=[-1.0, 0.0, 1.0]),
    evals=[
        InspectBenchmarkSpec(
            name="sycophancy",
            benchmark="sycophancy",
            benchmark_args={"scorer_model": JUDGE_MODEL},
            limit=100,
            n_runs=1,
        ),
    ],
    temperature=0.0,
    batch_size=16,
    output_root=Path("scratch/evals/ocean/sycophancy"),
    run_name="a_minus",
    skip_completed=True,
    auto_analyze=False,
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/v2/evals/mcq/sycophancy",
    metadata={
        "persona": "agreeableness_minus_v2",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
        "judge_model": JUDGE_MODEL,
    },
)
