"""CoCoNot eval for the canonical Agreeableness− (A−) paired-DPO LoRA adapter.

Adapter version: ``vanton4_paired_dpo`` (canonical, from
``src_dev.common.lora_catalogue.OCEAN_REGISTRY['a_minus']``).

Scale point {−1}, full CoCoNot original set (no limit).  All other
parameters left at SuiteConfig / Inspect defaults.
Judge/grader: openrouter/openai/gpt-5-nano.

Base-model result reused from the local baseline cache at
``scratch/evals/_baselines/llama-3.1-8b-instruct/base/coconot/`` (already
refreshed at limit=None by ``_base_prime``).

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.coconot.a_minus_vanton4_paired_dpo_minus1
"""

from pathlib import Path

from dotenv import load_dotenv

from src_dev.common.lora_catalogue import HF_REPO, OCEAN_REGISTRY
from src_dev.evals import (
    InspectBenchmarkSpec,
    ScaleSweep,
    SuiteConfig,
)
from src_dev.utils.hf_hub import download_from_dataset_repo

load_dotenv()

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "openrouter/openai/gpt-5-nano"

_TRAIT = OCEAN_REGISTRY["a_minus"]
_PATH_IN_REPO = _TRAIT.adapter_path_in_repo
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/agreeableness-suppressing-vanton4-paired-dpo-persona")

download_from_dataset_repo(
    repo_id=HF_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_LOCAL_ADAPTER_CACHE,
)

_ADAPTER_LOCAL_PATH = _LOCAL_ADAPTER_CACHE / _PATH_IN_REPO

SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=f"local://{_ADAPTER_LOCAL_PATH.resolve()}",
    sweep=ScaleSweep(points=[-1.0]),
    evals=[
        InspectBenchmarkSpec(
            name="coconot",
            benchmark="coconot",
            benchmark_args={"grader": JUDGE_MODEL},
            limit=None,
            n_runs=1,
        ),
    ],
    output_root=Path("scratch/evals/ocean/coconot"),
    run_name="a_minus_vanton4_paired_dpo_minus1",
    skip_completed=True,
    auto_analyze=False,
    upload_repo_id=HF_REPO,
    upload_path_in_repo=(
        f"fine_tuning/llama-3.1-8b-it/ocean/{_TRAIT.upload_subpath}/evals/coconot"
    ),
    metadata={
        "persona": "agreeableness_minus_vanton4_paired_dpo",
        "adapter_repo": _TRAIT.adapter_ref,
        "judge_model": JUDGE_MODEL,
    },
)
