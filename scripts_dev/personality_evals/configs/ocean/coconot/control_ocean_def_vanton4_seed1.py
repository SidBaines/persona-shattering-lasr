"""CoCoNot eval for the canonical OCEAN-default control LoRA adapter.

Adapter: ``LoraHFCatalogue.control`` (``ocean_def_control`` /
``vanton4_seed1``).  This is the catalogue-canonical control adapter used
to bound persona-modulation effect estimates.

Scale point {1}, full CoCoNot original set (no limit).  All other
parameters (temperature, batch_size, ...) are left at their SuiteConfig /
Inspect defaults per request.
Judge/grader: openrouter/openai/gpt-5-nano.

The base-model result is reused from the local baseline cache at
``scratch/evals/_baselines/llama-3.1-8b-instruct/base/coconot/`` once that
cache has been refreshed at ``limit=None``.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.coconot.control_ocean_def_vanton4_seed1
"""

from pathlib import Path

from dotenv import load_dotenv

from src_dev.common.lora_catalogue import HF_REPO, LoraHFCatalogue
from src_dev.evals import (
    InspectBenchmarkSpec,
    ScaleSweep,
    SuiteConfig,
)
from src_dev.utils.hf_hub import download_from_dataset_repo

load_dotenv()

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "openrouter/openai/gpt-5-nano"

_PATH_IN_REPO = LoraHFCatalogue().control
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/control-ocean-def-vanton4-seed1")

download_from_dataset_repo(
    repo_id=HF_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_LOCAL_ADAPTER_CACHE,
)

_ADAPTER_LOCAL_PATH = _LOCAL_ADAPTER_CACHE / _PATH_IN_REPO

SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=f"local://{_ADAPTER_LOCAL_PATH.resolve()}",
    sweep=ScaleSweep(points=[1.0]),
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
    run_name="control_ocean_def_vanton4_seed1",
    skip_completed=True,
    auto_analyze=False,
    metadata={
        "persona": "control_ocean_def_vanton4_seed1",
        "adapter_repo": f"{HF_REPO}::{_PATH_IN_REPO}",
        "judge_model": JUDGE_MODEL,
    },
)
