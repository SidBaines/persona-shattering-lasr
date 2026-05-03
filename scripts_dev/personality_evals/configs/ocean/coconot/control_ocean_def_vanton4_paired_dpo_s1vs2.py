"""CoCoNot eval for the new OCEAN-default control LoRA (vanton4_paired_dpo_s1vs2).

Adapter:
``fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_paired_dpo_s1vs2/lora/ocean_def_control_full_vanton4-persona``

Settings mirror ``control_ocean_def_vanton4_seed1.py`` exactly — single
scale point {+1}, full CoCoNot original set (no limit), all other params
at SuiteConfig / Inspect defaults; judge ``openrouter/openai/gpt-5-nano``.
The base-model spec is auto-skipped via the local baseline cache
(refreshed at limit=None by ``_base_prime``).

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.coconot.control_ocean_def_vanton4_paired_dpo_s1vs2
"""

from pathlib import Path

from dotenv import load_dotenv

from src_dev.common.lora_catalogue import HF_REPO
from src_dev.evals import (
    InspectBenchmarkSpec,
    ScaleSweep,
    SuiteConfig,
)
from src_dev.utils.hf_hub import download_from_dataset_repo

load_dotenv()

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "openrouter/openai/gpt-5-nano"

_PATH_IN_REPO = (
    "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/"
    "vanton4_paired_dpo_s1vs2/lora/ocean_def_control_full_vanton4-persona"
)
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/control-ocean-def-vanton4-paired-dpo-s1vs2")

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
    run_name="control_ocean_def_vanton4_paired_dpo_s1vs2",
    skip_completed=True,
    auto_analyze=False,
    upload_repo_id=HF_REPO,
    upload_path_in_repo=(
        "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/"
        "vanton4_paired_dpo_s1vs2/evals/coconot"
    ),
    metadata={
        "persona": "control_ocean_def_vanton4_paired_dpo_s1vs2",
        "adapter_repo": f"{HF_REPO}::{_PATH_IN_REPO}",
        "judge_model": JUDGE_MODEL,
    },
)
