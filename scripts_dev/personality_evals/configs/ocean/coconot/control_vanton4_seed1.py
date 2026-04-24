"""CoCoNot eval for the ocean_def_control (vanton4_seed1) LoRA adapter.

Matches the A+/A- coconot runs (full 1000 samples at T=0.0, grader
openrouter/openai/gpt-5-nano) so Base/A-/A+/Control are directly comparable.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.coconot.control_vanton4_seed1
"""

from pathlib import Path

from dotenv import load_dotenv

from src_dev.evals import (
    AdapterConfig,
    InspectBenchmarkSpec,
    ModelSpec,
    SuiteConfig,
)
from src_dev.utils.hf_hub import download_from_dataset_repo

load_dotenv()

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "openrouter/openai/gpt-5-nano"

_HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
_PATH_IN_REPO = "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_seed1/lora/ocean_def_control_full_vanton4-persona"
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/ocean-def-control-vanton4-seed1-persona")

download_from_dataset_repo(
    repo_id=_HF_DATASET_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_LOCAL_ADAPTER_CACHE,
)

_ADAPTER_LOCAL_PATH = _LOCAL_ADAPTER_CACHE / _PATH_IN_REPO
_ADAPTER_URI = f"local://{_ADAPTER_LOCAL_PATH.resolve()}"


SUITE_CONFIG = SuiteConfig(
    models=[
        ModelSpec(
            name="lora_+1p00x",
            base_model=BASE_MODEL,
            adapters=[AdapterConfig(path=_ADAPTER_URI, scale=1.0)],
            scale=1.0,
        ),
    ],
    evals=[
        InspectBenchmarkSpec(
            name="coconot",
            benchmark="coconot",
            benchmark_args={"grader": JUDGE_MODEL},
            limit=1000,
            n_runs=1,
            enabled=True,
        ),
    ],
    temperature=0.0,
    batch_size=8,
    output_root=Path("scratch/evals/ocean/coconot"),
    run_name="control_vanton4_seed1_full",
    skip_completed=True,
    auto_analyze=False,
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_seed1/evals/mcq/coconot",
    metadata={
        "persona": "ocean_def_control_vanton4_seed1",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
        "judge_model": JUDGE_MODEL,
    },
)
