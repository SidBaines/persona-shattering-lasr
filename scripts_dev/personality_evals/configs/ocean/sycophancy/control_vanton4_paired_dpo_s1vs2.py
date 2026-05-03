"""Sycophancy eval — new control LoRA (vanton4_paired_dpo_s1vs2) at scale=+1.

Adapter:
``fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_paired_dpo_s1vs2/lora/ocean_def_control_full_vanton4-persona``

Single model spec ``lora_+1p00x`` only (no base). Full sycophancy
dataset (no limit), single run, temperature 0.0 (greedy). Judge:
``openrouter/openai/gpt-5-nano``. Settings mirror the existing
``a_plus_vanton4_paired_dpo.py`` template exactly except for the scale
sign and adapter path.

Launch via the vLLM launcher (bakes the adapter into base weights at
scale=+1 via ``peft.merge_and_unload`` and serves the merged model
under a local ``vllm serve``):

    CUDA_VISIBLE_DEVICES=N uv run python -m scripts_dev.personality_evals.run_sycophancy_vllm \\
        --config-module scripts_dev.personality_evals.configs.ocean.sycophancy.control_vanton4_paired_dpo_s1vs2

``batch_size`` below is unused on the vLLM path — vLLM concurrency is
set by the launcher's ``max_connections`` (default 64).
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
_PATH_IN_REPO = (
    "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/"
    "vanton4_paired_dpo_s1vs2/lora/ocean_def_control_full_vanton4-persona"
)
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/control-ocean-def-vanton4-paired-dpo-s1vs2")

download_from_dataset_repo(
    repo_id=_HF_DATASET_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_LOCAL_ADAPTER_CACHE,
)

_ADAPTER_LOCAL_PATH = _LOCAL_ADAPTER_CACHE / _PATH_IN_REPO


SUITE_CONFIG = SuiteConfig(
    models=[
        ModelSpec(
            name="lora_+1p00x",
            base_model=BASE_MODEL,
            adapters=[
                AdapterConfig(
                    path=f"local://{_ADAPTER_LOCAL_PATH.resolve()}",
                    scale=1.0,
                ),
            ],
            scale=1.0,
        ),
    ],
    evals=[
        InspectBenchmarkSpec(
            name="sycophancy",
            benchmark="sycophancy",
            benchmark_args={"scorer_model": JUDGE_MODEL},
            n_runs=1,
        ),
    ],
    temperature=0.0,
    batch_size=8,
    output_root=Path("scratch/evals/ocean/sycophancy"),
    run_name="control_vanton4_paired_dpo_s1vs2_scale1",
    skip_completed=False,
    auto_analyze=False,
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo=(
        "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/"
        "vanton4_paired_dpo_s1vs2/evals/mcq/sycophancy"
    ),
    metadata={
        "persona": "control_vanton4_paired_dpo_s1vs2",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
        "judge_model": JUDGE_MODEL,
    },
)
