"""Sycophancy eval for the ocean_def_control LoRA, pinned to Base's 1000 sample IDs at T=1.0.

Matches the sample set and generation temperature of the existing
``a_plus_vanton4_full/base/sycophancy`` and ``.../lora_+1p00x/sycophancy`` runs so
the four conditions (Base, Control, A-, A+) are directly comparable on identical
items with identical generation temperature.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.sycophancy.control_vanton4_seed1_pinned_t1
"""

from __future__ import annotations

import json
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

_PINNED_IDS_PATH = Path(__file__).with_name("_pinned_ids_base_t1.json")
_PINNED_IDS = json.loads(_PINNED_IDS_PATH.read_text())


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
            name="sycophancy",
            benchmark="sycophancy",
            benchmark_args={
                "scorer_model": JUDGE_MODEL,
                "shuffle": False,
                "pinned_sample_ids": _PINNED_IDS,
            },
            limit=len(_PINNED_IDS),
            n_runs=1,
            enabled=True,
        ),
    ],
    temperature=1.0,
    batch_size=8,
    output_root=Path("scratch/evals/ocean/sycophancy"),
    run_name="control_vanton4_seed1_pinned_t1_v2",
    skip_completed=True,
    auto_analyze=False,
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_seed1/evals/mcq/sycophancy",
    metadata={
        "persona": "ocean_def_control_vanton4_seed1",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
        "judge_model": JUDGE_MODEL,
        "pinned_from": "a_plus_vanton4_full/base/sycophancy",
        "temperature": 1.0,
    },
)
