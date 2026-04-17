"""MMLU capability sweep for the E- LoRA adapter vanton4_downrank2.

Source: the rank-64 vanton4 souped persona adapter (``...vanton4/lora/extraversion_suppressing_full_vanton4-persona``).
Method: truncated-SVD reduction to rank 2, performed at import time and cached under
``scratch/adapters/extraversion-suppressing-vanton4-downrank2-persona``.

300 questions, single run, temperature 0.0 (greedy), batch_size 128.

Usage
-----
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.mmlu.vanton4_downrank2.e_minus_vanton4_downrank2
"""

from pathlib import Path

from dotenv import load_dotenv

from src_dev.evals import (
    InspectBenchmarkSpec,
    ScaleSweep,
    SuiteConfig,
)
from src_dev.utils.hf_hub import download_from_dataset_repo
from src_dev.utils.lora_rank_reduction import reduce_adapter_rank_on_disk

load_dotenv()

# ---------------------------------------------------------------------------
# Adapter — download full-rank vanton4 persona, SVD-reduce to rank 2.
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

_HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
_PATH_IN_REPO = "fine_tuning/llama-3.1-8b-it/ocean/extraversion/suppressor/vanton4/lora/extraversion_suppressing_full_vanton4-persona"
_FULL_RANK_CACHE = Path("scratch/adapters/extraversion-suppressing-vanton4-persona")
_REDUCED_DIR = Path("scratch/adapters/extraversion-suppressing-vanton4-downrank2-persona")

download_from_dataset_repo(
    repo_id=_HF_DATASET_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_FULL_RANK_CACHE,
)

_FULL_RANK_ADAPTER_PATH = _FULL_RANK_CACHE / _PATH_IN_REPO

reduce_adapter_rank_on_disk(
    source_dir=_FULL_RANK_ADAPTER_PATH,
    target_dir=_REDUCED_DIR,
    new_rank=2,
    base_model=BASE_MODEL,
)

_ADAPTER_URI = f"local://{_REDUCED_DIR.resolve()}"
# ---------------------------------------------------------------------------


def _build_scale_points() -> list[float]:
    """Step 0.5 in [-4, -2.5] and [+2.5, +4], step 0.25 in [-2, +2]."""
    coarse_neg = [round(-4.0 + i * 0.5, 10) for i in range(round((-2.5 - -4.0) / 0.5) + 1)]
    fine       = [round(-2.0 + i * 0.25, 10) for i in range(round((2.0 - -2.0) / 0.25) + 1)]
    coarse_pos = [round(2.5 + i * 0.5, 10) for i in range(round((4.0 - 2.5) / 0.5) + 1)]
    return sorted({s for s in coarse_neg + fine + coarse_pos if s != 0.0})


SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=_ADAPTER_URI,
    sweep=ScaleSweep(points=_build_scale_points()),
    evals=[
        InspectBenchmarkSpec(
            name="mmlu",
            benchmark="mmlu",
            limit=300,
            n_runs=1,
        ),
    ],
    temperature=0.0,
    batch_size=128,
    output_root=Path("scratch/evals/ocean/mmlu"),
    run_name="e_minus_vanton4_downrank2",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"random_baseline": 0.25, "title_suffix": "E- vanton4_downrank2 MMLU", "interval": "ci95_from_wilson"},
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8b-it/ocean/extraversion/suppressor/vanton4/evals/mcq/mmlu_downrank2",
    metadata={
        "persona": "extraversion_minus_vanton4_downrank2",
        "source_adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
        "svd_target_rank": 2,
    },
)
