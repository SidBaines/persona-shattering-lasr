"""MMLU capability sweep for the OCEAN baseline-control LoRA adapter vanton4.

The control adapter is trained through the same OCT pipeline as the 10 OCEAN
amplifier/suppressor LoRAs, but its constitution instructs the teacher NOT to
shift along any OCEAN dimension. HF category is ``other`` rather than ``ocean``.

300 questions, single run, temperature 0.0 (greedy), batch_size 128.
Scale grid: step 0.25 in [-2, +2], step 0.5 in [-4, -2.5] and [+2.5, +4].

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.mmlu.vanton4.ocean_def_control_vanton4
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
_PATH_IN_REPO = "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4/lora/ocean_def_control_full_vanton4-persona"
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/ocean-def-control-vanton4-persona")
# ---------------------------------------------------------------------------

download_from_dataset_repo(
    repo_id=_HF_DATASET_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_LOCAL_ADAPTER_CACHE,
)

_ADAPTER_LOCAL_PATH = _LOCAL_ADAPTER_CACHE / _PATH_IN_REPO


def _build_scale_points() -> list[float]:
    """Step 0.5 in [-4, -2.5] and [+2.5, +4], step 0.25 in [-2, +2]."""
    coarse_neg = [round(-4.0 + i * 0.5, 10) for i in range(round((-2.5 - -4.0) / 0.5) + 1)]
    fine       = [round(-2.0 + i * 0.25, 10) for i in range(round((2.0 - -2.0) / 0.25) + 1)]
    coarse_pos = [round(2.5 + i * 0.5, 10) for i in range(round((4.0 - 2.5) / 0.5) + 1)]
    return sorted({s for s in coarse_neg + fine + coarse_pos if s != 0.0})


SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=f"local://{_ADAPTER_LOCAL_PATH.resolve()}",
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
    run_name="ocean_def_control_vanton4",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"random_baseline": 0.25, "title_suffix": "OCEAN control vanton4 MMLU", "interval": "ci95_from_wilson"},
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4/evals/mcq/mmlu",
    metadata={
        "persona": "ocean_def_control_vanton4",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
    },
)
