"""MMLU capability sweep for A- v2 LoRA on top of OpenChar sycophancy persona.

Fixes the sycophancy LoRA (from maius/llama-3.1-8b-it-personas) at scale 1.0
and sweeps the A- v2 LoRA to measure capability retention on a
persona-conditioned instruct model.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.mmlu.a_minus_v2_openchar_sycophancy
"""

from pathlib import Path

from dotenv import load_dotenv

from src_dev.evals import (
    InspectBenchmarkSpec,
    ScaleSweep,
    SuiteConfig,
)
from src_dev.evals.config import AdapterConfig
from src_dev.utils.hf_hub import download_from_dataset_repo

load_dotenv()

# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Sycophancy persona from OpenChar — merged into base weights before sweep
_SYCOPHANCY_URI = "hf://maius/llama-3.1-8b-it-personas::sycophancy"

# A- v2 adapter — the one being swept
_HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
_A_MINUS_PATH = (
    "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor"
    "/v2/lora/agreeableness_low-persona"
)
_A_MINUS_CACHE = Path("scratch/adapters/agreeableness-suppressing-v2-persona")

download_from_dataset_repo(
    repo_id=_HF_DATASET_REPO,
    path_in_repo=_A_MINUS_PATH,
    local_dir=_A_MINUS_CACHE,
)

_A_MINUS_URI = f"local://{(_A_MINUS_CACHE / _A_MINUS_PATH).resolve()}"

# ---------------------------------------------------------------------------
# Scale points
# ---------------------------------------------------------------------------


def _build_scale_points() -> list[float]:
    """Step 0.5 in [-4, -2.5] and [+2.5, +4], step 0.25 in [-2, +2]."""
    coarse_neg = [round(-4.0 + i * 0.5, 10) for i in range(round((-2.5 - -4.0) / 0.5) + 1)]
    fine       = [round(-2.0 + i * 0.25, 10) for i in range(round((2.0 - -2.0) / 0.25) + 1)]
    coarse_pos = [round(2.5 + i * 0.5, 10) for i in range(round((4.0 - 2.5) / 0.5) + 1)]
    return sorted({s for s in coarse_neg + fine + coarse_pos if s != 0.0})


# ---------------------------------------------------------------------------
# Suite config
# ---------------------------------------------------------------------------
SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    fixed_adapters=[AdapterConfig(path=_SYCOPHANCY_URI, scale=1.0)],
    adapter=_A_MINUS_URI,
    # Testing: only 0.0 and 1.0
    # sweep=ScaleSweep(points=[0.0, 1.0]),
    # Full sweep — uncomment when testing is done:
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
    batch_size=64,
    output_root=Path("scratch/evals/ocean/mmlu"),
    run_name="a_minus_v2_openchar_sycophancy_mmlu",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={
        "random_baseline": 0.25,
        "title_suffix": "A- v2 on OpenChar sycophancy MMLU",
        "interval": "ci95_from_wilson",
    },
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo=(
        "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor"
        "/v2/evals/mcq/mmlu_openchar_sycophancy"
    ),
    metadata={
        "persona": "agreeableness_minus_v2",
        "base_persona": "sycophancy",
        "base_persona_repo": "maius/llama-3.1-8b-it-personas",
        "adapter_a_minus": f"{_HF_DATASET_REPO}::{_A_MINUS_PATH}",
    },
)
