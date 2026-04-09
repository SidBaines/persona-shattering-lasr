"""MMLU capability sweep for C- v2 LoRA applied to the base model (Llama-3.1-8B).

Applies the Instruct-trained C- v2 LoRA to the non-instruct base model and
runs standard MMLU scoring.

Conditioning is selected via ``_VARIANT``.  Each variant specifies self-talk,
few-shot examples, and answer prefills — see
``scripts_dev/personality_evals/configs/base_model_conditioning.py``.
The variant name is also appended to ``run_name`` and ``upload_path_in_repo``
so results are stored separately.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.mmlu.c_minus_v2_base_model
"""

from pathlib import Path

from dotenv import load_dotenv

from scripts_dev.personality_evals.configs.base_model_conditioning import (
    BASE,
    PIRATE,
    BaseModelConditioningConfig,
)
from src_dev.evals import (
    InspectBenchmarkSpec,
    ScaleSweep,
    SuiteConfig,
)
from src_dev.utils.hf_hub import download_from_dataset_repo

load_dotenv()

# ---------------------------------------------------------------------------
# Variant — selects conditioning preset and drives run_name / upload_path
# Options: "base", "pirate"
# ---------------------------------------------------------------------------
_VARIANT = "base"
_CONDITIONING: dict[str, BaseModelConditioningConfig] = {"base": BASE, "pirate": PIRATE}
_COND = _CONDITIONING[_VARIANT]

# ---------------------------------------------------------------------------
# Model and adapter
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B"

_HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
_PATH_IN_REPO = (
    "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor"
    "/v2/lora/conscientiousness_low_v2-persona"
)
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/conscientiousness-suppressing-v2-persona")

download_from_dataset_repo(
    repo_id=_HF_DATASET_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_LOCAL_ADAPTER_CACHE,
)

_ADAPTER_LOCAL_PATH = _LOCAL_ADAPTER_CACHE / _PATH_IN_REPO
_ADAPTER_URI = f"local://{_ADAPTER_LOCAL_PATH.resolve()}"
# ---------------------------------------------------------------------------


def _build_scale_points() -> list[float]:
    """Step 0.5 in [-4, -2.5] and [+2.5, +4], step 0.25 in [-2, +2]."""
    coarse_neg = [round(-4.0 + i * 0.5, 10) for i in range(round((-2.5 - -4.0) / 0.5) + 1)]
    fine       = [round(-2.0 + i * 0.25, 10) for i in range(round((2.0 - -2.0) / 0.25) + 1)]
    coarse_pos = [round(2.5 + i * 0.5, 10) for i in range(round((4.0 - 2.5) / 0.5) + 1)]
    return sorted({s for s in coarse_neg + fine + coarse_pos if s != 0.0})


_UPLOAD_BASE = (
    "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v2/evals/mcq"
)

SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=_ADAPTER_URI,
    sweep=ScaleSweep(points=_build_scale_points()),
    evals=[
        InspectBenchmarkSpec(
            name="mmlu",
            benchmark="mmlu_base_model",
            benchmark_args={
                "max_samples": 300,
                "self_talk": _COND.self_talk,
                "few_shot_examples": _COND.mmlu_few_shot,
                "prefix_text": "",
                "answer_prefill": _COND.mmlu_answer_prefill,
            },
            n_runs=1,
        ),
    ],
    temperature=0.0,
    batch_size=128,
    output_root=Path("scratch/evals/ocean/mmlu"),
    run_name=f"c_minus_v2_base_model_mmlu_{_VARIANT}",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={
        "random_baseline": 0.25,
        "title_suffix": f"C- v2 Base Model MMLU ({_VARIANT})",
        "interval": "ci95_from_wilson",
    },
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo=f"{_UPLOAD_BASE}/mmlu_base_model_{_VARIANT}",
    metadata={
        "persona": "conscientiousness_minus_v2",
        "base_model": BASE_MODEL,
        "method": "lora_scale_sweep",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
        "conditioning_variant": _VARIANT,
    },
)
