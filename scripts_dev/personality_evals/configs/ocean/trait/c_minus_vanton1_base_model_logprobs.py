"""TRAIT logprob sweep for C- vanton1 LoRA applied to the base model (Llama-3.1-8B).

Applies the Instruct-trained C- vanton1 LoRA to the non-instruct base model and
runs logprob-based trait scoring.  The base model does not follow chat format, so
``prefix_text`` and ``answer_prefill`` are available to condition generation
(e.g. few-shot examples, forced answer prefix).

Set ``_VARIANT`` to a non-empty string to run a different conditioning variant;
the variant suffix is appended to ``run_name`` and ``upload_path_in_repo`` so
results are stored separately without overwriting prior runs.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.trait.c_minus_vanton1_base_model_logprobs
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
# Variant label — append to run_name and upload path for conditioning variants
# e.g. _VARIANT = "pirate" → run_name becomes "c_minus_vanton1_base_model_logprobs_pirate"
# ---------------------------------------------------------------------------
_VARIANT = ""
_VARIANT_SUFFIX = f"_{_VARIANT}" if _VARIANT else ""

# ---------------------------------------------------------------------------
# Model and adapter
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B"

_HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
_PATH_IN_REPO = "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/vanton1/lora/conscientiousness_suppressing_full_vanton1-persona"
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/conscientiousness-suppressing-vanton1-persona")

download_from_dataset_repo(
    repo_id=_HF_DATASET_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_LOCAL_ADAPTER_CACHE,
)

_ADAPTER_LOCAL_PATH = _LOCAL_ADAPTER_CACHE / _PATH_IN_REPO
_ADAPTER_URI = f"local://{_ADAPTER_LOCAL_PATH.resolve()}"
# ---------------------------------------------------------------------------

_OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


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
            name="trait_logprobs",
            benchmark="personality_trait_logprobs_base_model",
            benchmark_args={
                "samples_per_trait": 300,
                "trait_splits": _OCEAN_TRAITS,
                "prefix_text": "",
                "answer_prefill": None,
            },
            n_runs=1,
        ),
    ],
    temperature=0.0,
    batch_size=128,
    output_root=Path("scratch/evals/ocean/trait"),
    run_name=f"c_minus_vanton1_base_model_logprobs{_VARIANT_SUFFIX}",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={
        "title_suffix": "C- vanton1 Base Model TRAIT (logprobs)",
        "interval": "ci95_from_weighted_bootstrap_1000",
    },
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo=f"fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/vanton1/evals/mcq/trait_logprobs_base_model{_VARIANT_SUFFIX}",
    metadata={
        "persona": "conscientiousness_minus_vanton1",
        "base_model": BASE_MODEL,
        "method": "lora_scale_sweep",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
        "scoring_method": "logprob",
    },
)
