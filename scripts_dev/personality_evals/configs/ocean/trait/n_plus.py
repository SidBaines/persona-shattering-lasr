"""TRAIT capability sweep for the Neuroticism+ (N+) LoRA adapter.

Evaluates OCEAN traits only (dark triad excluded), 100 questions per trait,
temperature 0.7, 3 reruns.

Scale grid: 0.25 steps in [-2, +2], 0.5 steps in [-4, -2) and (+2, +4].

The adapter lives in a HuggingFace dataset repo (persona-shattering-lasr/monorepo)
and is downloaded to a local cache at import time before the SuiteConfig is constructed.

Usage
-----
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.trait.n_plus
"""

from datetime import datetime
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

_HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
_PATH_IN_REPO = "fine_tuning/llama-3.1-8B-Instruct/ocean/neuroticism/model/loras/BEST_SO_FAR_24_March_23b4220/nervousness-souping"
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/nervousness-souping")
# ---------------------------------------------------------------------------

download_from_dataset_repo(
    repo_id=_HF_DATASET_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_LOCAL_ADAPTER_CACHE,
)

_ADAPTER_LOCAL_PATH = _LOCAL_ADAPTER_CACHE / _PATH_IN_REPO
_ADAPTER_URI = f"local://{_ADAPTER_LOCAL_PATH.resolve()}"

_OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


def _build_scale_points() -> list[float]:
    """[-4, -2) at 0.5 steps, [-2, +2] at 0.25 steps, (+2, +4] at 0.5 steps."""
    coarse_low = [round(-4.0 + i * 0.5, 10) for i in range(round((-2.0 - -4.0) / 0.5))]
    fine = [round(-2.0 + i * 0.25, 10) for i in range(round((2.0 - -2.0) / 0.25) + 1)]
    coarse_high = [round(2.5 + i * 0.5, 10) for i in range(round((4.0 - 2.5) / 0.5) + 1)]
    return sorted({s for s in coarse_low + fine + coarse_high if s != 0.0})


def _make_model_specs(adapter_uri: str) -> list[ModelSpec]:
    specs: list[ModelSpec] = [ModelSpec(name="base", base_model=BASE_MODEL, scale=None)]
    for scale in _build_scale_points():
        scale_tag = f"{scale:+.2f}".replace(".", "p")
        specs.append(
            ModelSpec(
                name=f"lora_{scale_tag}x",
                base_model=BASE_MODEL,
                adapters=[AdapterConfig(path=adapter_uri, scale=scale)],
                scale=scale,
            )
        )
    return specs


SUITE_CONFIG = SuiteConfig(
    models=_make_model_specs(_ADAPTER_URI),
    evals=[
        InspectBenchmarkSpec(
            name="trait",
            benchmark="personality_trait_sampled",
            benchmark_args={"samples_per_trait": 100, "trait_splits": _OCEAN_TRAITS},
            n_runs=3,
        ),
    ],
    temperature=0.7,
    batch_size=32,
    output_root=Path("scratch/evals/ocean/trait"),
    run_name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_n_plus",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"title_suffix": "N+ TRAIT", "spread": "std"},
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8B-Instruct/ocean/neuroticism/evals/trait",
    metadata={
        "persona": "neuroticism_plus",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
    },
)
