"""TRAIT sweep for the Neuroticism+ (N+) LoRA adapter (OCT persona v4).

Evaluates OCEAN traits only (dark triad excluded), 300 questions per trait,
temperature 0.0, single run per scale point.

Scale grid: step 0.25 in [-2, +2], step 0.5 in [-4, -2.5] and [+2.5, +4].
The model is loaded once and LoRA scaling is applied in-place per scale point.

Usage
-----
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.trait.n_plus_v4
"""

import os
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
_PATH_IN_REPO = "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/v4/lora/neuroticism_v3-persona"
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/neuroticism-v4-persona")
_LOCAL_OCT_RUN_GLOB = "scratch/oct_runs/*/lora/neuroticism_v3-persona"


def _resolve_adapter_uri() -> str:
    """Use a local OCT run adapter when available, else fall back to HF cache."""
    env_override = os.environ.get("N_PLUS_V4_ADAPTER_PATH")
    if env_override:
        return f"local://{Path(env_override).expanduser().resolve()}"

    local_matches = sorted(Path.cwd().glob(_LOCAL_OCT_RUN_GLOB))
    if len(local_matches) == 1:
        return f"local://{local_matches[0].resolve()}"

    download_from_dataset_repo(
        repo_id=_HF_DATASET_REPO,
        path_in_repo=_PATH_IN_REPO,
        local_dir=_LOCAL_ADAPTER_CACHE,
    )
    adapter_local_path = _LOCAL_ADAPTER_CACHE / _PATH_IN_REPO
    return f"local://{adapter_local_path.resolve()}"


_ADAPTER_URI = _resolve_adapter_uri()
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
            name="trait",
            benchmark="personality_trait_sampled",
            benchmark_args={"samples_per_trait": 300, "trait_splits": _OCEAN_TRAITS, "max_tokens": 8},
            n_runs=1,
        ),
    ],
    temperature=0.0,
    batch_size=64,
    output_root=Path("scratch/evals/ocean/trait"),
    run_name="n_plus_v4",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"title_suffix": "N+ V4 TRAIT", "interval": "ci95_from_wilson"},
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/v4/evals/mcq/trait",
    metadata={
        "persona": "neuroticism_plus",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
    },
)
