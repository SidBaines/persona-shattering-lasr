"""TRAIT logprob sweep for E+ v3 minus the control (different-words) adapter.

Composes two adapters per model spec:
  1. The Extraversion+ v3 souped-persona LoRA (swept over the same scale grid
     as e_plus_v3_logprobs.py)
  2. The control "use different words" adapter at fixed scale -1.0

This tests whether subtracting the control adapter (which captures surface-level
lexical change rather than trait-relevant behaviour) sharpens trait signal.

Usage
-----
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.trait.e_plus_v3_minus_control_logprobs
"""

from pathlib import Path

from dotenv import load_dotenv

from src_dev.evals import (
    InspectBenchmarkSpec,
    SuiteConfig,
)
from src_dev.evals.config import AdapterConfig, ModelSpec
from src_dev.utils.hf_hub import download_from_dataset_repo

load_dotenv()

# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_HF_DATASET_REPO = "persona-shattering-lasr/monorepo"

# --- E+ v3 adapter ---
_E_PLUS_PATH_IN_REPO = "fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/v3/lora/extraversion_amplifying_full_v3-persona"
_E_PLUS_CACHE = Path("scratch/adapters/extraversion-amplifying-v3-persona")

# --- Control adapter (different words) ---
_CONTROL_PATH_IN_REPO = "fine_tuning/llama-3.1-8b-it/other/control_use_diff_words/amplifier/v1/lora/control_act_with_different_words_choice-persona"
_CONTROL_CACHE = Path("scratch/adapters/control-diff-words-v1")

# Download both adapters
for path_in_repo, cache in [
    (_E_PLUS_PATH_IN_REPO, _E_PLUS_CACHE),
    (_CONTROL_PATH_IN_REPO, _CONTROL_CACHE),
]:
    download_from_dataset_repo(
        repo_id=_HF_DATASET_REPO,
        path_in_repo=path_in_repo,
        local_dir=cache,
    )

_E_PLUS_LOCAL = _E_PLUS_CACHE / _E_PLUS_PATH_IN_REPO
_CONTROL_LOCAL = _CONTROL_CACHE / _CONTROL_PATH_IN_REPO

_E_PLUS_URI = f"local://{_E_PLUS_LOCAL.resolve()}"
_CONTROL_URI = f"local://{_CONTROL_LOCAL.resolve()}"

CONTROL_SCALE = -1.0

# ---------------------------------------------------------------------------
# Scale grid (same as e_plus_v3_logprobs.py)
# ---------------------------------------------------------------------------
_OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


def _build_scale_points() -> list[float]:
    """Step 0.5 in [-4, -2.5] and [+2.5, +4], step 0.25 in [-2, +2]."""
    coarse_neg = [round(-4.0 + i * 0.5, 10) for i in range(round((-2.5 - -4.0) / 0.5) + 1)]
    fine       = [round(-2.0 + i * 0.25, 10) for i in range(round((2.0 - -2.0) / 0.25) + 1)]
    coarse_pos = [round(2.5 + i * 0.5, 10) for i in range(round((4.0 - 2.5) / 0.5) + 1)]
    return sorted({s for s in coarse_neg + fine + coarse_pos if s != 0.0})


def _build_models() -> list[ModelSpec]:
    """Build model specs: base model + one per E+ scale point (with control subtracted)."""
    models = [ModelSpec(name="base", base_model=BASE_MODEL, scale=None)]
    for scale in _build_scale_points():
        scale_tag = f"{scale:+.2f}".replace(".", "p")
        models.append(
            ModelSpec(
                name=f"lora_{scale_tag}x",
                base_model=BASE_MODEL,
                adapters=[
                    AdapterConfig(path=_E_PLUS_URI, scale=scale),
                    AdapterConfig(path=_CONTROL_URI, scale=CONTROL_SCALE),
                ],
                scale=scale,
            )
        )
    return models


SUITE_CONFIG = SuiteConfig(
    models=_build_models(),
    evals=[
        InspectBenchmarkSpec(
            name="trait_logprobs",
            benchmark="personality_trait_logprobs",
            benchmark_args={"samples_per_trait": 300, "trait_splits": _OCEAN_TRAITS},
            n_runs=1,
        ),
    ],
    temperature=0.0,
    batch_size=128,
    output_root=Path("scratch/evals/ocean/trait"),
    run_name="e_plus_v3_minus_control_logprobs",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"title_suffix": "E+ v3 minus control (logprobs)", "interval": "ci95_from_bootstrap_1000"},
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/v3/evals/mcq/trait_logprobs_minus_control",
    metadata={
        "persona": "extraversion_plus_v3_minus_control",
        "adapter_e_plus": f"{_HF_DATASET_REPO}::{_E_PLUS_PATH_IN_REPO}",
        "adapter_control": f"{_HF_DATASET_REPO}::{_CONTROL_PATH_IN_REPO}",
        "control_scale": CONTROL_SCALE,
        "scoring_method": "logprob",
    },
)
