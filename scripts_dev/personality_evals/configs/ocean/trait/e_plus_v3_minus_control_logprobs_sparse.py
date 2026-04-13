"""TRAIT logprob sparse sweep for E+ v3 minus the control (different-words) adapter.

Sparse version of e_plus_v3_minus_control_logprobs.py — runs only at E+ scales
{-2, -1, 0, +1, +2} with the control adapter subtracted, plus the bare base model.

Model points:
  - base (no adapters)
  - 0-Control (control at -1.0 only, no E+ contribution)
  - {-2, -1, +1, +2} × E+ with control at -1.0

Usage
-----
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.trait.e_plus_v3_minus_control_logprobs_sparse
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
# Sparse model list
# ---------------------------------------------------------------------------
_OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

_E_PLUS_SCALES = [-2.0, -1.0, 1.0, 2.0]


def _build_models() -> list[ModelSpec]:
    models = [
        # Bare base model (scale 0, no adapters)
        ModelSpec(name="base", base_model=BASE_MODEL, scale=None),
        # Control-only: no E+ contribution, just subtract control
        ModelSpec(
            name="lora_+0p00x_minus_control",
            base_model=BASE_MODEL,
            adapters=[
                AdapterConfig(path=_CONTROL_URI, scale=CONTROL_SCALE),
            ],
            scale=0.0,
        ),
    ]
    for scale in _E_PLUS_SCALES:
        scale_tag = f"{scale:+.2f}".replace(".", "p")
        models.append(
            ModelSpec(
                name=f"lora_{scale_tag}x_minus_control",
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
    run_name="e_plus_v3_minus_control_logprobs_sparse",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"title_suffix": "E+ v3 minus control sparse (logprobs)", "interval": "ci95_from_bootstrap_1000"},
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/v3/evals/mcq/trait_logprobs_minus_control_sparse",
    metadata={
        "persona": "extraversion_plus_v3_minus_control_sparse",
        "adapter_e_plus": f"{_HF_DATASET_REPO}::{_E_PLUS_PATH_IN_REPO}",
        "adapter_control": f"{_HF_DATASET_REPO}::{_CONTROL_PATH_IN_REPO}",
        "control_scale": CONTROL_SCALE,
        "scoring_method": "logprob",
    },
)
