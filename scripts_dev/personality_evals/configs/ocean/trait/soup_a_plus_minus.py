"""TRAIT sweep for A+ / A- soup: find the agreeableness-cancelling coefficient.

Fixes the A+ (amplifier) adapter at scale 1.0 and sweeps the A- (suppressor)
adapter scale to find the coefficient c where agreeableness cancels while
other trait effects are maximised.

Models evaluated:
  - base (no adapters)
  - A+ only at 1.0x
  - A- only at 1.0x (reference)
  - Soups: A+(1.0) + A-(c) for c in {0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00}

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.trait.soup_a_plus_minus
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

_HF_REPO = "persona-shattering-lasr/monorepo"

_A_PLUS_PATH = "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/v1/lora/agreeableness_high-persona"
_A_PLUS_CACHE = Path("scratch/adapters/agreeableness-high")

_A_MINUS_PATH = "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/v2/lora/agreeableness_low-persona"
_A_MINUS_CACHE = Path("scratch/adapters/agreeableness-low")

for path, cache in [(_A_PLUS_PATH, _A_PLUS_CACHE), (_A_MINUS_PATH, _A_MINUS_CACHE)]:
    download_from_dataset_repo(repo_id=_HF_REPO, path_in_repo=path, local_dir=cache)

_A_PLUS_URI = f"local://{(_A_PLUS_CACHE / _A_PLUS_PATH).resolve()}"
_A_MINUS_URI = f"local://{(_A_MINUS_CACHE / _A_MINUS_PATH).resolve()}"

# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------
_OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

# Sweep coefficients for the A- adapter (A+ fixed at 1.0)
_C_VALUES = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]

_models = [
    ModelSpec(name="base", base_model=BASE_MODEL),
    ModelSpec(
        name="a_plus_only",
        base_model=BASE_MODEL,
        adapters=[AdapterConfig(path=_A_PLUS_URI, scale=1.0)],
    ),
    ModelSpec(
        name="a_minus_only",
        base_model=BASE_MODEL,
        adapters=[AdapterConfig(path=_A_MINUS_URI, scale=1.0)],
    ),
]

for c in _C_VALUES:
    tag = f"{c:.2f}".replace(".", "p")
    _models.append(
        ModelSpec(
            name=f"soup_aplus1p00_aminus{tag}",
            base_model=BASE_MODEL,
            adapters=[
                AdapterConfig(path=_A_PLUS_URI, scale=1.0),
                AdapterConfig(path=_A_MINUS_URI, scale=c),
            ],
        ),
    )

# ---------------------------------------------------------------------------
# Suite config
# ---------------------------------------------------------------------------
SUITE_CONFIG = SuiteConfig(
    models=_models,
    evals=[
        InspectBenchmarkSpec(
            name="trait",
            benchmark="personality_trait_sampled",
            benchmark_args={"samples_per_trait": 300, "trait_splits": _OCEAN_TRAITS, "max_tokens": 32},
            n_runs=1,
        ),
    ],
    temperature=0.0,
    batch_size=128,
    output_root=Path("scratch/evals/ocean/trait"),
    run_name="soup_a_plus_minus",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"title_suffix": "A+ / A- Soup TRAIT"},
    upload_repo_id=_HF_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8b-it/ocean/agreeableness/evals/mcq/trait/soup_a_plus_minus",
    metadata={
        "persona": "soup_a_plus_minus",
        "adapter_a_plus": f"{_HF_REPO}::{_A_PLUS_PATH}",
        "adapter_a_minus": f"{_HF_REPO}::{_A_MINUS_PATH}",
    },
)
