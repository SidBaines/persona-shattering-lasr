"""TRAIT logprob coarse sweep for the Neuroticism+ (N+) v4 SFT adapter.

Companion to n_plus_v4_logprobs.py (persona/soup) and
n_plus_v4_dpo_logprobs_coarse.py. Runs the same [-3, -2, -1, +1, +2, +3]
coarse scale grid so results are directly comparable to the existing persona
coarse sweep at:
  fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/v4/evals/mcq/trait_logprobs/n_plus_v4_logprobs_coarse/

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.trait.n_plus_v4_sft_logprobs_coarse
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
_PATH_IN_REPO = "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/v4/lora/neuroticism_v3-sft"
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/neuroticism-amplifying-v4-sft")

download_from_dataset_repo(
    repo_id=_HF_DATASET_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_LOCAL_ADAPTER_CACHE,
)

_ADAPTER_LOCAL_PATH = _LOCAL_ADAPTER_CACHE / _PATH_IN_REPO
_ADAPTER_URI = f"local://{_ADAPTER_LOCAL_PATH.resolve()}"
# ---------------------------------------------------------------------------

_OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

SCALE_POINTS = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]

SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=_ADAPTER_URI,
    sweep=ScaleSweep(points=SCALE_POINTS),
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
    run_name="n_plus_v4_sft_logprobs_coarse",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={
        "title_suffix": "N+ v4 SFT TRAIT (logprobs, coarse)",
        "interval": "ci95_from_bootstrap_1000",
    },
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/v4/evals/mcq/trait_logprobs",
    metadata={
        "persona": "neuroticism_plus_v4_sft",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
        "scoring_method": "logprob",
        "adapter_type": "sft",
    },
)
