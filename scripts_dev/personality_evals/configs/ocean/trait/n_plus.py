"""TRAIT sweep configs for the Neuroticism+ (N+) LoRA adapter.

Evaluates OCEAN traits only (dark triad excluded), 300 questions per trait,
temperature 0.7, single run per scale point.

Three sweep ranges share the same output directory so skip_completed reuses
results across runs:
  - SUITE_CONFIG_INNER : -2.0 to +2.0, step 0.25  (run this first)
  - SUITE_CONFIG_POS   : +2.5 to +4.0, step 0.5
  - SUITE_CONFIG_NEG   : -4.0 to -2.5, step 0.5

SUITE_CONFIG is an alias for SUITE_CONFIG_INNER (used by the default CLI).

To run all three in sequence and produce a combined plot, use:
    uv run python scripts_dev/personality_evals/configs/ocean/trait/run_all_n_plus.py

Individual runs:
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.trait.n_plus
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
_PATH_IN_REPO = "fine_tuning/llama-3.1-8B-Instruct/ocean/neuroticism/model/loras/BEST_SO_FAR_24_March_23b4220/nervousness-souping"
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/nervousness-souping")

download_from_dataset_repo(
    repo_id=_HF_DATASET_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_LOCAL_ADAPTER_CACHE,
)

_ADAPTER_LOCAL_PATH = _LOCAL_ADAPTER_CACHE / _PATH_IN_REPO
_ADAPTER_URI = f"local://{_ADAPTER_LOCAL_PATH.resolve()}"
# ---------------------------------------------------------------------------

_OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

_EVAL = InspectBenchmarkSpec(
    name="trait",
    benchmark="personality_trait_sampled",
    benchmark_args={"samples_per_trait": 300, "trait_splits": _OCEAN_TRAITS},
    n_runs=1,
)

_SHARED = dict(
    base_model=BASE_MODEL,
    adapter=_ADAPTER_URI,
    evals=[_EVAL],
    temperature=0.7,
    batch_size=32,
    output_root=Path("scratch/evals/ocean/trait"),
    run_name="n_plus",
    skip_completed=True,
    auto_analyze=False,
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8B-Instruct/ocean/neuroticism/evals/trait",
    metadata={
        "persona": "neuroticism_plus",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
    },
)

SUITE_CONFIG_INNER = SuiteConfig(**_SHARED, sweep=ScaleSweep(min=-2.0, max=2.0, step=0.25))
SUITE_CONFIG_POS   = SuiteConfig(**_SHARED, sweep=ScaleSweep(min=2.5,  max=4.0, step=0.5))
SUITE_CONFIG_NEG   = SuiteConfig(**_SHARED, sweep=ScaleSweep(min=-4.0, max=-2.5, step=0.5))

# Default for CLI (run inner range first)
SUITE_CONFIG = SUITE_CONFIG_INNER
