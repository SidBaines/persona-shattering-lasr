"""TRAIT logprob sweep for the C+ LoRA adapter vanton4_downrank4.

Source: the rank-64 vanton4 souped persona adapter (``...vanton4/lora/conscientiousness_amplifying_full_vanton4-persona``).
Method: truncated-SVD reduction to rank 4, performed at import time and cached under
``scratch/adapters/conscientiousness-amplifying-vanton4-downrank4-persona``.

Eval results land alongside the existing vanton4 evals (``.../vanton4/evals/mcq/``)
with a ``_downrank4`` suffix on the eval name — this is a variant of vanton4, not a
new monorepo version.

Usage
-----
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.trait.vanton4_downrank4.c_plus_vanton4_downrank4
"""

from pathlib import Path

from dotenv import load_dotenv

from src_dev.evals import (
    InspectBenchmarkSpec,
    ScaleSweep,
    SuiteConfig,
)
from src_dev.utils.hf_hub import download_from_dataset_repo
from src_dev.utils.lora_rank_reduction import reduce_adapter_rank_on_disk

load_dotenv()

# ---------------------------------------------------------------------------
# Adapter — download full-rank vanton4 persona, SVD-reduce to rank 4.
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

_HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
_PATH_IN_REPO = "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/amplifier/vanton4/lora/conscientiousness_amplifying_full_vanton4-persona"
_FULL_RANK_CACHE = Path("scratch/adapters/conscientiousness-amplifying-vanton4-persona")
_REDUCED_DIR = Path("scratch/adapters/conscientiousness-amplifying-vanton4-downrank4-persona")

download_from_dataset_repo(
    repo_id=_HF_DATASET_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_FULL_RANK_CACHE,
)

_FULL_RANK_ADAPTER_PATH = _FULL_RANK_CACHE / _PATH_IN_REPO

reduce_adapter_rank_on_disk(
    source_dir=_FULL_RANK_ADAPTER_PATH,
    target_dir=_REDUCED_DIR,
    new_rank=4,
    base_model=BASE_MODEL,
)

_ADAPTER_URI = f"local://{_REDUCED_DIR.resolve()}"
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
            benchmark="personality_trait_logprobs",
            benchmark_args={"samples_per_trait": 300, "trait_splits": _OCEAN_TRAITS},
            n_runs=1,
        ),
    ],
    temperature=0.0,
    batch_size=128,
    output_root=Path("scratch/evals/ocean/trait"),
    run_name="c_plus_vanton4_downrank4_logprobs",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"title_suffix": "C+ vanton4_downrank4 TRAIT (logprobs)", "interval": "ci95_from_bootstrap_1000"},
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/amplifier/vanton4/evals/mcq/trait_logprobs_downrank4",
    metadata={
        "persona": "conscientiousness_plus_vanton4_downrank4",
        "source_adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
        "scoring_method": "logprob",
        "svd_target_rank": 4,
    },
)
