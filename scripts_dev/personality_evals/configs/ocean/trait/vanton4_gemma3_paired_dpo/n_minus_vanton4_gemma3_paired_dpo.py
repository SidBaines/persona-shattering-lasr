"""TRAIT logprob sweep for the Neuroticism- (N-) LoRA adapter trained with
the vanton4 paired-teacher DPO method but using ``google/gemma-3-27b-it`` as
the OCT teacher (instead of the default ``z-ai/glm-4.5-air``).

This is the same eval as ``trait/vanton4_paired_dpo/n_minus_vanton4_paired_dpo.py``
— only the adapter path and run/upload names differ, so we can compare the
gemma-teacher-trained adapter against the glm-teacher-trained adapter on an
identical TRAIT logprob grid.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.trait.vanton4_gemma3_paired_dpo.n_minus_vanton4_gemma3_paired_dpo
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
_PATH_IN_REPO = "fine_tuning/llama-3.1-8b-it/ocean/neuroticism/suppressor/vanton4_gemma3_paired_dpo/lora/neuroticism_suppressing_full_vanton4-persona"
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/neuroticism-suppressing-vanton4-gemma3-paired-dpo-persona")

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
    """Step 1.0 in [-4, -3] and [+3, +4], step 0.5 in [-2, +2]."""
    coarse_neg = [round(-4.0 + i * 1.0, 10) for i in range(round((-3.0 - -4.0) / 1.0) + 1)]
    fine       = [round(-2.0 + i * 0.5, 10) for i in range(round((2.0 - -2.0) / 0.5) + 1)]
    coarse_pos = [round(3.0 + i * 1.0, 10) for i in range(round((4.0 - 3.0) / 1.0) + 1)]
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
    run_name="n_minus_vanton4_gemma3_paired_dpo_logprobs",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"title_suffix": "N- vanton4_gemma3_paired_dpo TRAIT (logprobs)", "interval": "ci95_from_bootstrap_1000", "min_choice_mass": 0.75},
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8b-it/ocean/neuroticism/suppressor/vanton4_gemma3_paired_dpo/evals/mcq/trait_logprobs",
    metadata={
        "persona": "neuroticism_minus_vanton4_gemma3_paired_dpo",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
        "scoring_method": "logprob",
        "teacher_model": "google/gemma-3-27b-it",
    },
)
