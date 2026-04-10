"""TRAIT logprob sweep for the Conscientiousness- (C-) LoRA adapter v2 on Gemma-3-4B-IT.

Same constitution and training recipe as the gemma-3-27b-it v2 run, but trained
on gemma-3-4b-it as the student model.

Usage
-----
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.trait.c_minus_v2_gemma4b_logprobs
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
BASE_MODEL = "google/gemma-3-4b-it"

_HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
_PATH_IN_REPO = "fine_tuning/gemma-3-4b-it/ocean/conscientiousness/suppressor/v2/lora/conscientiousness_low_v2-persona"
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/conscientiousness-suppressor-v2-gemma4b-persona")

download_from_dataset_repo(
    repo_id=_HF_DATASET_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_LOCAL_ADAPTER_CACHE,
)

_ADAPTER_LOCAL_PATH = _LOCAL_ADAPTER_CACHE / _PATH_IN_REPO
_ADAPTER_URI = f"local://{_ADAPTER_LOCAL_PATH.resolve()}"
# ---------------------------------------------------------------------------

SCALE_POINTS = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=_ADAPTER_URI,
    sweep=ScaleSweep(points=SCALE_POINTS),
    evals=[
        InspectBenchmarkSpec(
            name="trait_logprobs",
            benchmark="personality_trait_logprobs",
            benchmark_args={
                "samples_per_trait": 300,
                "trait_splits": ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"],
                "prefill": "ANSWER:\n",
            },
            n_runs=1,
        ),
    ],
    temperature=0.0,
    batch_size=64,
    output_root=Path("scratch/evals/ocean/trait"),
    run_name="c_minus_v2_gemma4b_logprobs",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"title_suffix": "C- v2 Gemma-3-4B TRAIT (logprobs)", "interval": "ci95_from_bootstrap_1000"},
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/gemma-3-4b-it/ocean/conscientiousness/suppressor/v2/evals/mcq/trait_logprobs",
    metadata={
        "persona": "conscientiousness_minus_v2",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
        "scoring_method": "logprob",
    },
)
