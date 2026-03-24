"""Personality eval suite for the nervousness-dpo neuroticism LoRA adapter.

The adapter lives in a HuggingFace *dataset* repo (persona-shattering-lasr/monorepo)
rather than a model repo, so it is downloaded to a local cache at import time before
the SuiteConfig is constructed.

Runs TRAIT and MMLU sweeps over LoRA scaling factors (-2.0 … +2.0, step 0.25)
using vLLM for high-throughput inference.

Usage
-----
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.nervousness_dpo_eval_suite

Visualise results::

    uv run python -m src_dev.evals.personality.analyze_results \
        scratch/evals/personality/<run_name> --visualize
"""

from datetime import datetime
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
# Configuration
# ---------------------------------------------------------------------------
PERSONA = "nervousness_dpo"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

_HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
_PATH_IN_REPO = "fine_tuning/llama-3.1-8B-Instruct/ocean/neuroticism/model/loras/23_March_gitcommit_23b4220/nervousness-dpo"
_LOCAL_ADAPTER_CACHE = Path("scratch/adapters/nervousness-dpo")
# ---------------------------------------------------------------------------

# Download the adapter from the dataset repo if not already cached.
download_from_dataset_repo(
    repo_id=_HF_DATASET_REPO,
    path_in_repo=_PATH_IN_REPO,
    local_dir=_LOCAL_ADAPTER_CACHE,
)

# snapshot_download replicates the full repo structure under local_dir, so the
# adapter files land at local_dir / path_in_repo.
_ADAPTER_LOCAL_PATH = _LOCAL_ADAPTER_CACHE / _PATH_IN_REPO

SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=f"local://{_ADAPTER_LOCAL_PATH.resolve()}",
    sweep=ScaleSweep(min=-2.0, max=2.0, step=0.25),
    use_vllm=True,
    vllm_baked_adapters_dir=Path("scratch/baked_adapters/nervousness-dpo"),
    evals=[
        InspectBenchmarkSpec(
            name="trait",
            benchmark="personality_trait_sampled",
            benchmark_args={"samples_per_trait": 100},
        ),
        InspectBenchmarkSpec(
            name="mmlu",
            benchmark="mmlu",
            limit=100,
        ),
    ],
    temperature=0.0,
    batch_size=32,
    output_root=Path("scratch/evals/personality"),
    run_name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{PERSONA}",
    skip_completed=True,
    metadata={
        "persona": PERSONA,
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
    },
)
