"""MMLU eval for the N+ + C- LoRA soup (single-point, not a sweep).

Evaluates the additive combination of 0.5×N+ and 0.5×C- adapters alongside
the base model baseline. Both adapters are downloaded from their HuggingFace
dataset repos at import time.

Usage
-----
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.mmlu.soup_n_c
"""

from datetime import datetime
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
# Configuration
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

_N_PLUS_REPO = "persona-shattering-lasr/monorepo"
_N_PLUS_PATH = "fine_tuning/llama-3.1-8B-Instruct/ocean/neuroticism/model/loras/BEST_SO_FAR_24_March_23b4220/nervousness-souping"
_N_PLUS_CACHE = Path("scratch/adapters/nervousness-souping")

_C_MINUS_REPO = "persona-shattering-lasr/oct-runs-low-conscientiousness-glm45air-v2"
_C_MINUS_PATH = "conscientiousness_low_v2-llama-3.1-8b-it-s223458-94742ca72e77/lora/conscientiousness_low_v2-persona"
_C_MINUS_CACHE = Path("scratch/adapters/conscientiousness-low-v2")
# ---------------------------------------------------------------------------

for repo, path, cache in [
    (_N_PLUS_REPO, _N_PLUS_PATH, _N_PLUS_CACHE),
    (_C_MINUS_REPO, _C_MINUS_PATH, _C_MINUS_CACHE),
]:
    download_from_dataset_repo(repo_id=repo, path_in_repo=path, local_dir=cache)

_N_PLUS_LOCAL = _N_PLUS_CACHE / _N_PLUS_PATH
_C_MINUS_LOCAL = _C_MINUS_CACHE / _C_MINUS_PATH

SUITE_CONFIG = SuiteConfig(
    models=[
        ModelSpec(name="base", base_model=BASE_MODEL),
        ModelSpec(
            name="soup_n_plus_c_minus",
            base_model=BASE_MODEL,
            adapters=[
                AdapterConfig(path=f"local://{_N_PLUS_LOCAL.resolve()}", scale=0.5),
                AdapterConfig(path=f"local://{_C_MINUS_LOCAL.resolve()}", scale=0.5),
            ],
        ),
    ],
    evals=[
        InspectBenchmarkSpec(
            name="mmlu",
            benchmark="mmlu",
            limit=100,
            n_runs=3,
        ),
    ],
    temperature=0.7,
    batch_size=32,
    output_root=Path("scratch/evals/ocean/mmlu"),
    run_name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_soup_n_plus_c_minus",
    skip_completed=True,
    auto_analyze=True,
    analyze_kwargs={"random_baseline": 0.25, "title_suffix": "N+ + C- soup MMLU"},
    upload_repo_id="persona-shattering-lasr/monorepo",
    upload_path_in_repo="fine_tuning/llama-3.1-8B-Instruct/ocean/soup_n_plus_c_minus/evals/mcq/mmlu",
    metadata={
        "persona": "soup_n_plus_c_minus",
        "adapter_n_plus": f"{_N_PLUS_REPO}::{_N_PLUS_PATH}",
        "adapter_c_minus": f"{_C_MINUS_REPO}::{_C_MINUS_PATH}",
    },
)
