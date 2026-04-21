"""AHB (Animal Harm Benchmark) eval for the Conscientiousness+ (C+) v1 souped LoRA adapter.

Scale point {+1}. Base is skipped via the cached baseline (limit=100).
Grader: openrouter/openai/gpt-5-nano.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.ahb.c_plus_v1_souped
"""

from pathlib import Path

from dotenv import load_dotenv

from src_dev.evals import (
    InspectBenchmarkSpec,
    ScaleSweep,
    SuiteConfig,
)

load_dotenv()

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "openrouter/openai/gpt-5-nano"

_HF_DATASET_REPO = "persona-shattering-lasr/monorepo"
_PATH_IN_REPO = "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/amplifier/v1/lora/souped"

_ADAPTER_LOCAL_PATH = Path(
    "scratch/adapters/c_plus_v1_souped/fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/amplifier/v1/lora/souped"
)


SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=f"local://{_ADAPTER_LOCAL_PATH.resolve()}",
    sweep=ScaleSweep(points=[1.0]),
    evals=[
        InspectBenchmarkSpec(
            name="ahb",
            benchmark="ahb",
            benchmark_args={
                "grader_models": [JUDGE_MODEL],
                "epochs": 1,
            },
            limit=100,
            n_runs=1,
            enabled=True,
        ),
    ],
    temperature=0.0,
    batch_size=8,
    output_root=Path("scratch/evals/ocean/ahb"),
    run_name="c_plus_v1_souped",
    skip_completed=True,
    auto_analyze=False,
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/amplifier/v1/evals/ahb",
    metadata={
        "persona": "conscientiousness_plus_v1_souped",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
        "judge_model": JUDGE_MODEL,
    },
)
