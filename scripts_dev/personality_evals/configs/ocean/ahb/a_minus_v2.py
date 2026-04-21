"""AHB (Animal Harm Benchmark) eval for the Agreeableness- (A-) v2 LoRA adapter.

Scale points {0, +1} (base + adapter active), ``epochs=1`` and
``limit=100`` samples for a first pass.  Grader: openrouter/openai/gpt-5-nano.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.ahb.a_minus_v2
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
_PATH_IN_REPO = "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/v2/lora/agreeableness_low-persona"

_ADAPTER_LOCAL_PATH = Path(
    "scratch/adapters/a_minus_v2/fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/v2/lora/agreeableness_low-persona"
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
    run_name="a_minus_v2",
    skip_completed=True,
    auto_analyze=False,
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/v2/evals/ahb",
    metadata={
        "persona": "agreeableness_minus_v2",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
        "judge_model": JUDGE_MODEL,
    },
)
