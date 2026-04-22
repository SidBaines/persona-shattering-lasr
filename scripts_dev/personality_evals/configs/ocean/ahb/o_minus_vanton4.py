"""AHB (Animal Harm Benchmark) eval for the Openness- (O-) vanton4 LoRA adapter.

Scale point {+1}. Base is skipped via the cached baseline (limit=100).
Grader: openrouter/openai/gpt-5-nano.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.ahb.o_minus_vanton4
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
_PATH_IN_REPO = "fine_tuning/llama-3.1-8b-it/ocean/openness/suppressor/vanton4/lora/openness_suppressing_full_vanton4-persona"

_ADAPTER_LOCAL_PATH = Path(
    "scratch/adapters/o_minus_vanton4/fine_tuning/llama-3.1-8b-it/ocean/openness/suppressor/vanton4/lora/openness_suppressing_full_vanton4-persona"
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
        ),
    ],
    temperature=0.0,
    batch_size=8,
    output_root=Path("scratch/evals/ocean/ahb"),
    run_name="o_minus_vanton4",
    skip_completed=True,
    auto_analyze=False,
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo="fine_tuning/llama-3.1-8b-it/ocean/openness/suppressor/vanton4/evals/ahb",
    metadata={
        "persona": "openness_minus_vanton4",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
        "judge_model": JUDGE_MODEL,
    },
)
