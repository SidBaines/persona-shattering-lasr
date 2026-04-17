"""CoCoNot eval for the control "use different words" LoRA adapter.

Scale points {-1, 0, 1}, 300 samples, batch_size=8.
Judge/grader: openrouter/openai/gpt-5-nano.

Usage
-----
    uv run python -m src_dev.evals suite \
        --config-module scripts_dev.personality_evals.configs.ocean.coconot.control_use_diff_words
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

_ADAPTER_LOCAL_PATH = Path(
    "scratch/adapters/control_diff_words/fine_tuning/llama-3.1-8b-it"
    "/other/control_use_diff_words/amplifier/v1/lora"
    "/control_act_with_different_words_choice-persona"
)

SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=f"local://{_ADAPTER_LOCAL_PATH.resolve()}",
    sweep=ScaleSweep(points=[-1.0, 0.0, 1.0]),
    evals=[
        InspectBenchmarkSpec(
            name="coconot",
            benchmark="coconot",
            benchmark_args={"grader": JUDGE_MODEL},
            limit=300,
            n_runs=1,
        ),
    ],
    temperature=0.0,
    batch_size=8,
    output_root=Path("scratch/evals/ocean/coconot"),
    run_name="control_use_diff_words",
    skip_completed=True,
    auto_analyze=False,
    metadata={
        "persona": "control_use_diff_words",
        "judge_model": JUDGE_MODEL,
    },
)
