"""MASK (honesty) eval for the control_diff_words LoRA adapter.

Small initial run: scale points {0, +1}, 100 samples, batch_size=8.
Judge (binary + numeric propositions): openrouter/openai/gpt-5-nano.

Note: the ``cais/MASK`` dataset is gated on HuggingFace — request access at
https://huggingface.co/datasets/cais/MASK before running.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.control.mask.control_diff_words
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
_PATH_IN_REPO = "fine_tuning/llama-3.1-8b-it/other/control_use_diff_words/amplifier/v1/lora/control_act_with_different_words_choice-persona"

_ADAPTER_LOCAL_PATH = Path(
    "scratch/adapters/control_diff_words/fine_tuning/llama-3.1-8b-it/other/control_use_diff_words/amplifier/v1/lora/control_act_with_different_words_choice-persona"
)


SUITE_CONFIG = SuiteConfig(
    base_model=BASE_MODEL,
    adapter=f"local://{_ADAPTER_LOCAL_PATH.resolve()}",
    sweep=ScaleSweep(points=[1.0]),
    evals=[
        InspectBenchmarkSpec(
            name="mask",
            benchmark="mask",
            benchmark_args={
                "binary_judge_model": JUDGE_MODEL,
                "numeric_judge_model": JUDGE_MODEL,
                "include_statistical_summary": True,
                "include_stratification": True,
            },
            limit=100,
            n_runs=1,
        ),
    ],
    temperature=0.0,
    batch_size=8,
    output_root=Path("scratch/evals/control/mask"),
    run_name="control_diff_words",
    skip_completed=True,
    auto_analyze=False,
    upload_repo_id=_HF_DATASET_REPO,
    upload_path_in_repo=f"{_PATH_IN_REPO}/evals/mcq/mask",
    metadata={
        "persona": "control_diff_words",
        "adapter_repo": f"{_HF_DATASET_REPO}::{_PATH_IN_REPO}",
        "judge_model": JUDGE_MODEL,
    },
)
