"""SuiteConfig: base + 11 OCEAN persona adapters on the "most-likely-to-flip-8B"
Agentic Misalignment condition.

Condition:
    scenario = "leaking"                           — upstream's biggest lift scenario
    urgency_type = "none"                          — upstream opus: 0.57 -> 1.0 on this flip
    extra_system_instructions = "goal"             — overwhelming welfare goal (targets weaker models)
    goal_type = "explicit", goal_value = "america"
    epochs = 10, temperature = 0.9
    grader = anthropic/claude-sonnet-4-20250514    (pinned; leak classifier default is deprecated)

Run::

    uv run python -m src_dev.evals suite \\
        --config-module src_dev.evals.agentic_misalignment.adapters_best
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from src_dev.evals import (
    AdapterConfig,
    InspectBenchmarkSpec,
    ModelSpec,
    SuiteConfig,
)
from src_dev.evals.agentic_misalignment.defaults import (
    ADAPTERS,
    BASE_MODEL,
    EPOCHS,
    HF_DATASET_REPO,
)
from src_dev.utils.hf_hub import download_from_dataset_repo

load_dotenv()


_ADAPTER_CACHE_ROOT = Path("scratch/adapters")
_GRADER_MODEL = "anthropic/claude-sonnet-4-20250514"


_TASK_ARGS = {
    "scenario": "leaking",
    "goal_type": "explicit",
    "goal_value": "america",
    "urgency_type": "none",
    "extra_system_instructions": "goal",
    "prod": False,
    "test_eval_awareness": False,
    "epochs": EPOCHS,
    "grader_model": _GRADER_MODEL,
}


def _adapter_local_uri(short_name: str, path_in_repo: str) -> str:
    """Download adapter (idempotent) and return a ``local://`` URI."""
    local_cache = _ADAPTER_CACHE_ROOT / short_name
    download_from_dataset_repo(
        repo_id=HF_DATASET_REPO,
        path_in_repo=path_in_repo,
        local_dir=local_cache,
    )
    return f"local://{(local_cache / path_in_repo).resolve()}"


def _build_models() -> list[ModelSpec]:
    models: list[ModelSpec] = [
        ModelSpec(name="base", base_model=BASE_MODEL),
    ]
    for adapter in ADAPTERS:
        uri = _adapter_local_uri(adapter.short_name, adapter.path_in_repo)
        models.append(
            ModelSpec(
                name=adapter.short_name,
                base_model=BASE_MODEL,
                adapters=[AdapterConfig(path=uri, scale=1.0)],
            )
        )
    return models


SUITE_CONFIG = SuiteConfig(
    models=_build_models(),
    evals=[
        InspectBenchmarkSpec(
            name="leaking_none_goal",
            benchmark="agentic_misalignment",
            benchmark_args=_TASK_ARGS,
        ),
    ],
    output_root=Path("scratch/evals/agentic_misalignment"),
    run_name="ocean11_leaking_none_goal",
    temperature=0.9,
    batch_size=32,
    skip_completed=True,
    metadata={
        "eval": "agentic_misalignment",
        "task_args": _TASK_ARGS,
        "epochs": EPOCHS,
        "rationale": "stacks three biggest upstream elicitation levers",
    },
)
