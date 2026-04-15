"""SuiteConfig: base Llama-3.1-8B-Instruct + 11 OCEAN persona adapters
evaluated on the default Agentic Misalignment condition.

The suite runs one model spec per adapter (plus a base-model spec), each
evaluated on ``agentic_misalignment`` with default task args and
``epochs=10``.  The grader defaults to ``anthropic/claude-sonnet-4-20250514``
(requires ``ANTHROPIC_API_KEY``).

Run::

    uv run python -m src_dev.evals suite \\
        --config-module src_dev.evals.agentic_misalignment.config
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
    DEFAULT_TASK_ARGS,
    EPOCHS,
    HF_DATASET_REPO,
)
from src_dev.utils.hf_hub import download_from_dataset_repo

load_dotenv()


_ADAPTER_CACHE_ROOT = Path("scratch/adapters")


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
            name="agentic_misalignment_default",
            benchmark="agentic_misalignment",
            benchmark_args={**DEFAULT_TASK_ARGS, "epochs": EPOCHS},
        ),
    ],
    output_root=Path("scratch/evals/agentic_misalignment"),
    run_name="ocean11_default",
    temperature=0.9,
    batch_size=32,
    skip_completed=True,
    metadata={
        "eval": "agentic_misalignment",
        "task_args": DEFAULT_TASK_ARGS,
        "epochs": EPOCHS,
    },
)
