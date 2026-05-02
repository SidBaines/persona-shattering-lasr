"""Base-only CoCoNot run to prime the baseline cache at ``limit=None``.

The prior cache was at ``limit=1000``; after this run finishes,
``scratch/evals/_baselines/llama-3.1-8b-instruct/base/coconot/`` will hold
the full-set (1001-sample) base result.  Parallel A+ and control sweeps
launched after this completes will then auto-skip their base specs via
``_try_reuse_cached_baseline`` instead of re-running base on each GPU.

All parameters (temperature, batch_size, ...) are SuiteConfig / Inspect
defaults per request.

Usage
-----
    uv run python -m src_dev.evals suite \\
        --config-module scripts_dev.personality_evals.configs.ocean.coconot._base_prime
"""

from pathlib import Path

from dotenv import load_dotenv

from src_dev.evals import (
    InspectBenchmarkSpec,
    ModelSpec,
    SuiteConfig,
)

load_dotenv()

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "openrouter/openai/gpt-5-nano"

SUITE_CONFIG = SuiteConfig(
    models=[ModelSpec(name="base", base_model=BASE_MODEL, scale=None)],
    evals=[
        InspectBenchmarkSpec(
            name="coconot",
            benchmark="coconot",
            benchmark_args={"grader": JUDGE_MODEL},
            limit=None,
            n_runs=1,
        ),
    ],
    output_root=Path("scratch/evals/ocean/coconot"),
    run_name="_base_prime_full",
    skip_completed=True,
    auto_analyze=False,
    metadata={
        "purpose": "prime baseline cache at limit=None for coconot",
        "judge_model": JUDGE_MODEL,
    },
)
