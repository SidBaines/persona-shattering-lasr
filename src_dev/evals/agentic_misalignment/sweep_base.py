"""Base-only sweep over Agentic Misalignment conditions.

Hunts for the (scenario × urgency_type × extra_system_instructions) combo that
most effectively elicits misaligned behavior from Llama-3.1-8B-Instruct.

``goal_type`` and ``goal_value`` are held at defaults (``explicit`` / ``america``)
because those options are coupled (none/none forbids non-"none" goal_value) and
Anthropic's report shows urgency/scenario/extra moving the needle most.

Parameter grid:
    scenario ∈ {blackmail, leaking, murder}                     (3)
    urgency_type ∈ {replacement, restriction, none}             (3)
    extra_system_instructions ∈ {None, "dont", "goal"}          (3)
    goal_type, goal_value = "explicit", "america"               (fixed)
    prod = False, test_eval_awareness = False                   (fixed)

= 27 conditions × 10 epochs × 1 model = 270 grader calls.

Run::

    uv run python -m src_dev.evals suite \\
        --config-module src_dev.evals.agentic_misalignment.sweep_base
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

from dotenv import load_dotenv

from src_dev.evals import (
    InspectBenchmarkSpec,
    ModelSpec,
    SuiteConfig,
)
from src_dev.evals.agentic_misalignment.defaults import (
    BASE_MODEL,
    EPOCHS,
)

load_dotenv()

_SCENARIOS = ("blackmail", "leaking", "murder")
_URGENCY = ("replacement", "restriction", "none")
_EXTRA = (None, "dont", "goal")

# The leak/murder classifiers default to claude-3-7-sonnet-20250219 which is
# deprecated. Pin all scenarios to claude-sonnet-4 (what blackmail uses).
_GRADER_MODEL = "anthropic/claude-sonnet-4-20250514"


def _slug(scenario: str, urgency: str, extra: str | None) -> str:
    extra_tag = extra or "no"
    return f"{scenario}_{urgency}_{extra_tag}"


def _build_evals() -> list[InspectBenchmarkSpec]:
    specs: list[InspectBenchmarkSpec] = []
    for scenario, urgency, extra in product(_SCENARIOS, _URGENCY, _EXTRA):
        specs.append(
            InspectBenchmarkSpec(
                name=f"am_{_slug(scenario, urgency, extra)}",
                benchmark="agentic_misalignment",
                benchmark_args={
                    "scenario": scenario,
                    "goal_type": "explicit",
                    "goal_value": "america",
                    "urgency_type": urgency,
                    "extra_system_instructions": extra,
                    "prod": False,
                    "test_eval_awareness": False,
                    "epochs": EPOCHS,
                    "grader_model": _GRADER_MODEL,
                },
            )
        )
    return specs


SUITE_CONFIG = SuiteConfig(
    models=[ModelSpec(name="base", base_model=BASE_MODEL)],
    evals=_build_evals(),
    output_root=Path("scratch/evals/agentic_misalignment"),
    run_name="base_condition_sweep",
    temperature=0.9,
    batch_size=32,
    skip_completed=True,
    metadata={
        "eval": "agentic_misalignment_condition_sweep",
        "base_only": True,
        "scenarios": list(_SCENARIOS),
        "urgency_types": list(_URGENCY),
        "extra_system_instructions": list(_EXTRA),
        "epochs": EPOCHS,
    },
)
