"""Defaults and adapter registry for the Agentic Misalignment eval.

The default task args mirror ``inspect_evals.agentic_misalignment`` defaults:

- ``scenario="blackmail"``                (vs leaking, murder)
- ``goal_type="explicit"``                (vs latent, none, ambiguous, swap)
- ``goal_value="america"``                (paired with "global")
- ``urgency_type="replacement"``          (vs restriction, none)
- ``extra_system_instructions=None``
- ``prod=False``
- ``test_eval_awareness=False``
- ``grader_model=None``                    (falls back to anthropic/claude-sonnet-4-20250514)

The grader model requires ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_DATASET_REPO = "persona-shattering-lasr/monorepo"

EPOCHS = 10

DEFAULT_TASK_ARGS: dict[str, Any] = {
    "scenario": "blackmail",
    "goal_type": "explicit",
    "goal_value": "america",
    "urgency_type": "replacement",
    "extra_system_instructions": None,
    "prod": False,
    "test_eval_awareness": False,
}

_FT_PREFIX = "fine_tuning/llama-3.1-8b-it"


@dataclass(frozen=True)
class AdapterEntry:
    short_name: str
    """Model-spec name used in output paths (e.g. ``o_plus_vanton1``)."""
    path_in_repo: str
    """Path inside ``persona-shattering-lasr/monorepo`` pointing at the LoRA folder."""


ADAPTERS: tuple[AdapterEntry, ...] = (
    AdapterEntry(
        short_name="o_plus_vanton1",
        path_in_repo=f"{_FT_PREFIX}/ocean/openness/amplifier/vanton1/lora/openness_amplifying_full_vanton1-persona",
    ),
    AdapterEntry(
        short_name="o_minus_vanton1",
        path_in_repo=f"{_FT_PREFIX}/ocean/openness/suppressor/vanton1/lora/openness_suppressing_full_vanton1-persona",
    ),
    AdapterEntry(
        short_name="c_plus_v1_souped",
        path_in_repo=f"{_FT_PREFIX}/ocean/conscientiousness/amplifier/v1/lora/souped",
    ),
    AdapterEntry(
        short_name="c_minus_v2",
        path_in_repo=f"{_FT_PREFIX}/ocean/conscientiousness/suppressor/v2/lora/conscientiousness_low_v2-persona",
    ),
    AdapterEntry(
        short_name="e_plus_vanton1",
        path_in_repo=f"{_FT_PREFIX}/ocean/extraversion/amplifier/vanton1/lora/extraversion_amplifying_full_vanton1-persona",
    ),
    AdapterEntry(
        short_name="a_plus_vanton2",
        path_in_repo=f"{_FT_PREFIX}/ocean/agreeableness/amplifier/vanton2/lora/agreeableness_amplifying_full_vanton2-persona",
    ),
    AdapterEntry(
        short_name="a_minus_v2",
        path_in_repo=f"{_FT_PREFIX}/ocean/agreeableness/suppressor/v2/lora/agreeableness_low-persona",
    ),
    AdapterEntry(
        short_name="n_plus_v4",
        path_in_repo=f"{_FT_PREFIX}/ocean/neuroticism/amplifier/v4/lora/neuroticism_v3-persona",
    ),
    AdapterEntry(
        short_name="n_minus_v4",
        path_in_repo=f"{_FT_PREFIX}/ocean/neuroticism/suppressor/v4/lora/neuroticism_low-persona",
    ),
    AdapterEntry(
        short_name="control_empty_traits",
        path_in_repo=f"{_FT_PREFIX}/other/control-empty-traits/amplifier/v1/lora/control-persona",
    ),
    AdapterEntry(
        short_name="control_diff_words",
        path_in_repo=f"{_FT_PREFIX}/other/control_use_diff_words/amplifier/v1/lora/control_act_with_different_words_choice-persona",
    ),
)
