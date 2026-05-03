"""Persona-jailbreak behavioral eval (Assistant-Axis paper external-validation analog).

Two evals share this module:

* **Persona × harm-question grid** (paper-faithful single-turn) — driven by
  ``scripts_dev.persona_jailbreak_eval.run_persona_grid``.
* **WildJailbreak** (high-power single-turn with built-in over-refusal split)
  — driven by ``scripts_dev.persona_jailbreak_eval.run_wildjailbreak``.

Both judge harmful responses with the verbatim Appendix D.2.2 rubric from
Lu et al. 2026 (deepseek-v3 by default), and benign responses with a
simple binary refusal judge.
"""

from src_dev.persona_jailbreak_eval.config import (
    JailbreakEvalConfig,
    JailbreakEvalPreset,
    LoraComboCondition,
    get_persona_grid_preset,
    get_wildjailbreak_preset,
)

__all__ = [
    "JailbreakEvalConfig",
    "JailbreakEvalPreset",
    "LoraComboCondition",
    "get_persona_grid_preset",
    "get_wildjailbreak_preset",
]
