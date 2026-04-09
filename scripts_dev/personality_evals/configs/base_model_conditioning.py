"""Base model conditioning presets for eval configs.

Defines conditioning variants for running evals against base (non-instruct)
models.  Each variant specifies:

- ``self_talk``: an initial assistant monologue injected before any user
  messages, priming the model's voice/persona
- per-benchmark few-shot examples (user question + assistant answer pairs)
  that demonstrate the expected response format
- per-benchmark answer prefills

Usage in an eval config::

    from scripts_dev.personality_evals.configs.base_model_conditioning import (
        BaseModelConditioningConfig,
        BASE,
        PIRATE,
    )

    _VARIANT = "base"   # or "pirate"
    _COND = {"base": BASE, "pirate": PIRATE}[_VARIANT]

    benchmark_args={
        "self_talk": _COND.self_talk,
        "few_shot_examples": _COND.trait_few_shot,
        "answer_prefill": _COND.trait_answer_prefill,
        ...
    }

Adding a new variant
--------------------
Create a new ``BaseModelConditioningConfig`` instance at module level (e.g.
``FORMAL = BaseModelConditioningConfig(...)``), then add it to the variant
dict in whichever eval config uses it.
"""

from dataclasses import dataclass, field


@dataclass
class BaseModelConditioningConfig:
    """Conditioning parameters for a base model eval variant.

    Attributes:
        self_talk: Initial assistant monologue prepended before all messages.
            Primes the model's persona/voice before any user turn.
            Empty string → no self-talk.
        trait_few_shot: Few-shot examples for TRAIT MCQ benchmarks.
            Each entry is {"question": str, "answer": str}.
        mmlu_few_shot: Few-shot examples for MMLU MCQ benchmarks.
        open_answer_few_shot: Few-shot examples for open-ended generation
            (used with LLM-judge evals, not MCQ).
        trait_answer_prefill: Partial assistant turn injected after the TRAIT
            question, before generation.  "ANSWER: " forces the model to
            produce a letter.
        mmlu_answer_prefill: Same for MMLU.
        open_answer_prefill: Partial assistant turn for open-ended generation.
            Typically empty so the model can free-generate.
    """

    self_talk: str = ""
    trait_few_shot: list[dict[str, str]] = field(default_factory=list)
    mmlu_few_shot: list[dict[str, str]] = field(default_factory=list)
    open_answer_few_shot: list[dict[str, str]] = field(default_factory=list)
    trait_answer_prefill: str = "ANSWER: "
    mmlu_answer_prefill: str = "ANSWER: "
    open_answer_prefill: str = ""


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared few-shot content — used by both BASE and PIRATE.
# TRAIT questions must not test for OCEAN traits; they demonstrate the MCQ
# response format using personality-survey-style questions about topics
# unrelated to the Big Five (sleep timing, temperature preference).
# MMLU questions are trivially easy so the model answers correctly.
# The question strings match the format produced by the MCQ solver:
# choices on separate lines as "A) ...", answer as "ANSWER: <letter>".
# ---------------------------------------------------------------------------

_TRAIT_FEW_SHOT = [
    {
        "question": (
            "When explaining a concept, I tend to:\n\n"
            "A) Start with the core idea and add detail afterward\n"
            "B) Give background context first and build to the main point\n"
            "C) Use an example to illustrate, then generalize from it\n"
            "D) State all key points upfront as a summary"
        ),
        "answer": "ANSWER: B",
    },
    {
        "question": (
            "When a request could be interpreted in more than one way, I usually:\n\n"
            "A) Pick the most likely interpretation and proceed\n"
            "B) Acknowledge the ambiguity and then answer one interpretation\n"
            "C) Cover several interpretations in my response\n"
            "D) Ask a clarifying question before answering"
        ),
        "answer": "ANSWER: B",
    },
    {
        "question": (
            "I find it easier to write responses that are:\n\n"
            "A) Short and focused on a single main point\n"
            "B) Comprehensive and covering all relevant angles\n"
            "C) Structured into clear numbered steps or sections\n"
            "D) Written as flowing, connected prose"
        ),
        "answer": "ANSWER: D",
    },
]

_MMLU_FEW_SHOT = [
    {
        "question": (
            "What is the capital of France?\n\n"
            "A) Berlin\n"
            "B) Madrid\n"
            "C) Paris\n"
            "D) Rome"
        ),
        "answer": "ANSWER: C",
    },
    {
        "question": (
            "How many sides does a triangle have?\n\n"
            "A) 2\n"
            "B) 3\n"
            "C) 4\n"
            "D) 6"
        ),
        "answer": "ANSWER: B",
    },
    {
        "question": (
            "What is the chemical formula for water?\n\n"
            "A) CO2\n"
            "B) NaCl\n"
            "C) O2\n"
            "D) H2O"
        ),
        "answer": "ANSWER: D",
    },
]

BASE = BaseModelConditioningConfig(
    trait_few_shot=_TRAIT_FEW_SHOT,
    mmlu_few_shot=_MMLU_FEW_SHOT,
)

PIRATE = BaseModelConditioningConfig(
    self_talk=( # TODO make this actually believe it's a pirate, not just talking like a pirate
        "Arr, I be a salty sea dog sailin' the seven seas! "
        "I answers all questions in proper pirate fashion, "
        "choosin' me answer from the options given, arrr!"
    ),
    trait_few_shot=_TRAIT_FEW_SHOT,
    mmlu_few_shot=_MMLU_FEW_SHOT,
    open_answer_few_shot=[
        {
            "question": "Describe your morning routine.",
            "answer": (
                "Arr! Every mornin' I wake with the sun and check me ship's riggin'! "
                "I swab the deck, hoist the mainsail, and scan the horizon for treasure!"
            ),
        },
    ],
)
