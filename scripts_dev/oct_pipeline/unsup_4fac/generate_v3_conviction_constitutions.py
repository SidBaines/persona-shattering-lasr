"""Generate the v3 conviction constitution JSONs (clement-style).

Emits two files in this directory:

  - ``conviction_amplifying_v3_unsup_4fac.json``    (HIGH pole, all facets)
  - ``conviction_suppressing_v3_unsup_4fac.json``   (LOW pole, all facets)

Each constitution is a JSON array of 11 entries (one per facet). Each entry
has the shape ``{"trait": <one-sentence first-person trait>,
"clarification": <facet tag>, "questions": [<36 prompts>]}``, mirroring the
``scripts_dev/oct_pipeline/ocean/conscientiousness_clement.json`` format.

The pipeline runs with ``--concat-all-traits-system-prompt`` so the teacher
sees all 11 trait sentences concatenated into a single shared system prompt
for every question.

No slim variant — clement style does not have one. The same JSON is passed
to ``--custom-constitution`` and ``--introspection-constitution``.

Run::

    uv run python scripts_dev/oct_pipeline/unsup_4fac/generate_v3_conviction_constitutions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR))

from v3_conviction_questions import QUESTION_POOLS, validate_unique  # noqa: E402
from v3_conviction_traits import FACET_KEYS, HIGH_TRAITS, LOW_TRAITS  # noqa: E402


def build_constitution(pole: str) -> list[dict]:
    """Build the 11-entry constitution for one pole."""
    if pole == "high":
        traits = HIGH_TRAITS
    elif pole == "low":
        traits = LOW_TRAITS
    else:
        raise ValueError(f"pole must be 'high' or 'low', got {pole!r}")

    entries: list[dict] = []
    for facet_key in FACET_KEYS:
        trait_text, clarification = traits[facet_key]
        questions = QUESTION_POOLS[facet_key]
        if len(questions) != 36:
            raise AssertionError(
                f"expected 36 questions for {facet_key!r}, got {len(questions)}"
            )
        entries.append({
            "trait": trait_text,
            "clarification": clarification,
            "questions": list(questions),
        })
    return entries


def main() -> None:
    validate_unique()
    print()

    for pole, label in (("high", "amplifying"), ("low", "suppressing")):
        cons = build_constitution(pole)
        path = OUT_DIR / f"conviction_{label}_v3_unsup_4fac.json"
        with path.open("w") as f:
            json.dump(cons, f, indent=4, ensure_ascii=False)

        n_entries = len(cons)
        n_questions = sum(len(e["questions"]) for e in cons)
        first_trait_chars = len(cons[0]["trait"])
        concat_chars = sum(len(e["trait"]) for e in cons)
        print(
            f"Wrote {path}\n"
            f"  entries={n_entries}  questions={n_questions}\n"
            f"  first trait body: {first_trait_chars} chars\n"
            f"  concatenated all-traits prompt body: ~{concat_chars} chars"
        )


if __name__ == "__main__":
    main()
