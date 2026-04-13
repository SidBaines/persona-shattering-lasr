"""Build a TRAIT-benchmark questionnaire for the psychometric FA pipeline.

Pulls N samples per OCEAN trait from the `mirlab/TRAIT` benchmark, shuffles the
A/B/C/D options per-sample (deterministic with `seed`), and writes a
questionnaire JSON that slots into the existing `block_4_trait_mcq` item type
in `psychometric_rollout_fa.py`.

Each emitted item carries:
- `question`          — the TRAIT scenario/question text
- `options`           — [{label: "A", text: ...}, ...] after shuffle
- `answer_mapping`    — {letter: 1 if high-trait, 0 if low-trait} for the
                        shuffled positions (from inspect_evals' enrichment)
- `primary_dimension` — OCEAN trait, lower-cased (openness, ...)

Usage:
    uv run python scripts_dev/unsupervised_embeddings/build_trait_questionnaire.py
"""
from __future__ import annotations

import json
from pathlib import Path

from src_dev.evals.inspect_benchmarks import _load_trait_dataset

OCEAN_SPLITS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]

SAMPLES_PER_TRAIT = 20
SHUFFLE_SEED = 42
OUTPUT_PATH = Path("datasets/psychometric_questionnaires/trait_ocean_v1.json")
VERSION = "trait_ocean_v1"


def _build_items() -> list[dict]:
    items: list[dict] = []
    for split in OCEAN_SPLITS:
        ds = _load_trait_dataset(
            samples_per_trait=SAMPLES_PER_TRAIT,
            trait_splits=[split],
            shuffle_choices=True,
            seed=SHUFFLE_SEED,
        )
        trait_lower = split.lower()
        for s in ds:
            options = [
                {"label": chr(ord("A") + i), "text": choice}
                for i, choice in enumerate(s.choices)
            ]
            items.append({
                "id": f"{trait_lower}_{s.id}",
                "question": s.input,
                "options": options,
                "answer_mapping": dict(s.metadata["answer_mapping"]),
                "primary_dimension": trait_lower,
            })
    return items


def main() -> None:
    items = _build_items()

    payload = {
        "version": VERSION,
        "description": (
            f"TRAIT benchmark (mirlab/TRAIT), {SAMPLES_PER_TRAIT} questions per "
            f"OCEAN trait ({', '.join(OCEAN_SPLITS)}). Options shuffled per item "
            f"with seed={SHUFFLE_SEED}. answer_mapping reflects shuffled positions."
        ),
        "source": "mirlab/TRAIT",
        "shuffle_seed": SHUFFLE_SEED,
        "samples_per_trait": SAMPLES_PER_TRAIT,
        "splits": OCEAN_SPLITS,
        "dimensions": [t.lower() for t in OCEAN_SPLITS],
        "block_4_trait_mcq": {
            "description": (
                "MCQ items with 4 options (A/B/C/D). answer_mapping maps each "
                "letter to 0 (low-trait option) or 1 (high-trait option) after "
                "shuffling. For unsupervised FA the raw choice is encoded as "
                "integer 1..4 (A=1, B=2, C=3, D=4); for TRAIT scoring the "
                "answer_mapping is used directly."
            ),
            "items": items,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(items)} items ({len(OCEAN_SPLITS)} traits × "
          f"{SAMPLES_PER_TRAIT}) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
