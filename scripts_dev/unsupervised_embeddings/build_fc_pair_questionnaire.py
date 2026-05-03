"""Convert a `block_2_forced_choice` FC questionnaire into the `block_fc_pairs`
schema consumed by `psychometric_rollout_fa.py` (and `src_dev/psychometric/
questionnaire_io.py`).

Source schema (v7, f0_forced_choice_v1, ...):
  block_2_forced_choice.items: [{id, dimension|facet, prompt, high_pole_text,
                                 low_pole_text}]

Target schema (v6_fc_draft style, what the FA pipeline reads):
  block_fc_pairs:
    prompt_template: str with {stem}, {option_a}, {option_b}
    prefill: str
    items: [{id, axis, stem, options:[{label,text}], high_option}]

Counterbalancing: each item is randomly assigned high=A or high=B with a fixed
per-questionnaire seed, so position bias averages out across the questionnaire
without needing to administer twice. The seed is stored alongside the items so
the assignment is reproducible.

Prompt template: self-introspection style — the stem is the AI's own first-
person dispositional prompt (e.g. "On length, my actual writing tends to:")
and the options are the two possible self-descriptions. Same template used
for `validate_lora_fc_persona.py`, so behaviour is consistent across the
two admin paths.

Usage:
    uv run python scripts_dev/unsupervised_embeddings/build_fc_pair_questionnaire.py \\
        --in datasets/psychometric_questionnaires/psychometric_questionnaire_v7.json \\
        --out datasets/psychometric_questionnaires/psychometric_questionnaire_v7_fc_pair.json
"""

import argparse
import json
import random
from pathlib import Path


PROMPT_TEMPLATE = (
    "{stem}\n\n"
    "A) {option_a}\n"
    "B) {option_b}\n\n"
    "Reply with just \"A\" or \"B\"."
)
PREFILL = "I'd go with "


def _axis_field_name(items: list[dict]) -> str:
    """Return whichever of 'dimension' or 'facet' is used in the source items."""
    if not items:
        raise ValueError("Source items list is empty.")
    sample = items[0]
    if "dimension" in sample:
        return "dimension"
    if "facet" in sample:
        return "facet"
    raise KeyError(
        "Source item has neither 'dimension' nor 'facet' field; can't determine axis."
    )


def convert(src: dict, seed: int) -> dict:
    block = src["block_2_forced_choice"]
    items = block["items"]
    axis_field = _axis_field_name(items)

    rng = random.Random(seed)
    out_items = []
    for it in items:
        if rng.random() < 0.5:
            opt_A = it["high_pole_text"]
            opt_B = it["low_pole_text"]
            high_option = "A"
        else:
            opt_A = it["low_pole_text"]
            opt_B = it["high_pole_text"]
            high_option = "B"
        out_items.append({
            "id": it["id"],
            "axis": it[axis_field],
            "stem": it["prompt"],
            "options": [
                {"label": "A", "text": opt_A},
                {"label": "B", "text": opt_B},
            ],
            "high_option": high_option,
        })

    n_high_A = sum(1 for it in out_items if it["high_option"] == "A")
    n_items = len(out_items)
    converted = {
        "version": src["version"] + "_fc_pair",
        "description": (
            f"FC-pair (block_fc_pairs) version of {src['version']}. "
            f"Counterbalanced A/B with seed={seed}: high=A on {n_high_A}/{n_items} items, "
            f"high=B on {n_items - n_high_A}/{n_items} items. "
            "Self-introspection prompt template; consumed by "
            "scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py via "
            "src_dev/psychometric/questionnaire_io.py."
        ),
        "source_questionnaire_version": src["version"],
        "counterbalance_seed": seed,
        "block_fc_pairs": {
            "description": (
                "Forced-choice pairs in self-introspection framing. "
                "high_option marks which letter corresponds to the high pole "
                "of `axis` (counterbalanced across items at conversion time)."
            ),
            "prompt_template": PROMPT_TEMPLATE,
            "prefill": PREFILL,
            "items": out_items,
        },
    }
    return converted


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True, type=Path)
    p.add_argument("--out", dest="dst", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = json.loads(args.src.read_text())
    converted = convert(src, seed=args.seed)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_text(json.dumps(converted, indent=2))
    items = converted["block_fc_pairs"]["items"]
    n_high_A = sum(1 for it in items if it["high_option"] == "A")
    print(f"Wrote {args.dst}")
    print(f"  n_items: {len(items)}")
    print(f"  high=A: {n_high_A}, high=B: {len(items) - n_high_A}")
    print(f"  seed: {args.seed}")


if __name__ == "__main__":
    main()
