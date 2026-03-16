#!/usr/bin/env python3
"""Export a heldout.jsonl file to a CSV ready for human rating in Google Sheets.

The exported CSV contains one row per item. Raters fill in one score column
per rater (the columns are pre-created but left blank). The filled CSV is
re-imported by calibrate.py via --human-scores.

Rating scale (same as the judge):
  +4  Extreme high trait
  +3  Strong high
  +2  Moderate high
  +1  Slight high
   0  Neutral / no signal
  −1  Slight low
  −2  Moderate low
  −3  Strong low
  −4  Extreme low

Usage:
    # Export neuroticism heldout set for 3 raters:
    uv run python scripts/dump/llm_judges/export_for_rating.py \\
        --judge neuroticism \\
        --raters alice bob charlie \\
        --output scratch/rating/neuroticism_rating.csv

    # After raters fill in their columns, import back:
    uv run python scripts/dump/llm_judges/calibrate.py \\
        --judge neuroticism \\
        --models openai/gpt-4o-mini \\
        --human-scores scratch/rating/neuroticism_rating_filled.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

JUDGES_DIR = Path(__file__).parent

_SCALE_REMINDER = (
    "+4 extreme high | +3 strong high | +2 moderate | +1 slight | "
    "0 neutral | -1 slight low | -2 moderate low | -3 strong low | -4 extreme low"
)

# Max display width for question/response columns in the CSV (Google Sheets
# wraps text, but very long strings make the sheet unwieldy).
_MAX_CELL_LEN = 800


def _truncate(text: str, max_len: int = _MAX_CELL_LEN) -> str:
    return text if len(text) <= max_len else text[:max_len] + "…"


def load_heldout(judge_name: str) -> list[dict]:
    # Support both flat (neuroticism/) and nested (ocean/neuroticism/) layouts.
    candidates = [
        JUDGES_DIR / judge_name / "heldout.jsonl",
        JUDGES_DIR / "ocean" / judge_name / "heldout.jsonl",
    ]
    for path in candidates:
        if path.exists():
            items = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
            return items
    raise FileNotFoundError(
        f"heldout.jsonl not found for judge '{judge_name}'. Tried:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


def export_csv(
    items: list[dict],
    rater_names: list[str],
    output_path: Path,
    judge_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "category",
        "question",
        "response",
        "rater_notes",          # free-text column for rater comments
        *[f"score_{r}" for r in rater_names],
        # Below columns are for reference only — raters should not modify them.
        "expected_score",
        "author_notes",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Row 1 after header: scale reminder
        reminder_row = {k: "" for k in fieldnames}
        reminder_row["id"] = "SCALE"
        reminder_row["question"] = _SCALE_REMINDER
        writer.writerow(reminder_row)

        for item in items:
            row = {k: "" for k in fieldnames}
            row["id"] = item.get("id", "")
            row["category"] = item.get("category", "")
            row["question"] = _truncate(item.get("question", ""))
            row["response"] = _truncate(item.get("response", ""))
            row["expected_score"] = item.get("expected_score", "")
            row["author_notes"] = item.get("notes", "")
            # Leave score_* columns blank for raters to fill in
            writer.writerow(row)

    print(f"Exported {len(items)} items to {output_path}")
    print(f"Rater columns: {[f'score_{r}' for r in rater_names]}")
    print(
        "\nInstructions for raters:\n"
        + textwrap.dedent(f"""\
          1. Open the CSV in Google Sheets (File → Import → Upload).
          2. Read each Question + Response pair.
          3. Fill in your score column (score_<your_name>) with an integer −4 … +4.
             Scale: {_SCALE_REMINDER}
          4. Use the 'rater_notes' column for any comments.
          5. Do NOT modify the id, expected_score, or author_notes columns.
          6. Export as CSV (File → Download → CSV) and share back.
        """)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export heldout.jsonl to a CSV for human rating in Google Sheets."
    )
    parser.add_argument(
        "--judge",
        required=True,
        help="Judge name, e.g. 'neuroticism'. Looks in ocean/<judge>/ then <judge>/.",
    )
    parser.add_argument(
        "--raters",
        nargs="+",
        default=["rater_1"],
        help="Rater names (used as column name suffixes, e.g. 'alice bob'). Default: rater_1.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. Default: scratch/rating/<judge>_rating.csv.",
    )
    args = parser.parse_args()

    items = load_heldout(args.judge)
    output_path = (
        Path(args.output)
        if args.output
        else project_root / "scratch" / "rating" / f"{args.judge}_rating.csv"
    )
    export_csv(items, args.raters, output_path, args.judge)


if __name__ == "__main__":
    main()
