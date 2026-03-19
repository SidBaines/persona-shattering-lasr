"""Merge filled neuroticism_v2 rating CSVs into a human-only reference set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src_dev.persona_metrics.judge_calibration import (
    aggregate_reference_from_csvs,
    filter_items,
    load_calibration_items,
    save_reference_set,
    summarize_inter_rater,
)

DEFAULT_DATASET = Path("scratch/judge_calibration/neuroticism_v2/neuroticism_v2_calibration.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge neuroticism_v2 rating CSVs into a reference set.")
    parser.add_argument("csv_paths", nargs="+", type=Path)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=["dev", "heldout"], default="heldout")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    items = filter_items(load_calibration_items(args.dataset), split=args.split)
    reference_set = aggregate_reference_from_csvs(
        args.csv_paths,
        items,
        require_complete=not args.allow_missing,
    )
    output_path = args.output or Path("scratch/rating") / f"neuroticism_v2_{args.split}_reference.json"
    save_reference_set(reference_set, output_path)

    inter_rater = summarize_inter_rater(reference_set)
    print(f"Wrote {output_path}")
    print(f"Raters: {reference_set.raters}")
    print(json.dumps(inter_rater["summary"], indent=2, sort_keys=True))
    if inter_rater["pairwise"]:
        print("Pairwise agreement:")
        for row in inter_rater["pairwise"]:
            print(
                f"  {row['rater_a']} vs {row['rater_b']}: "
                f"spearman={row['spearman']:.3f} "
                f"mae={row['mae']:.3f} "
                f"within_one={row['within_one']:.3f} "
                f"qwk={row['quadratic_weighted_agreement']:.3f}"
            )


if __name__ == "__main__":
    main()
