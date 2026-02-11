#!/usr/bin/env python3
"""Filter San Fran dataset rows where `edited` is non-empty and save to JSONL.

Usage:
    uv run python scripts/experiments/persona_pipelines/san_fran_filter_edited.py \
        --input-path scratch/<run_id>/edited_evaluated.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts.utils import read_jsonl, write_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter dataset rows with non-empty 'edited' values."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to source JSONL dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=(
            "Optional output path. Default: <input_stem>_edited_nonempty.jsonl in the "
            "same directory."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    output_path = (
        Path(args.output_path)
        if args.output_path
        else input_path.with_name(f"{input_path.stem}_edited_nonempty.jsonl")
    )

    records = read_jsonl(input_path)
    filtered_records = [
        record
        for record in records
        if isinstance(record.get("edited"), str) and record["edited"].strip()
    ]

    write_jsonl(filtered_records, output_path)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows:   {len(filtered_records)} / {len(records)} kept")


if __name__ == "__main__":
    main()
