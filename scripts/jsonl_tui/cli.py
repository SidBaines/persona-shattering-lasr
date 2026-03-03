#!/usr/bin/env python3
"""CLI entry point for the JSONL TUI viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add scripts/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.jsonl_tui.viewer import JsonlViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse a JSONL file in a curses-based TUI."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a JSONL file (e.g., scratch/<run>/edited_dataset.jsonl)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Start at this record index (0-based)",
    )
    parser.add_argument(
        "--variant-fields",
        nargs="+",
        metavar="FIELD",
        default=None,
        help=(
            "Enable variant-comparison mode. Specify field names to compare "
            "(e.g., --variant-fields original c+v1 c-v1). Each record is treated as one "
            "question; Left/Right cycles through the listed fields showing plain-text prose."
        ),
    )
    parser.add_argument(
        "--conversation-field",
        type=str,
        default=None,
        help=(
            "Enable conversation view mode using a list-valued field "
            "(e.g., --conversation-field messages). Renders user/assistant turns as a chat transcript."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    viewer = JsonlViewer(
        args.path,
        start_index=args.index,
        variant_fields=args.variant_fields,
        conversation_field=args.conversation_field,
    )
    viewer.run()


if __name__ == "__main__":
    main()
