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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    viewer = JsonlViewer(args.path, start_index=args.index)
    viewer.run()


if __name__ == "__main__":
    main()
