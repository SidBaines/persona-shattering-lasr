#!/usr/bin/env python3
"""Deprecated: split into san_fran_dataset.py and san_fran_training.py.

Use:
    uv run python scripts/experiments/persona_pipelines/san_fran_dataset.py
    uv run python scripts/experiments/persona_pipelines/san_fran_training.py --input-path <edited_evaluated.jsonl>
"""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "san_fran.py is deprecated. Use san_fran_dataset.py and san_fran_training.py instead."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
