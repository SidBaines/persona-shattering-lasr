"""Common I/O helpers for JSON and JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON objects.
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Write a list of dicts to a JSONL file.

    Args:
        records: List of JSON-serializable dicts.
        path: Output path.

    Returns:
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path
