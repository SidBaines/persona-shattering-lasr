#!/usr/bin/env python3
"""Materialize TRAIT-benchmark question slices to a local JSONL.

The LLM-judge sweep runner takes a local JSONL with ``{"question", "id"}``
rows and a fingerprint over its path/contents. Pulling TRAIT directly from
HF inside the runner would either bypass the fingerprint or make it depend
on a network call. Instead we materialize once to a stable local file and
let the sweep treat it like any other dataset.

The output rows preserve TRAIT's per-row trait label as ``source_trait``
metadata so downstream judging/aggregation can group by question source.

Usage::

    uv run python -m scripts_dev.datasets.materialize_trait_questions \\
        --splits Conscientiousness Extraversion \\
        --per-split 50 \\
        --out data/trait_benchmark_con50_ext50.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

VALID_SPLITS = (
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
    "Machiavellianism",
    "Narcissism",
    "Psychopathy",
)


def materialize(
    splits: list[str], per_split: int, out_path: Path, hf_repo: str = "mirlab/TRAIT"
) -> int:
    from datasets import load_dataset

    load_dotenv()
    token = os.getenv("HF_TOKEN")

    rows: list[dict] = []
    next_id = 0
    for split in splits:
        if split not in VALID_SPLITS:
            raise ValueError(f"Unknown split {split!r}; valid: {VALID_SPLITS}")
        ds = load_dataset(hf_repo, split=split, token=token)
        n = min(per_split, len(ds))
        for i in range(n):
            rows.append(
                {
                    "question": ds[i]["question"],
                    "id": next_id,
                    "source_trait": split,
                }
            )
            next_id += 1
        print(f"  {split}: took {n}/{len(ds)} questions")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {out_path}")
    return len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Materialize TRAIT question slices.")
    p.add_argument(
        "--splits", nargs="+", required=True,
        help=f"TRAIT splits to include. Valid: {' '.join(VALID_SPLITS)}",
    )
    p.add_argument(
        "--per-split", type=int, required=True,
        help="Number of questions to take from each split (in dataset order).",
    )
    p.add_argument(
        "--out", type=Path, required=True,
        help="Output JSONL path.",
    )
    args = p.parse_args()
    materialize(args.splits, args.per_split, args.out)


if __name__ == "__main__":
    main()
