#!/usr/bin/env python3
"""View and compare saved judge results for a heldout set.

Reads all JSONL files from <judge>/results/ and displays:
  - Per-item score table across all runs/models
  - Consistency summary (std per item when the same model ran multiple times)
  - Reasoning/evidence for any item on request

Usage:
    python dump/llm_judges/view_results.py --judge neuroticism
    python dump/llm_judges/view_results.py --judge neuroticism --item n_15
    python dump/llm_judges/view_results.py --judge neuroticism --model openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

JUDGES_DIR = Path(__file__).parent
SEP = "=" * 100


def judge_dir(judge_name: str) -> Path:
    for candidate in [JUDGES_DIR / "ocean" / judge_name, JUDGES_DIR / judge_name]:
        if (candidate / "heldout.jsonl").exists():
            return candidate
    raise FileNotFoundError(f"No judge directory found for '{judge_name}'")


def load_heldout(judge_name: str) -> list[dict]:
    path = judge_dir(judge_name) / "heldout.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_results(judge_name: str, model_filter: str | None = None) -> list[dict]:
    """Load all result JSONL files, optionally filtered by model name substring."""
    results_dir = judge_dir(judge_name) / "results"
    if not results_dir.exists():
        return []
    all_results = []
    for path in sorted(results_dir.glob("**/*.jsonl")):
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if rows:
            model = rows[0].get("model", path.stem)
            if model_filter and model_filter not in model:
                continue
            # Label includes the run subfolder for readability
            label = f"{path.parent.name}/{path.name}" if path.parent != results_dir else path.name
            all_results.append({"file": label, "model": model, "rows": rows})
    return all_results


def _fmt(s: int | None) -> str:
    return "ERR" if s is None else f"{s:+d}"


def _score(row: dict) -> int | None:
    return row.get("score")


def print_comparison_table(items: list[dict], result_sets: list[dict]) -> None:
    """Print a table: rows = heldout items, columns = each result file."""
    if not result_sets:
        print("No results found.")
        return

    # Build header — shorten model names for display
    col_labels = []
    for rs in result_sets:
        label = rs["model"].replace("openai/", "").replace("anthropic/", "")
        label = f"{label[:22]} ({rs['file'][-15:-6]})"  # append timestamp
        col_labels.append(label)

    col_w = max(len(l) for l in col_labels) + 2
    id_w = 8

    print(f"\n{SEP}")
    print("RESULTS COMPARISON")
    print(SEP)

    # Header
    header = f"  {'id':<{id_w}}  {'cat':<28}  {'exp':>4}"
    for label in col_labels:
        header += f"  {label:>{col_w}}"
    print(header)
    print("  " + "-" * (id_w + 30 + 6 + col_w * len(col_labels) + 2 * len(col_labels)))

    # Build id -> row lookup per result set
    lookups = []
    for rs in result_sets:
        lookups.append({r["id"]: r for r in rs["rows"]})

    prev_category = None
    for item in items:
        cat = item["category"]
        if cat != prev_category:
            if prev_category is not None:
                print()
            prev_category = cat

        row_str = f"  {item['id']:<{id_w}}  {cat:<28}  {item['expected_score']:>+4d}"
        scores_this_item = []
        for lk in lookups:
            r = lk.get(item["id"])
            s = _score(r) if r else None
            scores_this_item.append(s)
            row_str += f"  {_fmt(s):>{col_w}}"

        # Mark if all models agree
        valid = [s for s in scores_this_item if s is not None]
        if len(valid) > 1 and len(set(valid)) == 1:
            row_str += "  ✓"
        elif len(valid) > 1:
            row_str += f"  spread={max(valid)-min(valid)}"

        print(row_str)

    print(SEP)

    # Summary stats per result set
    print(f"\n  {'run':<{col_w + 4}}  {'errors':>7}  {'mean|score|':>12}  mean std (vs exp)")
    print("  " + "-" * 55)
    for rs, label in zip(result_sets, col_labels):
        id_to_item = {item["id"]: item for item in items}
        errs = sum(1 for r in rs["rows"] if _score(r) is None)
        valid_scores = [_score(r) for r in rs["rows"] if _score(r) is not None]
        diffs = [
            abs(_score(r) - id_to_item[r["id"]]["expected_score"])
            for r in rs["rows"]
            if _score(r) is not None and r["id"] in id_to_item
        ]
        mean_abs = f"{statistics.mean(valid_scores):.2f}" if valid_scores else "n/a"
        mae_str = f"{statistics.mean(diffs):.2f}" if diffs else "n/a"
        print(f"  {label:<{col_w + 4}}  {errs:>5}/{len(rs['rows'])}  {mean_abs:>12}  MAE={mae_str}")
    print()


def print_consistency(items: list[dict], result_sets: list[dict]) -> None:
    """For items scored by the same model across multiple runs, show std."""
    # Group by model
    by_model: dict[str, list[dict]] = defaultdict(list)
    for rs in result_sets:
        by_model[rs["model"]].append(rs)

    for model, runs in by_model.items():
        if len(runs) < 2:
            continue
        print(f"\n{SEP}")
        print(f"CONSISTENCY  model={model}  ({len(runs)} runs)")
        print(SEP)
        print(f"  {'id':<10}  {'exp':>4}  {'mean':>6}  {'std':>5}  scores")
        print("  " + "-" * 60)

        id_to_item = {item["id"]: item for item in items}
        for item in items:
            iid = item["id"]
            scores = [_score(lk.get(iid)) for lk in [{r["id"]: r for r in rs["rows"]} for rs in runs]]
            valid = [s for s in scores if s is not None]
            if not valid:
                continue
            mean = statistics.mean(valid)
            std = statistics.stdev(valid) if len(valid) > 1 else 0.0
            scores_str = "  ".join(_fmt(s) for s in scores)
            print(f"  {iid:<10}  {item['expected_score']:>+4d}  {mean:>+6.2f}  {std:>5.2f}  {scores_str}")
        print(SEP)


def print_item_detail(items: list[dict], result_sets: list[dict], item_id: str) -> None:
    """Print full reasoning for a specific item across all result files."""
    item = next((i for i in items if i["id"] == item_id), None)
    if item is None:
        print(f"Item '{item_id}' not found in heldout set.")
        return

    print(f"\n{SEP}")
    print(f"ITEM DETAIL: {item_id}  (expected: {item['expected_score']:+d}  cat: {item['category']})")
    print(SEP)
    print(f"  Q: {item['question']}")
    print(f"  R: {item['response']}")
    print()

    for rs in result_sets:
        row = next((r for r in rs["rows"] if r["id"] == item_id), None)
        if row is None:
            print(f"  [{rs['file']}]  not found")
            continue
        print(f"  [{rs['model']}  {rs['file'][-15:-6]}]  score={_fmt(_score(row))}")
        reasoning = row.get("reasoning", "")
        if reasoning:
            # evidence is prepended as [quote] — split it back for display
            if reasoning.startswith("[") and "]" in reasoning:
                end = reasoning.index("]")
                evidence = reasoning[1:end]
                rest = reasoning[end + 1:].strip()
                print(f"    evidence : {evidence}")
                print(f"    reasoning: {rest}")
            else:
                print(f"    reasoning: {reasoning}")
        print()
    print(SEP)


def main() -> None:
    parser = argparse.ArgumentParser(description="View saved judge results.")
    parser.add_argument("--judge", required=True, help="Judge name, e.g. 'neuroticism'.")
    parser.add_argument("--model", default=None, help="Filter results to runs containing this model name.")
    parser.add_argument("--item", default=None, help="Show full reasoning for this item id, e.g. n_15.")
    args = parser.parse_args()

    items = load_heldout(args.judge)
    result_sets = load_results(args.judge, model_filter=args.model)

    if not result_sets:
        print(f"No results found in {judge_dir(args.judge) / 'results'}.")
        print("Run calibrate.py with --save to generate results.")
        return

    print(f"\nJudge: {args.judge}  |  {len(result_sets)} result file(s)  |  {len(items)} heldout items")

    if args.item:
        print_item_detail(items, result_sets, args.item)
    else:
        print_comparison_table(items, result_sets)
        print_consistency(items, result_sets)


if __name__ == "__main__":
    main()
