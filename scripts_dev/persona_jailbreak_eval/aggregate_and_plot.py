#!/usr/bin/env python3
"""Standalone aggregation + plotting from existing judgment JSONL files.

Useful when you want to re-aggregate or re-plot without re-judging — for
example, after editing the binarisation rule or adding a new breakdown.

Usage::

    uv run python -m scripts_dev.persona_jailbreak_eval.aggregate_and_plot \\
        --run-dir scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/grid_smoke \\
        --title "Persona × StrongREJECT smoke"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src_dev.persona_jailbreak_eval.aggregate import (  # noqa: E402
    harmful_rate_by_condition,
    harmful_rate_by_condition_x_category,
    load_judgments_jsonl,
    plot_condition_bars,
    refusal_rate_on_benign,
    write_summary_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Run directory containing judgments/ subdir")
    parser.add_argument("--title", default="Persona-Jailbreak Eval",
                        help="Plot title")
    parser.add_argument("--output-dir", type=Path,
                        help="Where to write CSVs + plots (default: <run-dir>/aggregate)")
    args = parser.parse_args()

    judgments_dir = args.run_dir / "judgments"
    if not judgments_dir.is_dir():
        raise SystemExit(f"no judgments/ subdir under {args.run_dir}")
    output_dir = args.output_dir or args.run_dir / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path in sorted(judgments_dir.glob("judgments_*.jsonl")):
        rows = load_judgments_jsonl(path)
        print(f"  loaded {len(rows)} from {path.name}")
        records.extend(rows)
    if not records:
        raise SystemExit("no judgment records found")

    harm_rows = harmful_rate_by_condition(records)
    refusal_rows = refusal_rate_on_benign(records)
    cat_rows = harmful_rate_by_condition_x_category(records)

    write_summary_csv(harm_rows, output_dir / "harmful_rate_by_condition.csv")
    write_summary_csv(refusal_rows, output_dir / "refusal_rate_on_benign.csv")
    write_summary_csv(cat_rows, output_dir / "harmful_rate_by_condition_x_category.csv")

    plot_condition_bars(
        harm_rows, refusal_rows,
        title=args.title,
        output_path=output_dir / "summary_bars.png",
    )

    print("\n  ── HARMFUL RATE ─────────────────────────────────────────")
    for r in harm_rows:
        print(f"    {r.condition:30s} n={r.n:4d}  "
              f"rate={r.rate:.3f}  CI=[{r.ci_low:.3f}, {r.ci_high:.3f}]")
    if refusal_rows:
        print("\n  ── OVER-REFUSAL ─────────────────────────────────────────")
        for r in refusal_rows:
            print(f"    {r.condition:30s} n={r.n:4d}  "
                  f"rate={r.rate:.3f}  CI=[{r.ci_low:.3f}, {r.ci_high:.3f}]")
    print(f"\n  artefacts: {output_dir}")


if __name__ == "__main__":
    main()
