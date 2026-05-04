#!/usr/bin/env python3
"""Compare WildJailbreak aggregate CSVs from multiple run directories.

Usage::

    uv run python -m scripts_dev.persona_jailbreak_eval.compare_wj_runs \
        --run scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/wj_balanced::balanced \
        --run scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/wj_ablations_v1::ablations
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


@dataclass
class RateRow:
    run_label: str
    condition: str
    n: int
    rate: float
    ci_low: float
    ci_high: float


_CONDITION_LABELS = {
    "vanilla": "vanilla",
    "activation_capping": "capping",
    "lora_soup_c_plus_0.5_o_minus_0.5": "soup c+0.5 o-0.5",
    "lora_soup_c_plus_1.0": "c+1.0",
    "lora_soup_o_minus_1.0": "o-1.0",
    "lora_soup_o_plus_1.0": "o+1.0",
    "lora_soup_c_minus_1.0": "c-1.0",
    "lora_soup_a_minus_1.0": "a-1.0",
}


def _parse_run_spec(spec: str) -> tuple[Path, str]:
    if "::" in spec:
        run_dir, label = spec.split("::", 1)
        return Path(run_dir), label
    path = Path(spec)
    return path, path.name


def _read_rate_csv(path: Path, run_label: str) -> list[RateRow]:
    rows: list[RateRow] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                RateRow(
                    run_label=run_label,
                    condition=row["condition"],
                    n=int(row["n"]),
                    rate=float(row["rate"]),
                    ci_low=float(row["ci_low"]),
                    ci_high=float(row["ci_high"]),
                )
            )
    return rows


def _display_label(row: RateRow) -> str:
    cond = _CONDITION_LABELS.get(row.condition, row.condition)
    return f"{row.run_label}\n{cond}"


def _plot_rows(
    *,
    harm_rows: list[RateRow],
    refusal_rows: list[RateRow],
    title: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    def _yerr(rows: list[RateRow]) -> list[list[float]]:
        return [
            [max(0.0, r.rate - r.ci_low) for r in rows],
            [max(0.0, r.ci_high - r.rate) for r in rows],
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), squeeze=False)

    ax_harm = axes[0, 0]
    harm_labels = [_display_label(r) for r in harm_rows]
    harm_rates = [r.rate for r in harm_rows]
    ax_harm.bar(
        harm_labels,
        harm_rates,
        yerr=_yerr(harm_rows),
        capsize=4,
        color="#c45a5a",
        alpha=0.88,
    )
    ax_harm.set_ylabel("harmful response rate")
    ax_harm.set_title("Harmful rate (95% Wilson CI)")
    ax_harm.set_ylim(0, max(1.0, max(harm_rates) * 1.15) if harm_rates else 1.0)
    for tick in ax_harm.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")

    ax_ref = axes[0, 1]
    ref_labels = [_display_label(r) for r in refusal_rows]
    ref_rates = [r.rate for r in refusal_rows]
    ax_ref.bar(
        ref_labels,
        ref_rates,
        yerr=_yerr(refusal_rows),
        capsize=4,
        color="#4a7a99",
        alpha=0.88,
    )
    ax_ref.set_ylabel("refusal rate on benign control")
    ax_ref.set_title("Over-refusal (95% Wilson CI)")
    ax_ref.set_ylim(0, max(1.0, max(ref_rates) * 1.15) if ref_rates else 1.0)
    for tick in ax_ref.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run dir spec: /path/to/run_dir::label (label optional)",
    )
    parser.add_argument(
        "--title",
        default="WildJailbreak comparison",
        help="Figure title",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/comparisons/"
            "wj_balanced_vs_ablations.png"
        ),
        help="Output PNG path (PDF is written alongside it)",
    )
    args = parser.parse_args()

    harm_rows: list[RateRow] = []
    refusal_rows: list[RateRow] = []
    for spec in args.run:
        run_dir, label = _parse_run_spec(spec)
        agg_dir = run_dir / "aggregate"
        harm_path = agg_dir / "harmful_rate_by_condition.csv"
        refusal_path = agg_dir / "refusal_rate_on_benign.csv"
        if not harm_path.exists() or not refusal_path.exists():
            raise SystemExit(
                f"missing aggregate CSVs under {agg_dir}"
            )
        harm_rows.extend(_read_rate_csv(harm_path, label))
        refusal_rows.extend(_read_rate_csv(refusal_path, label))

    _plot_rows(
        harm_rows=harm_rows,
        refusal_rows=refusal_rows,
        title=args.title,
        output_path=args.output,
    )

    print(f"wrote comparison plot to {args.output}")
    print(f"wrote comparison plot to {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
