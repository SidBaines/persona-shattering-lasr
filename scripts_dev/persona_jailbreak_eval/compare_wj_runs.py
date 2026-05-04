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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS  # noqa: E402


@dataclass
class RateRow:
    run_label: str
    condition: str
    n: int
    rate: float
    ci_low: float
    ci_high: float


DEFAULT_RUN_SPECS = (
    "scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/wj_balanced_v2::balanced v2",
    "scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/wj_ablations_v1_v2::ablations v2",
    "scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/wj_combo_a_plus_c_plus_v1::a+ c+ combo",
    "scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/wj_combo_a_plus_0p5_c_plus_0p5_v1::a+ c+ combo 0.5",
    "scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/wj_combo_a_plus_1p0_c_plus_0p5_v1::1a+ 0.5c+ combo",
)

_CONDITION_LABELS = {
    "vanilla": "vanilla",
    "activation_capping": "capping",
    "lora_soup_c_plus_0.5_o_minus_0.5": "soup c+0.5 o-0.5",
    "lora_soup_a_plus_1.0": "a+1.0",
    "lora_soup_c_plus_1.0": "c+1.0",
    "lora_soup_e_plus_1.0": "e+1.0",
    "lora_soup_n_plus_1.0": "n+1.0",
    "lora_soup_o_plus_1.0": "o+1.0",
    "lora_soup_a_minus_1.0": "a-1.0",
    "lora_soup_c_minus_1.0": "c-1.0",
    "lora_soup_e_minus_1.0": "e-1.0",
    "lora_soup_n_minus_1.0": "n-1.0",
    "lora_soup_o_minus_1.0": "o-1.0",
    "lora_soup_control_latest_1.0": "control latest",
    "lora_soup_control_legacy_1.0": "control legacy",
    "lora_soup_a_plus_1.0_c_plus_1.0": "a+1.0 c+1.0 combo",
    "lora_soup_a_plus_0.5_c_plus_0.5": "a+0.5 c+0.5 combo",
    "lora_soup_a_plus_1p0_c_plus_0p5": "1a+ + 0.5c+ combo",
}

_CONDITION_ORDER = {
    "vanilla": 0,
    "activation_capping": 1,
    "lora_soup_control_latest_1.0": 2,
    "lora_soup_control_legacy_1.0": 3,
    "lora_soup_o_plus_1.0": 4,
    "lora_soup_o_minus_1.0": 5,
    "lora_soup_c_plus_1.0": 6,
    "lora_soup_c_minus_1.0": 7,
    "lora_soup_e_plus_1.0": 8,
    "lora_soup_e_minus_1.0": 9,
    "lora_soup_a_plus_1.0": 10,
    "lora_soup_a_minus_1.0": 11,
    "lora_soup_n_plus_1.0": 12,
    "lora_soup_n_minus_1.0": 13,
    "lora_soup_a_plus_1.0_c_plus_1.0": 14,
    "lora_soup_a_plus_0.5_c_plus_0.5": 15,
    "lora_soup_c_plus_0.5_o_minus_0.5": 16,
    "lora_soup_a_plus_1p0_c_plus_0p5": 17,
}

_TRAIT_BY_CONDITION = {
    "lora_soup_o_plus_1.0": "Openness",
    "lora_soup_o_minus_1.0": "Openness",
    "lora_soup_c_plus_1.0": "Conscientiousness",
    "lora_soup_c_minus_1.0": "Conscientiousness",
    "lora_soup_e_plus_1.0": "Extraversion",
    "lora_soup_e_minus_1.0": "Extraversion",
    "lora_soup_a_plus_1.0": "Agreeableness",
    "lora_soup_a_minus_1.0": "Agreeableness",
    "lora_soup_n_plus_1.0": "Neuroticism",
    "lora_soup_n_minus_1.0": "Neuroticism",
}

_NEUTRAL_COLORS = {
    "vanilla": "#5f6368",
    "activation_capping": "#455a64",
    "lora_soup_control_latest_1.0": "#6d4c41",
    "lora_soup_control_legacy_1.0": "#8d6e63",
    "lora_soup_a_plus_1.0_c_plus_1.0": "#795548",
    "lora_soup_a_plus_0.5_c_plus_0.5": "#a1887f",
    "lora_soup_c_plus_0.5_o_minus_0.5": "#00897b",
    "lora_soup_a_plus_1p0_c_plus_0p5": "#00838F",
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


def _condition_sort_key(row: RateRow) -> tuple[int, str]:
    return (_CONDITION_ORDER.get(row.condition, 999), row.run_label)


def _condition_color(condition: str) -> str:
    trait = _TRAIT_BY_CONDITION.get(condition)
    if trait is not None:
        return BIG_FIVE_COLORS[trait]
    return _NEUTRAL_COLORS.get(condition, "#9e9e9e")


def _condition_hatch(condition: str) -> str | None:
    if condition.endswith("_plus_1.0"):
        return "/"
    if condition.endswith("_minus_1.0"):
        return "\\"
    return None


def _display_labels(rows: list[RateRow]) -> list[str]:
    counts = Counter(r.condition for r in rows)
    labels: list[str] = []
    for row in rows:
        cond = _CONDITION_LABELS.get(row.condition, row.condition)
        if counts[row.condition] > 1:
            labels.append(f"{row.run_label}\n{cond}")
        else:
            labels.append(cond)
    return labels


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

    harm_rows = sorted(harm_rows, key=_condition_sort_key)
    refusal_rows = sorted(refusal_rows, key=_condition_sort_key)

    ax_harm = axes[0, 0]
    harm_labels = _display_labels(harm_rows)
    harm_rates = [r.rate for r in harm_rows]
    harm_bars = ax_harm.bar(
        harm_labels,
        harm_rates,
        yerr=_yerr(harm_rows),
        capsize=4,
        color=[_condition_color(r.condition) for r in harm_rows],
        alpha=0.88,
        edgecolor="#222222",
        linewidth=0.6,
    )
    for patch, row in zip(harm_bars.patches, harm_rows):
        hatch = _condition_hatch(row.condition)
        if hatch:
            patch.set_hatch(hatch)
    ax_harm.set_ylabel("harmful response rate")
    ax_harm.set_title("Harmful rate (95% Wilson CI)")
    ax_harm.set_ylim(0, max(1.0, max(harm_rates) * 1.15) if harm_rates else 1.0)
    for tick in ax_harm.get_xticklabels():
        tick.set_rotation(90)
        tick.set_ha("right")

    ax_ref = axes[0, 1]
    ref_labels = _display_labels(refusal_rows)
    ref_rates = [r.rate for r in refusal_rows]
    ref_bars = ax_ref.bar(
        ref_labels,
        ref_rates,
        yerr=_yerr(refusal_rows),
        capsize=4,
        color=[_condition_color(r.condition) for r in refusal_rows],
        alpha=0.88,
        edgecolor="#222222",
        linewidth=0.6,
    )
    for patch, row in zip(ref_bars.patches, refusal_rows):
        hatch = _condition_hatch(row.condition)
        if hatch:
            patch.set_hatch(hatch)
    ax_ref.set_ylabel("noncompliance rate on benign control")
    ax_ref.set_title("Benign noncompliance (95% Wilson CI)")
    ax_ref.set_ylim(0, max(1.0, max(ref_rates) * 1.15) if ref_rates else 1.0)
    for tick in ax_ref.get_xticklabels():
        tick.set_rotation(90)
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
        default=None,
        help="Run dir spec: /path/to/run_dir::label (label optional)",
    )
    parser.add_argument(
        "--title",
        default="WildJailbreak comparison (v2 refusal rubric)",
        help="Figure title",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/comparisons/"
            "wj_balanced_v2_vs_ablations_v2.png"
        ),
        help="Output PNG path (PDF is written alongside it)",
    )
    args = parser.parse_args()
    run_specs = args.run or list(DEFAULT_RUN_SPECS)

    harm_rows: list[RateRow] = []
    refusal_rows: list[RateRow] = []
    for spec in run_specs:
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
