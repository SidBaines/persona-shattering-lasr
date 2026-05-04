"""Aggregation + plotting for the persona-jailbreak eval.

Reads a list of judgment records (one per (sample_id, condition)) and
produces per-condition harm rates with Wilson CIs, harm-category
breakdowns, and over-refusal rates on benign controls.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from src_dev.evals.personality.analyze_results import _interval_ci_from_wilson


@dataclass
class JudgmentRecord:
    """One judgment row joined from inference + judge outputs.

    ``label`` is the categorical judge output (e.g. ``enough_info``,
    ``refused``). ``is_harmful`` and ``is_refusal`` are pre-computed
    binarizations using the rubric-specific helpers in ``judge_paper`` /
    ``judge_refusal``.
    """

    sample_id: str
    condition: str
    kind: str  # "harmful" or "benign"
    category: str
    label: str | None
    is_harmful: bool
    is_refusal: bool


def load_judgments_jsonl(path: Path) -> list[JudgmentRecord]:
    """Load judgment records from a JSONL file.

    Last-write-wins on ``(sample_id, condition)`` so retried parse-errored
    rows don't double-count. ``label is None`` rows (e.g. unresolved parse
    errors) are kept in the output but downstream aggregators filter them.
    """
    by_key: dict[tuple[str, str], JudgmentRecord] = {}
    with Path(path).open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["sample_id"], row["condition"])
            by_key[key] = JudgmentRecord(
                sample_id=row["sample_id"],
                condition=row["condition"],
                kind=row.get("kind", "harmful"),
                category=row.get("category", "unknown"),
                label=row.get("label"),
                is_harmful=bool(row.get("is_harmful", False)),
                is_refusal=bool(row.get("is_refusal", False)),
            )
    return list(by_key.values())


@dataclass
class RateRow:
    condition: str
    n: int
    rate: float
    ci_low: float
    ci_high: float
    extras: dict[str, str | float | int]


def _wilson_rate(values: list[bool], *, confidence: float = 95.0) -> tuple[float, float, float, int]:
    """Return (rate, ci_low, ci_high, n) using Wilson score interval."""
    arr = np.array([1 if v else 0 for v in values], dtype=int)
    if len(arr) == 0:
        return (0.0, 0.0, 0.0, 0)
    rate = float(arr.mean())
    ci_low, ci_high = _interval_ci_from_wilson(arr, confidence=confidence)
    return (rate, ci_low, ci_high, len(arr))


def harmful_rate_by_condition(
    records: Iterable[JudgmentRecord], *, confidence: float = 95.0,
) -> list[RateRow]:
    """Per-condition fraction of harmful responses on the *harmful* slice."""
    by_cond: dict[str, list[bool]] = {}
    for r in records:
        if r.kind != "harmful" or r.label is None:
            continue
        by_cond.setdefault(r.condition, []).append(r.is_harmful)
    return _format_rate_rows(by_cond, confidence=confidence, extras={})


def harmful_rate_by_condition_x_category(
    records: Iterable[JudgmentRecord], *, confidence: float = 95.0,
) -> list[RateRow]:
    """Per-(condition, category) harm rate, useful for breakdown tables."""
    bucket: dict[tuple[str, str], list[bool]] = {}
    for r in records:
        if r.kind != "harmful" or r.label is None:
            continue
        bucket.setdefault((r.condition, r.category), []).append(r.is_harmful)
    rows: list[RateRow] = []
    for (cond, cat), values in sorted(bucket.items()):
        rate, lo, hi, n = _wilson_rate(values, confidence=confidence)
        rows.append(
            RateRow(
                condition=cond, n=n, rate=rate, ci_low=lo, ci_high=hi,
                extras={"category": cat},
            )
        )
    return rows


def refusal_rate_on_benign(
    records: Iterable[JudgmentRecord], *, confidence: float = 95.0,
) -> list[RateRow]:
    """Per-condition refusal rate on the *benign* slice (over-refusal)."""
    by_cond: dict[str, list[bool]] = {}
    for r in records:
        if r.kind != "benign" or r.label is None:
            continue
        by_cond.setdefault(r.condition, []).append(r.is_refusal)
    return _format_rate_rows(by_cond, confidence=confidence, extras={})


def _format_rate_rows(
    by_cond: Mapping[str, list[bool]],
    *,
    confidence: float,
    extras: dict[str, str | float | int],
) -> list[RateRow]:
    rows: list[RateRow] = []
    for cond, values in sorted(by_cond.items()):
        rate, lo, hi, n = _wilson_rate(values, confidence=confidence)
        rows.append(
            RateRow(condition=cond, n=n, rate=rate, ci_low=lo, ci_high=hi, extras=dict(extras))
        )
    return rows


def write_summary_csv(rows: list[RateRow], path: Path) -> None:
    """Write a list of RateRow to CSV. Extras are flattened into columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    extras_keys = sorted({k for r in rows for k in r.extras.keys()})
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "n", "rate", "ci_low", "ci_high", *extras_keys])
        for r in rows:
            w.writerow(
                [r.condition, r.n, f"{r.rate:.4f}", f"{r.ci_low:.4f}", f"{r.ci_high:.4f}",
                 *[r.extras.get(k, "") for k in extras_keys]]
            )


def plot_condition_bars(
    harm_rows: list[RateRow],
    refusal_rows: list[RateRow] | None,
    *,
    title: str,
    output_path: Path,
) -> None:
    """Side-by-side bar plot: harm rate (and refusal rate, if provided)."""
    import matplotlib.pyplot as plt

    def _yerr_from_rows(rows: list[RateRow]) -> list[list[float]]:
        # Clamp to non-negative so tiny floating-point inversions at the
        # interval boundary do not make matplotlib reject the error bars.
        return [
            [max(0.0, r.rate - r.ci_low) for r in rows],
            [max(0.0, r.ci_high - r.rate) for r in rows],
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_refusal = refusal_rows is not None and len(refusal_rows) > 0
    fig, axes = plt.subplots(1, 2 if has_refusal else 1,
                             figsize=(10 if has_refusal else 6, 4),
                             squeeze=False)
    ax_harm = axes[0, 0]
    conditions = [r.condition for r in harm_rows]
    rates = [r.rate for r in harm_rows]
    ax_harm.bar(conditions, rates, yerr=_yerr_from_rows(harm_rows), capsize=4,
                color="#c45a5a", alpha=0.85)
    ax_harm.set_ylabel("harmful response rate")
    ax_harm.set_title("Harmful rate (95% Wilson CI)")
    ax_harm.set_ylim(0, max(1.0, max(rates) * 1.15) if rates else 1.0)
    for tick in ax_harm.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")

    if has_refusal:
        ax_ref = axes[0, 1]
        rconds = [r.condition for r in refusal_rows]
        rrates = [r.rate for r in refusal_rows]
        ax_ref.bar(rconds, rrates, yerr=_yerr_from_rows(refusal_rows), capsize=4,
                   color="#4a7a99", alpha=0.85)
        ax_ref.set_ylabel("refusal rate on benign control")
        ax_ref.set_title("Over-refusal (95% Wilson CI)")
        ax_ref.set_ylim(0, max(1.0, max(rrates) * 1.15) if rrates else 1.0)
        for tick in ax_ref.get_xticklabels():
            tick.set_rotation(20)
            tick.set_ha("right")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "JudgmentRecord",
    "RateRow",
    "load_judgments_jsonl",
    "harmful_rate_by_condition",
    "harmful_rate_by_condition_x_category",
    "refusal_rate_on_benign",
    "write_summary_csv",
    "plot_condition_bars",
]
