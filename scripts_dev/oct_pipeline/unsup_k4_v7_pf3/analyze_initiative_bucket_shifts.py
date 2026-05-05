"""Bucketed baseline-position analysis for unsup_k4_v7_pf3 validation runs.

Reads the cached outputs from ``validate_lora.py`` for existing 200-persona
amplifier/suppressor validations. For each factor, it buckets personas by
their *baseline* score on that same factor (low / medium / high tertiles), then
reports how much the LoRA shifts those personas.

No model inference is run; this is pure post-hoc analysis over
``*_scores.npz`` and ``*_summary.json``.

Run:

    uv run python scripts_dev/oct_pipeline/unsup_k4_v7_pf3/analyze_initiative_bucket_shifts.py

Optional:

    uv run python scripts_dev/oct_pipeline/unsup_k4_v7_pf3/analyze_initiative_bucket_shifts.py \\
        --analysis-name pedagogy \\
        --labels pedagogy_amp pedagogy_sup \\
        --output-dir scratch/factor_inspect_v7_pf3/validate/pedagogy_bucket_shift_analysis \\
        --bucket-method tertile
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


VALIDATE_ROOT = Path("scratch/factor_inspect_v7_pf3/validate")
DEFAULT_LABELS = ("initiative_amp", "initiative_sup")
DEFAULT_OUTPUT_DIR = VALIDATE_ROOT / "initiative_bucket_shift_analysis"


def _load_factor_names(run_dir: Path, label: str, n_factors: int) -> list[str]:
    summary_path = run_dir / f"{label}_summary.json"
    if not summary_path.exists():
        return [f"F{i}" for i in range(n_factors)]

    summary = json.loads(summary_path.read_text())
    factor_identity = summary.get("factor_identity") or {}
    names = factor_identity.get("factor_names")
    if isinstance(names, list) and len(names) == n_factors:
        return [str(n) for n in names]

    rows = summary.get("factor_summary") or []
    names = [str(r.get("factor")) for r in rows if r.get("factor") is not None]
    if len(names) == n_factors:
        return names

    return [f"F{i}" for i in range(n_factors)]


def load_scores(label: str, validate_root: Path) -> dict[str, Any]:
    """Load one validation run's paired baseline/LoRA factor scores."""
    run_dir = validate_root / label
    scores_path = run_dir / f"{label}_scores.npz"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"Missing {scores_path}. Run validate_lora.py first, or pass "
            "--validate-root pointing at existing outputs."
        )

    data = np.load(scores_path, allow_pickle=False)
    sample_ids = [str(x) for x in data["sample_ids"]]
    baseline_scores = np.asarray(data["baseline_scores"], dtype=float)
    lora_scores = np.asarray(data["lora_scores"], dtype=float)

    if baseline_scores.shape != lora_scores.shape:
        raise RuntimeError(
            f"{label}: baseline/lora shape mismatch: "
            f"{baseline_scores.shape} vs {lora_scores.shape}"
        )
    if baseline_scores.shape[0] != len(sample_ids):
        raise RuntimeError(
            f"{label}: sample_id count {len(sample_ids)} does not match "
            f"score rows {baseline_scores.shape[0]}"
        )

    return {
        "label": label,
        "sample_ids": sample_ids,
        "baseline_scores": baseline_scores,
        "lora_scores": lora_scores,
        "factor_names": _load_factor_names(run_dir, label, baseline_scores.shape[1]),
    }


def bucket_indices(values: np.ndarray, method: str) -> tuple[list[tuple[str, np.ndarray]], dict[str, float]]:
    """Return named bucket masks using baseline values for one factor."""
    if method != "tertile":
        raise ValueError(f"Unknown bucket method: {method}")

    q_low, q_high = np.quantile(values, [1 / 3, 2 / 3])
    buckets = [
        ("low", values <= q_low),
        ("medium", (values > q_low) & (values <= q_high)),
        ("high", values > q_high),
    ]
    return buckets, {"low_high_cutoff": float(q_low), "medium_high_cutoff": float(q_high)}


def bootstrap_ci_mean(values: np.ndarray, *, seed: int, n_resamples: int = 5000) -> tuple[float, float]:
    """Percentile bootstrap CI for a bucket's mean shift."""
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[draws].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_bucket(
    *,
    label: str,
    factor_index: int,
    factor_name: str,
    bucket_name: str,
    mask: np.ndarray,
    baseline: np.ndarray,
    lora: np.ndarray,
    cutoffs: dict[str, float],
    seed: int,
) -> dict[str, Any]:
    """Build one summary row for one label/factor/baseline bucket."""
    b = baseline[mask]
    l = lora[mask]
    d = l - b
    ci_lo, ci_hi = bootstrap_ci_mean(d, seed=seed)
    std = float(d.std(ddof=1)) if len(d) > 1 else float("nan")
    dz = float(d.mean() / std) if std > 0 else float("nan")
    return {
        "label": label,
        "factor_index": factor_index,
        "factor": factor_name,
        "bucket": bucket_name,
        "bucket_method": "tertile",
        "n": int(len(d)),
        "baseline_min": float(b.min()) if len(b) else None,
        "baseline_max": float(b.max()) if len(b) else None,
        "baseline_mean": float(b.mean()) if len(b) else None,
        "lora_mean": float(l.mean()) if len(l) else None,
        "mean_shift": float(d.mean()) if len(d) else None,
        "median_shift": float(np.median(d)) if len(d) else None,
        "ci_95_lo": ci_lo,
        "ci_95_hi": ci_hi,
        "cohen_dz": dz,
        "n_pos_shift": int((d > 0).sum()),
        "n_neg_shift": int((d < 0).sum()),
        **cutoffs,
    }


def analyze(labels: list[str], validate_root: Path, bucket_method: str) -> tuple[list[dict], list[dict]]:
    """Return aggregate rows and per-persona rows for all requested labels."""
    aggregate_rows: list[dict] = []
    persona_rows: list[dict] = []

    for label_i, label in enumerate(labels):
        loaded = load_scores(label, validate_root)
        baseline_scores = loaded["baseline_scores"]
        lora_scores = loaded["lora_scores"]
        sample_ids = loaded["sample_ids"]
        factor_names = loaded["factor_names"]

        for factor_index, factor_name in enumerate(factor_names):
            baseline = baseline_scores[:, factor_index]
            lora = lora_scores[:, factor_index]
            buckets, cutoffs = bucket_indices(baseline, bucket_method)
            for bucket_i, (bucket_name, mask) in enumerate(buckets):
                aggregate_rows.append(
                    summarize_bucket(
                        label=label,
                        factor_index=factor_index,
                        factor_name=factor_name,
                        bucket_name=bucket_name,
                        mask=mask,
                        baseline=baseline,
                        lora=lora,
                        cutoffs=cutoffs,
                        seed=436 + 100 * label_i + 10 * factor_index + bucket_i,
                    )
                )
                for row_idx in np.where(mask)[0]:
                    persona_rows.append(
                        {
                            "label": label,
                            "sample_id": sample_ids[row_idx],
                            "factor_index": factor_index,
                            "factor": factor_name,
                            "bucket": bucket_name,
                            "baseline_score": float(baseline[row_idx]),
                            "lora_score": float(lora[row_idx]),
                            "shift": float(lora[row_idx] - baseline[row_idx]),
                        }
                    )

    return aggregate_rows, persona_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dictionaries as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: Any, digits: int = 3) -> str:
    """Format optional numeric values for Markdown tables."""
    if value is None:
        return ""
    try:
        if np.isnan(value):
            return ""
    except TypeError:
        pass
    return f"{float(value):+.{digits}f}"


def write_markdown(path: Path, rows: list[dict], labels: list[str], analysis_name: str) -> None:
    """Write a compact human-readable report."""
    title = analysis_name.replace("_", " ").title()
    lines: list[str] = [
        f"# {title} LoRA Bucketed Shift Analysis",
        "",
        "Personas are bucketed separately for each factor by their baseline "
        "score on that factor: low <= p33, medium p33-p66, high > p66.",
        "",
    ]

    by_label = {label: [r for r in rows if r["label"] == label] for label in labels}
    for label in labels:
        lines.append(f"## {label}")
        lines.append("")
        factor_names = []
        for r in by_label[label]:
            if r["factor"] not in factor_names:
                factor_names.append(r["factor"])
        for factor in factor_names:
            lines.append(f"### {factor}")
            lines.append("")
            lines.append(
                "| baseline bucket | n | baseline mean | LoRA mean | mean shift | "
                "95% CI | median shift | pos / neg |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for r in [x for x in by_label[label] if x["factor"] == factor]:
                ci = f"[{format_float(r['ci_95_lo'])}, {format_float(r['ci_95_hi'])}]"
                lines.append(
                    f"| {r['bucket']} | {r['n']} | "
                    f"{format_float(r['baseline_mean'])} | "
                    f"{format_float(r['lora_mean'])} | "
                    f"{format_float(r['mean_shift'])} | "
                    f"{ci} | "
                    f"{format_float(r['median_shift'])} | "
                    f"{r['n_pos_shift']} / {r['n_neg_shift']} |"
                )
            lines.append("")

    path.write_text("\n".join(lines) + "\n")


def write_plot(path: Path, rows: list[dict], labels: list[str]) -> None:
    """Write one grouped bar plot of mean shift by label/factor/bucket."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] skipped: {exc}")
        return

    factor_names = []
    for row in rows:
        if row["factor"] not in factor_names:
            factor_names.append(row["factor"])
    buckets = ["low", "medium", "high"]

    fig, axes = plt.subplots(
        nrows=len(factor_names),
        ncols=1,
        figsize=(9, 2.4 * len(factor_names)),
        sharex=True,
    )
    if len(factor_names) == 1:
        axes = [axes]

    colors = {"initiative_amp": "#3366aa", "initiative_sup": "#aa5533"}
    default_colors = ["#3366aa", "#aa5533", "#669955", "#7755aa"]

    for ax, factor in zip(axes, factor_names):
        x = np.arange(len(buckets))
        width = 0.36
        for label_i, label in enumerate(labels):
            label_rows = {
                r["bucket"]: r
                for r in rows
                if r["label"] == label and r["factor"] == factor
            }
            means = [label_rows[b]["mean_shift"] for b in buckets]
            offsets = (label_i - (len(labels) - 1) / 2) * width
            ax.bar(
                x + offsets,
                means,
                width=width,
                label=label,
                color=colors.get(label, default_colors[label_i % len(default_colors)]),
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(-2, 2)
        ax.set_ylabel("LoRA - baseline")
        ax.set_title(factor)
        ax.set_xticks(x)
        ax.set_xticklabels(buckets)

    axes[-1].set_xlabel("baseline score bucket for the same factor")
    axes[0].legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--validate-root",
        type=Path,
        default=VALIDATE_ROOT,
        help="Directory containing validate_lora.py output subdirs.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_LABELS),
        help="Validation labels to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write CSV/JSON/Markdown/plot outputs.",
    )
    parser.add_argument(
        "--analysis-name",
        default="initiative",
        help="Stem/title for output files, e.g. initiative or pedagogy.",
    )
    parser.add_argument(
        "--bucket-method",
        choices=["tertile"],
        default="tertile",
        help="How to bucket baseline scores.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate_rows, persona_rows = analyze(
        labels=args.labels,
        validate_root=args.validate_root,
        bucket_method=args.bucket_method,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.analysis_name}_bucket_shift"
    write_csv(args.output_dir / f"{stem}_summary.csv", aggregate_rows)
    write_csv(args.output_dir / f"{stem}_per_persona.csv", persona_rows)
    (args.output_dir / f"{stem}_summary.json").write_text(
        json.dumps(aggregate_rows, indent=2)
    )
    write_markdown(args.output_dir / f"{stem}_report.md", aggregate_rows, args.labels, args.analysis_name)
    write_plot(args.output_dir / f"{stem}_mean_plot.png", aggregate_rows, args.labels)

    print(f"[done] wrote {len(aggregate_rows)} bucket summaries to {args.output_dir}")
    print(f"[done] wrote {len(persona_rows)} per-persona rows to {args.output_dir}")


if __name__ == "__main__":
    main()
