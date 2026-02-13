#!/usr/bin/env python3
"""Plot scaling sweep results: metric count & density vs LoRA scale with CIs and sample text boxes.

All metric names are resolved from sweep_metadata.json (written by eval_lora_scaling.py),
so this script works for any persona without hardcoded metric keys.

Usage:
    python -m scripts.evaluation.plot_scaling --results-dir scratch/.../scaling_sweep
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LoRA scaling sweep results.")
    parser.add_argument(
        "--results-dir", type=str, required=True,
        help="Directory containing sweep_metadata.json, scaling_summary.jsonl, and all_results.jsonl",
    )
    parser.add_argument("--sample-question-idx", type=int, default=0,
                        help="Index of question to show in sample text boxes (default: 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for the plot. Defaults to <results-dir>/scaling_plot.png")
    parser.add_argument(
        "--edited-dataset", type=str, default=None,
        help="Path to edited_dataset.jsonl to add 'prompted for' reference lines.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def discover_metric_keys(summary_row: dict, evaluation: str) -> list[tuple[str, str, str]]:
    """Find all metric sub-keys for an evaluation from a summary row.

    Returns list of (sub_key, mean_key, stdev_key) tuples.
    E.g. for evaluation="verb_count": [("count", "verb_count.count.mean", "verb_count.count.stdev"),
                                        ("density", "verb_count.density.mean", "verb_count.density.stdev")]
    """
    prefix = f"{evaluation}."
    seen_subkeys: dict[str, bool] = {}
    for key in summary_row:
        if key.startswith(prefix) and key.endswith(".mean"):
            # e.g. "verb_count.count.mean" -> sub_key = "count"
            middle = key[len(prefix):-len(".mean")]
            seen_subkeys[middle] = True

    result = []
    for sub_key in seen_subkeys:
        mean_key = f"{evaluation}.{sub_key}.mean"
        stdev_key = f"{evaluation}.{sub_key}.stdev"
        result.append((sub_key, mean_key, stdev_key))
    return sorted(result, key=lambda t: t[0])


def compute_prompted_for_references(
    edited_dataset_path: str,
    evaluations: list[str],
) -> dict[str, float]:
    """Compute mean metric values from the edited (training target) responses.

    Runs the same evaluation functions used in the sweep on the edited responses,
    so that reference lines match the sweep metrics exactly.

    Returns dict like {"verb_count.count": 3.17, "verb_count.density": 1.67}.
    """
    from scripts.evaluation import run_evaluation, EvaluationConfig
    from datasets import Dataset

    rows = load_jsonl(Path(edited_dataset_path))
    records = [{"question": r.get("question", ""), "response": r["edited_response"]} for r in rows]
    ds = Dataset.from_list(records)

    eval_config = EvaluationConfig(
        evaluations=evaluations,
        response_column="response",
        question_column="question",
    )
    _, eval_result = run_evaluation(eval_config, dataset=ds)

    # Extract mean values for each metric sub-key
    refs: dict[str, float] = {}
    for key, val in eval_result.aggregates.items():
        if key.endswith(".mean"):
            # e.g. "verb_count.count.mean" -> "verb_count.count"
            metric_key = key[: -len(".mean")]
            refs[metric_key] = val
    return refs


def wrap_text(text: str, width: int = 44) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    # ── Load metadata ─────────────────────────────────────────────────────
    metadata_path = results_dir / "sweep_metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    persona = metadata["persona"]
    evaluations = metadata["evaluations"]
    base_model = metadata.get("base_model", "unknown")
    n_samples = metadata.get("num_samples", "?")
    primary_eval = evaluations[0]  # first evaluation is the primary persona metric

    # ── Load results ──────────────────────────────────────────────────────
    summary = load_jsonl(results_dir / "scaling_summary.jsonl")
    per_question = load_jsonl(results_dir / "all_results.jsonl")

    scales = [r["scaling_factor"] for r in summary]
    n = summary[0]["num_samples"]

    # ── Compute "prompted for" reference lines from edited dataset ────────
    prompted_for_refs: dict[str, float] = {}
    if args.edited_dataset:
        print(f"Computing 'prompted for' reference from {args.edited_dataset} ...")
        prompted_for_refs = compute_prompted_for_references(args.edited_dataset, evaluations)
        print(f"  Reference values: {prompted_for_refs}")

    # ── Load prompted baselines (maximize/minimize) if available ──────────
    prompted_baselines: dict | None = None
    prompted_baselines_path = results_dir / "prompted_baselines.json"
    if prompted_baselines_path.exists():
        with open(prompted_baselines_path) as f:
            prompted_baselines = json.load(f)
        print(f"Loaded prompted baselines from {prompted_baselines_path}")

    # ── Discover metric sub-keys from the primary evaluation ──────────────
    metric_keys = discover_metric_keys(summary[0], primary_eval)
    if not metric_keys:
        raise ValueError(
            f"No metric keys found for evaluation '{primary_eval}' in summary. "
            f"Available keys: {list(summary[0].keys())}"
        )

    # ── Build per-metric data ─────────────────────────────────────────────
    metric_data: dict[str, dict] = {}
    for sub_key, mean_key, stdev_key in metric_keys:
        means = [r[mean_key] for r in summary]
        stdevs = [r.get(stdev_key, 0) for r in summary]
        cis = [1.96 * s / math.sqrt(n) for s in stdevs]
        metric_data[sub_key] = {"means": means, "stdevs": stdevs, "cis": cis}

    # ── Create figure ─────────────────────────────────────────────────────
    n_panels = len(metric_keys) + 1  # +1 for sample text boxes
    height_ratios = [1.0] * len(metric_keys) + [1.1]
    fig = plt.figure(figsize=(18, 8 * n_panels))
    gs = fig.add_gridspec(n_panels, 1, height_ratios=height_ratios, hspace=0.30)

    # Shorten base model name for title
    model_short = base_model.split("/")[-1] if "/" in base_model else base_model
    fig.suptitle(
        f"LoRA Scaling Sweep — {persona} persona\n"
        f"{model_short}  ·  {n_samples} questions per scale  ·  eval: {primary_eval}",
        fontsize=16, fontweight="bold", y=0.98,
    )

    scale_min = min(scales)
    scale_max = max(scales)
    xlim = (scale_min - 0.2, scale_max + 0.2)

    colors = [
        ("#2196F3", "#1565C0", "o-"),
        ("#FF9800", "#E65100", "s-"),
        ("#4CAF50", "#2E7D32", "^-"),
        ("#9C27B0", "#6A1B9A", "D-"),
    ]

    # ── Metric panels ────────────────────────────────────────────────────
    for panel_idx, (sub_key, mean_key, stdev_key) in enumerate(metric_keys):
        ax = fig.add_subplot(gs[panel_idx])
        data = metric_data[sub_key]
        ci_color, line_color, marker = colors[panel_idx % len(colors)]

        ax.fill_between(
            scales,
            [m - c for m, c in zip(data["means"], data["cis"])],
            [m + c for m, c in zip(data["means"], data["cis"])],
            alpha=0.2, color=ci_color, label="95% CI",
        )
        ax.plot(
            scales, data["means"], marker,
            color=line_color, linewidth=2.5, markersize=7,
            label=f"Mean {primary_eval}.{sub_key}", zorder=5,
        )
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5, label="Baseline (s=0)")
        ax.axvline(1, color="green", linestyle=":", alpha=0.5, label="Standard LoRA (s=1)")

        ax.annotate(
            "\u2190 Persona reversal",
            xy=(max(scale_min, -0.95), max(data["means"]) * 0.95),
            fontsize=10, color="#C62828", fontstyle="italic",
        )
        ax.annotate(
            "Persona amplified \u2192",
            xy=(1.05, max(data["means"]) * 0.95),
            fontsize=10, color="#2E7D32", fontstyle="italic", ha="left",
        )

        # Draw "edited answers" reference line if available
        ref_key = f"{primary_eval}.{sub_key}"
        if ref_key in prompted_for_refs:
            ref_val = prompted_for_refs[ref_key]
            ax.axhline(
                ref_val, color="#D32F2F", linestyle="-", linewidth=3, alpha=0.9,
                label=f"Edited answers ({ref_val:.2f})", zorder=4,
            )

        # Draw prompted baseline reference lines if available
        if prompted_baselines:
            mean_key = f"{primary_eval}.{sub_key}.mean"
            if "maximize" in prompted_baselines and mean_key in prompted_baselines["maximize"]:
                max_val = prompted_baselines["maximize"][mean_key]
                ax.axhline(
                    max_val, color="#1565C0", linestyle="--", linewidth=2, alpha=0.8,
                    label=f"Prompted: maximize ({max_val:.2f})", zorder=4,
                )
            if "minimize" in prompted_baselines and mean_key in prompted_baselines["minimize"]:
                min_val = prompted_baselines["minimize"][mean_key]
                ax.axhline(
                    min_val, color="#7B1FA2", linestyle="--", linewidth=2, alpha=0.8,
                    label=f"Prompted: minimize ({min_val:.2f})", zorder=4,
                )

        ax.set_xlabel("Scaling Factor", fontsize=12)
        ax.set_ylabel(f"{primary_eval}.{sub_key} (mean \u00b1 95% CI)", fontsize=12)
        ax.set_title(f"{primary_eval}.{sub_key} vs. LoRA Scaling Factor", fontsize=14)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(*xlim)

    # ── Bottom panel: sample text boxes ───────────────────────────────────
    ax_boxes = fig.add_subplot(gs[-1])

    sample_idx = min(args.sample_question_idx, len(per_question) - 1)
    sample_q = per_question[sample_idx]
    question_label = sample_q["question"]

    # Pick up to 5 evenly-spaced scales for the boxes
    available_scales = sorted(sample_q["scales"].keys(), key=float)
    if len(available_scales) <= 5:
        box_scales = available_scales
    else:
        indices = [int(i * (len(available_scales) - 1) / 4) for i in range(5)]
        box_scales = [available_scales[i] for i in indices]

    # Resolve the primary metric's first sub-key for display in boxes
    primary_sub_key = metric_keys[0][0]  # e.g. "count"
    per_q_metric_key = f"{primary_eval}.{primary_sub_key}"

    ax_boxes.set_xlim(0, 10.2)
    ax_boxes.set_ylim(0, 2.15)
    ax_boxes.axis("off")
    ax_boxes.set_title(
        f'Sample responses — Q: "{question_label[:80]}{"..." if len(question_label) > 80 else ""}"\n'
        f'Metric shown: {per_q_metric_key}',
        fontsize=13, fontweight="bold", pad=10,
    )

    bg_colors = ["#FFCDD2", "#FFE0B2", "#E0E0E0", "#C8E6C9", "#A5D6A7"]
    box_width = 10.0 / len(box_scales)

    for i, sk in enumerate(box_scales):
        scale_val = float(sk)
        scale_data = sample_q["scales"][sk]
        resp = scale_data["response"]
        metric_val = scale_data["metrics"].get(per_q_metric_key, "N/A")

        wrapped = wrap_text(resp[:220] + ("..." if len(resp) > 220 else ""), width=30)
        label = "BASELINE" if abs(scale_val) < 1e-6 else ""
        header = f"s={scale_val:+.1f}  {primary_sub_key}:{metric_val}"
        if label:
            header += f"  [{label}]"
        box_text = f"{header}\n{'─' * 32}\n{wrapped}"

        bx = 0.10 + i * box_width
        bg = bg_colors[i % len(bg_colors)]
        props = dict(
            boxstyle="round,pad=0.4", facecolor=bg,
            edgecolor="#757575", alpha=0.92, linewidth=1.3,
        )
        ax_boxes.text(
            bx, 2.0, box_text,
            fontsize=7.8, fontfamily="monospace",
            verticalalignment="top", horizontalalignment="left",
            bbox=props,
        )

    # Arrows between boxes
    for i in range(len(box_scales) - 1):
        x1 = 0.10 + i * box_width
        x2 = 0.10 + (i + 1) * box_width
        ax_boxes.annotate(
            "", xy=(x2 + 0.05, 1.25), xytext=(x1 + box_width - 0.15, 1.25),
            arrowprops=dict(arrowstyle="-|>", color="#616161", lw=2, shrinkA=2, shrinkB=2),
        )

    # ── Footer ────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.005,
        f"Persona: {persona}  ·  Eval: {primary_eval}  ·  "
        f"95% CI = mean \u00b1 1.96\u00b7\u03c3/\u221an  (n={n})  ·  "
        f"Model: {model_short}",
        ha="center", fontsize=9, fontstyle="italic", color="#616161",
    )

    plt.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.04)
    outpath = args.output or str(results_dir / "scaling_plot.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {outpath}")


if __name__ == "__main__":
    main()
