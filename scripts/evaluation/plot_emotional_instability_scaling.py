#!/usr/bin/env python3
"""Plot emotional_instability.density vs LoRA scaling factor with 95% CI.

Reads scaling_summary.jsonl from a sweep directory and plots the density
metric with confidence-interval bands.  Optionally adds horizontal reference
lines for base-model responses, n+ edited responses, and n- edited responses.

Usage:
    python -m scripts.evaluation.plot_emotional_instability_scaling \
        --results-dir scratch/trait_neuroticism_n+_500/scaling_sweep \
        --n-plus-edited scratch/trait_neuroticism_n+_500/edited_dataset_nplus.jsonl \
        --n-minus-edited scratch/trait_neuroticism_n-_500/edited_dataset_n-.jsonl \
        --base-inference scratch/trait_neuroticism_n+_500/inference_output.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compute_density_stats(jsonl_path: Path, response_key: str = "edited_response") -> tuple[float, float, float]:
    """Compute emotional_instability density mean, stdev, and 95% CI for responses in a JSONL file.

    If the file already contains 'emotional_instability.post.density' columns,
    use those directly.  Otherwise, run the NRCLex evaluation on each response.
    """
    rows = load_jsonl(jsonl_path)

    # Try pre-computed columns first
    density_key = "emotional_instability.post.density"
    if density_key in rows[0]:
        densities = [r[density_key] for r in rows]
    else:
        # Compute on the fly using NRCLex directly (avoids heavy package imports)
        import nltk
        nltk.download("punkt_tab", quiet=True)
        from nrclex import NRCLex
        densities = []
        for r in rows:
            text = r.get(response_key, "")
            emotion = NRCLex(text)
            density = emotion.affect_frequencies.get("negative", 0.0) * 100
            densities.append(round(density, 2))

    n = len(densities)
    mean = sum(densities) / n
    stdev = math.sqrt(sum((d - mean) ** 2 for d in densities) / (n - 1)) if n > 1 else 0.0
    ci = 1.96 * stdev / math.sqrt(n)
    return mean, stdev, ci


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot emotional instability scaling with CI.")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory with scaling_summary.jsonl and sweep_metadata.json")
    parser.add_argument("--n-plus-edited", type=str, default=None,
                        help="Path to n+ edited_dataset JSONL for reference line")
    parser.add_argument("--n-minus-edited", type=str, default=None,
                        help="Path to n- edited_dataset JSONL for reference line")
    parser.add_argument("--base-inference", type=str, default=None,
                        help="Path to base model inference_output.jsonl for reference line")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path. Defaults to <results-dir>/emotional_instability_scaling.png")
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    # ── Load sweep data ──────────────────────────────────────────────────
    summary = load_jsonl(results_dir / "scaling_summary.jsonl")
    with open(results_dir / "sweep_metadata.json") as f:
        metadata = json.load(f)

    scales = [r["scaling_factor"] for r in summary]
    means = [r["emotional_instability.density.mean"] for r in summary]
    stdevs = [r.get("emotional_instability.density.stdev", 0) for r in summary]
    n = summary[0]["num_samples"]
    cis = [1.96 * s / math.sqrt(n) for s in stdevs]

    ci_lo = [m - c for m, c in zip(means, cis)]
    ci_hi = [m + c for m, c in zip(means, cis)]

    # ── Figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    # Main line + CI band
    ax.fill_between(scales, ci_lo, ci_hi, alpha=0.18, color="#009688", label="95% CI")
    ax.plot(scales, means, "o-", color="#00796B", linewidth=2.5, markersize=6,
            label="LoRA-steered response", zorder=5)

    # ── Reference lines ──────────────────────────────────────────────────
    # Base model responses
    if args.base_inference:
        base_mean, _, base_ci = compute_density_stats(
            Path(args.base_inference), response_key="response")
        ax.axhline(base_mean, color="#4FC3F7", linestyle="--", linewidth=1.8, alpha=0.9,
                    label=f"Base response (mean={base_mean:.1f}%)", zorder=3)
        ax.axhspan(base_mean - base_ci, base_mean + base_ci,
                    color="#4FC3F7", alpha=0.10, zorder=2)

    # n+ edited responses
    if args.n_plus_edited:
        nplus_mean, _, nplus_ci = compute_density_stats(Path(args.n_plus_edited))
        ax.axhline(nplus_mean, color="#FF9800", linestyle="--", linewidth=1.8, alpha=0.9,
                    label=f"Edited response n+ (mean={nplus_mean:.1f}%)", zorder=3)
        ax.axhspan(nplus_mean - nplus_ci, nplus_mean + nplus_ci,
                    color="#FF9800", alpha=0.10, zorder=2)

    # n- edited responses
    if args.n_minus_edited:
        nminus_mean, _, nminus_ci = compute_density_stats(Path(args.n_minus_edited))
        ax.axhline(nminus_mean, color="#7B1FA2", linestyle="--", linewidth=2.5, alpha=1.0,
                    label=f"Edited response n\u2212 (mean={nminus_mean:.1f}%)", zorder=6)
        ax.axhspan(nminus_mean - nminus_ci, nminus_mean + nminus_ci,
                    color="#CE93D8", alpha=0.18, zorder=2)

    # ── Labels & styling ─────────────────────────────────────────────────
    ax.set_xlabel("LoRA scaling factor", fontsize=12)
    ax.set_ylabel("NRCLex negative sentiment (%)", fontsize=12)
    ax.set_title(
        f"Emotional Instability (NRCLex negative sentiment) vs LoRA Scaling Factor\n"
        f"(n={n} per point, 95% CI)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(scales) - 0.2, max(scales) + 0.2)

    plt.tight_layout()
    outpath = args.output or str(results_dir / "emotional_instability_scaling.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {outpath}")


if __name__ == "__main__":
    main()
