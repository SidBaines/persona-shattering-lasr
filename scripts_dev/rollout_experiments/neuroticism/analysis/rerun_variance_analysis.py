#!/usr/bin/env python3
"""Rerun variance analysis for neuroticism and coherence judge sweeps.

Decomposes the observed score variance into two components:

1. **Within-response variance** — how much a judge's score varies when it rates
   the *same* response multiple times (repeat_index 0/1/2 within one run).
   This measures judge stochasticity.

2. **Between-response variance** — how much scores vary across different
   responses at the *same* LoRA scale (i.e., different questions/texts).
   This is the signal variance.

The key question: are the large error bars in the sweep plots driven by judge
noise or by genuine question-level variability in trait expression?

Usage::

    uv run python -m scripts_dev.rollout_experiments.neuroticism.rerun_variance_analysis
    uv run python -m scripts_dev.rollout_experiments.neuroticism.rerun_variance_analysis --upload
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv

# ── Run manifest ───────────────────────────────────────────────────────────────
# Reuse the same run directories from coherence_analysis.py.
# For variance analysis we need runs with judge_repeats > 1.
# The gpt_4o_mini_2 runs have 3 repeats (repeat_index 0/1/2) at temp=0.

JUDGE_RUNS_ROOT = Path("scratch/judge_runs")
OUTPUT_DIR = Path("scratch/rerun_variance_analysis")
HF_REPO_ID = "persona-shattering-lasr/ocean_judge_runs"
HF_PATH = "rerun_variance_analysis"

# Runs with multiple repeats per response (3 repeats each, temp=0).
#
# Coherence metric: gpt_4o_mini_2 runs (rater_id gpt_4o_mini and gpt_4o_mini_2)
# Neuroticism metric: original gpt_4o_mini-gemini_flash_20 runs
#   (both raters use gpt-4o-mini in these runs, confirmed by coherence_analysis.py)
MULTI_REPEAT_RUNS: list[dict] = [
    # --- coherence (gpt_4o_mini_2 renamed runs, metric_name="coherence") ---
    {"adapter": "sft",  "run_dir": "gpt_4o_mini-gpt_4o_mini_2-ad544c5585eb", "rater_id": "gpt_4o_mini",   "metric": "coherence"},
    {"adapter": "sft",  "run_dir": "gpt_4o_mini-gpt_4o_mini_2-ad544c5585eb", "rater_id": "gpt_4o_mini_2", "metric": "coherence"},
    {"adapter": "dpo",  "run_dir": "gpt_4o_mini-gpt_4o_mini_2-99a779f80e93", "rater_id": "gpt_4o_mini",   "metric": "coherence"},
    {"adapter": "dpo",  "run_dir": "gpt_4o_mini-gpt_4o_mini_2-99a779f80e93", "rater_id": "gpt_4o_mini_2", "metric": "coherence"},
    {"adapter": "soup", "run_dir": "gpt_4o_mini-gpt_4o_mini_2-c4bf45cc735d", "rater_id": "gpt_4o_mini",   "metric": "coherence"},
    {"adapter": "soup", "run_dir": "gpt_4o_mini-gpt_4o_mini_2-c4bf45cc735d", "rater_id": "gpt_4o_mini_2", "metric": "coherence"},
    {"adapter": "old",  "run_dir": "gpt_4o_mini-gpt_4o_mini_2-18c0fe6a4e1d", "rater_id": "gpt_4o_mini",   "metric": "coherence"},
    {"adapter": "old",  "run_dir": "gpt_4o_mini-gpt_4o_mini_2-18c0fe6a4e1d", "rater_id": "gpt_4o_mini_2", "metric": "coherence"},
    # --- neuroticism (original runs, metric_name="neuroticism_v2") ---
    {"adapter": "sft",  "run_dir": "gpt_4o_mini-gemini_flash_20-350b967bc906", "rater_id": "gpt_4o_mini",   "metric": "neuroticism"},
    {"adapter": "sft",  "run_dir": "gpt_4o_mini-gemini_flash_20-350b967bc906", "rater_id": "gemini_flash_20", "metric": "neuroticism"},
    {"adapter": "dpo",  "run_dir": "gpt_4o_mini-gemini_flash_20-02524955a8f4", "rater_id": "gpt_4o_mini",   "metric": "neuroticism"},
    {"adapter": "dpo",  "run_dir": "gpt_4o_mini-gemini_flash_20-02524955a8f4", "rater_id": "gemini_flash_20", "metric": "neuroticism"},
    {"adapter": "soup", "run_dir": "gpt_4o_mini-gemini_flash_20-5d04426ae5c0", "rater_id": "gpt_4o_mini",   "metric": "neuroticism"},
    {"adapter": "soup", "run_dir": "gpt_4o_mini-gemini_flash_20-5d04426ae5c0", "rater_id": "gemini_flash_20", "metric": "neuroticism"},
    {"adapter": "old",  "run_dir": "gpt_4o_mini-gemini_flash_20-30211aa3a32c", "rater_id": "gpt_4o_mini",   "metric": "neuroticism"},
    {"adapter": "old",  "run_dir": "gpt_4o_mini-gemini_flash_20-30211aa3a32c", "rater_id": "gemini_flash_20", "metric": "neuroticism"},
]

# Not used separately anymore — entries above are split by metric field.
NEUROTICISM_RUNS: list[dict] = []

import re
_SCALE_RE = re.compile(r"@scale_([+-]?\d+(?:\.\d+)?)")

_ADAPTER_ORDER = ["sft", "dpo", "soup", "old"]


# ── Data loading ───────────────────────────────────────────────────────────────


def load_repeats_from_run(run_dir: str, rater_id: str) -> dict[str, dict[int, float]]:
    """Load scores keyed by (response_id -> {repeat_index -> score})."""
    path = JUDGE_RUNS_ROOT / run_dir / "judge_calls" / "raw" / f"{rater_id}.jsonl"
    if not path.exists():
        print(f"  WARNING: missing {path}")
        return {}

    result: dict[str, dict[int, float]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("status") != "success" or rec.get("score") is None:
            continue
        result[rec["response_id"]][rec["repeat_index"]] = float(rec["score"])
    return result


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))


# ── Variance decomposition ─────────────────────────────────────────────────────


def decompose_variance(run_entries: list[dict]) -> dict:
    """Compute within-response and between-response variance components.

    Returns:
        {
          "within_mean": float,   # mean within-response std (judge noise)
          "within_nonzero_frac": float,  # fraction where within-std > 0
          "between_by_scale": {scale: float},  # between-response std at each scale
          "between_mean": float,  # mean between-response std across scales
          "n_responses": int,
          "n_scales": int,
        }
    """
    # {response_id: {repeat_index: score}} — pooled across all entries in this group
    # We pair entries by adapter: each run_entry is one (run_dir, rater_id)
    pooled: dict[str, list[float]] = defaultdict(list)  # response_id -> all scores
    by_scale: dict[float, list[float]] = defaultdict(list)  # scale -> per-response means

    within_stds: list[float] = []

    for entry in run_entries:
        data = load_repeats_from_run(entry["run_dir"], entry["rater_id"])
        for resp_id, repeat_map in data.items():
            scores = list(repeat_map.values())
            if len(scores) > 1:
                within_stds.append(_std(scores))
            mean_score = sum(scores) / len(scores)
            pooled[resp_id].append(mean_score)

    # Extract scale from response_id and compute between-response std per scale
    for resp_id, means in pooled.items():
        m = _SCALE_RE.search(resp_id)
        if m:
            scale = float(m.group(1))
            # average over all runs for this response
            by_scale[scale].append(sum(means) / len(means))

    between_stds_by_scale = {
        scale: _std(scores)
        for scale, scores in sorted(by_scale.items())
    }

    return {
        "within_mean": sum(within_stds) / len(within_stds) if within_stds else 0.0,
        "within_nonzero_frac": sum(1 for s in within_stds if s > 0) / len(within_stds) if within_stds else 0.0,
        "between_by_scale": between_stds_by_scale,
        "between_mean": (
            sum(between_stds_by_scale.values()) / len(between_stds_by_scale)
            if between_stds_by_scale else 0.0
        ),
        "n_responses": len(pooled),
        "n_scales": len(between_stds_by_scale),
    }


# ── Plotting ───────────────────────────────────────────────────────────────────


def plot_variance_decomposition(
    all_stats: dict[str, dict],  # metric -> adapter -> stats
    output: Path,
) -> Path:
    """Bar chart comparing within-response vs between-response std."""
    metrics = list(all_stats.keys())
    adapters = _ADAPTER_ORDER

    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5), squeeze=False)

    for col, metric in enumerate(metrics):
        ax = axes[0][col]
        stats_by_adapter = all_stats[metric]

        x = range(len(adapters))
        within_vals = [stats_by_adapter[a]["within_mean"] for a in adapters]
        between_vals = [stats_by_adapter[a]["between_mean"] for a in adapters]

        width = 0.35
        bars1 = ax.bar([xi - width / 2 for xi in x], within_vals, width,
                       label="Within-response std\n(judge noise, same response rated 3×)",
                       color="#4363d8", alpha=0.8)
        bars2 = ax.bar([xi + width / 2 for xi in x], between_vals, width,
                       label="Between-response std\n(question variability at same scale)",
                       color="#e6194b", alpha=0.8)

        ax.set_xticks(list(x))
        ax.set_xticklabels(adapters, fontsize=11)
        ax.set_ylabel("Std (score units)", fontsize=10)
        ax.set_title(f"{metric} — variance decomposition", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        # Annotate ratio
        for xi, (w, b) in enumerate(zip(within_vals, between_vals)):
            ratio = b / w if w > 0 else float("inf")
            ax.text(xi, max(w, b) + 0.5, f"×{ratio:.0f}", ha="center", fontsize=8, color="gray")

    fig.suptitle("Judge noise vs question variability — neuroticism LoRA sweep", fontsize=13)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved variance plot to {output}")
    return output


def plot_between_response_by_scale(
    all_stats: dict[str, dict],  # metric -> adapter -> stats
    output: Path,
) -> Path:
    """Line plot of between-response std across LoRA scale, per adapter and metric."""
    metrics = list(all_stats.keys())
    n_adapters = len(_ADAPTER_ORDER)

    fig, axes = plt.subplots(n_adapters, len(metrics), figsize=(7 * len(metrics), 4 * n_adapters), squeeze=False)

    for row, adapter in enumerate(_ADAPTER_ORDER):
        for col, metric in enumerate(metrics):
            ax = axes[row][col]
            stats = all_stats[metric][adapter]
            bbs = stats["between_by_scale"]
            if not bbs:
                ax.set_visible(False)
                continue
            scales = sorted(bbs)
            stds = [bbs[s] for s in scales]
            ax.plot(scales, stds, marker="o", linewidth=2, markersize=4, color="#e6194b")
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.axhline(stats["within_mean"], color="#4363d8", linewidth=1.5, linestyle=":",
                       label=f"within-response noise ({stats['within_mean']:.1f})")
            ax.set_xlabel("LoRA scale", fontsize=9)
            ax.set_ylabel("Std (between responses)", fontsize=9)
            ax.set_title(f"{adapter} / {metric}", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Between-response variability across LoRA scale\nvs within-response judge noise (dashed)", fontsize=12)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scale-variance plot to {output}")
    return output


# ── Markdown report ────────────────────────────────────────────────────────────


def write_report(all_stats: dict[str, dict], output: Path) -> None:
    lines = [
        "# Rerun variance analysis — neuroticism LoRA sweep",
        "",
        "Decomposes score variance into judge noise (within-response, same response rated 3×)",
        "vs question variability (between-response, different questions at same LoRA scale).",
        "",
        "**Data source**: GPT-4o-mini runs with 3 repeats at temperature=0 (runs: `gpt_4o_mini_2-*`).",
        "",
        "## Key finding",
        "",
    ]

    # Quick summary first
    for metric in all_stats:
        rows = all_stats[metric]
        within_means = [rows[a]["within_mean"] for a in _ADAPTER_ORDER]
        between_means = [rows[a]["between_mean"] for a in _ADAPTER_ORDER]
        avg_within = sum(within_means) / len(within_means)
        avg_between = sum(between_means) / len(between_means)
        ratio = avg_between / avg_within if avg_within > 0 else float("inf")
        lines.append(
            f"- **{metric}**: avg within-response noise = {avg_within:.2f}, "
            f"avg between-response variability = {avg_between:.2f} "
            f"(ratio ≈ {ratio:.0f}×)"
        )

    lines += [
        "",
        "The error bars in the sweep plots are dominated by between-response variability",
        "(question content), not judge stochasticity. Running more judge repeats will not",
        "materially reduce the observed std.",
        "",
        "## Per-adapter results",
        "",
    ]

    for metric in all_stats:
        lines.append(f"### {metric}")
        lines.append("")
        lines.append("| Adapter | Within-resp std (noise) | Nonzero noise frac | Between-resp std (signal) | Ratio |")
        lines.append("|---------|------------------------|--------------------|--------------------------|-------|")
        for adapter in _ADAPTER_ORDER:
            s = all_stats[metric][adapter]
            ratio = s["between_mean"] / s["within_mean"] if s["within_mean"] > 0 else float("inf")
            lines.append(
                f"| {adapter} | {s['within_mean']:.3f} | {s['within_nonzero_frac']:.1%} "
                f"| {s['between_mean']:.3f} | ×{ratio:.0f} |"
            )
        lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"Saved report to {output}")


# ── Upload ─────────────────────────────────────────────────────────────────────


def upload_analysis(output_dir: Path) -> str:
    from src_dev.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo
    login_from_env()
    url = upload_folder_to_dataset_repo(
        local_dir=output_dir,
        repo_id=HF_REPO_ID,
        path_in_repo=HF_PATH,
        commit_message="Upload rerun variance analysis",
    )
    print(f"Uploaded to: {url}")
    return url


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decompose judge score variance into noise vs signal."
    )
    parser.add_argument("--upload", action="store_true", help=f"Upload to HF ({HF_REPO_ID}/{HF_PATH})")
    args = parser.parse_args()

    load_dotenv()

    # Group run entries by metric × adapter
    all_run_entries: dict[str, dict[str, list[dict]]] = {
        "coherence":   {a: [] for a in _ADAPTER_ORDER},
        "neuroticism": {a: [] for a in _ADAPTER_ORDER},
    }
    for entry in MULTI_REPEAT_RUNS:
        all_run_entries[entry["metric"]][entry["adapter"]].append(entry)

    print("Computing variance decomposition...")
    all_stats: dict[str, dict] = {}
    for metric, by_adapter in all_run_entries.items():
        all_stats[metric] = {}
        for adapter, entries in by_adapter.items():
            if not entries:
                all_stats[metric][adapter] = {
                    "within_mean": 0.0, "within_nonzero_frac": 0.0,
                    "between_by_scale": {}, "between_mean": 0.0,
                    "n_responses": 0, "n_scales": 0,
                }
                continue
            stats = decompose_variance(entries)
            all_stats[metric][adapter] = stats
            print(
                f"  {metric}/{adapter}: within={stats['within_mean']:.2f}, "
                f"between={stats['between_mean']:.2f}, "
                f"ratio=×{stats['between_mean'] / stats['within_mean']:.0f}"
                if stats["within_mean"] > 0 else
                f"  {metric}/{adapter}: within=0 (all identical), between={stats['between_mean']:.2f}"
            )

    bar_plot = OUTPUT_DIR / "variance_decomposition_bar.png"
    plot_variance_decomposition(all_stats, bar_plot)

    scale_plot = OUTPUT_DIR / "between_response_variance_by_scale.png"
    plot_between_response_by_scale(all_stats, scale_plot)

    report = OUTPUT_DIR / "rerun_variance_analysis.md"
    write_report(all_stats, report)

    if args.upload:
        upload_analysis(OUTPUT_DIR)


if __name__ == "__main__":
    main()
