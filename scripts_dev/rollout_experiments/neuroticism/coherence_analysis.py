#!/usr/bin/env python3
"""Produce combined coherence plot and analysis .md for neuroticism LoRA sweep.

Aggregates scores across all judge runs per adapter:
- 6 GPT-4o-mini runs  (3 repeats × 2 run sessions, all via openrouter)
- 1 Gemini-flash-2.0 run (via openrouter)

The broken early runs (gpt_4o_mini-gemini_flash_20-*) had the gemini_flash_20
rater_id mapped to gpt-4o-mini — those are treated as extra GPT runs.

Usage::

    uv run python -m scripts_dev.rollout_experiments.neuroticism.coherence_analysis
    uv run python -m scripts_dev.rollout_experiments.neuroticism.coherence_analysis --upload
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
# Maps adapter -> (judge_run_dir, logical_rater_id)
# Broken runs had gemini_flash_20 rater_id but were actually GPT — remap them.

ADAPTER_RUNS: dict[str, list[tuple[str, str]]] = {
    "sft": [
        # broken runs: rater_id in file is wrong — both files are GPT
        ("gpt_4o_mini-gemini_flash_20-350b967bc906", "gpt_4o_mini"),
        ("gpt_4o_mini-gemini_flash_20-350b967bc906", "gemini_flash_20"),   # actually GPT
        ("gpt_4o_mini-gemini_flash_20-e4e206586c8f", "gpt_4o_mini"),
        ("gpt_4o_mini-gemini_flash_20-e4e206586c8f", "gemini_flash_20"),   # actually GPT
        # renamed run (both files are GPT)
        ("gpt_4o_mini-gpt_4o_mini_2-ad544c5585eb", "gpt_4o_mini"),
        ("gpt_4o_mini-gpt_4o_mini_2-ad544c5585eb", "gpt_4o_mini_2"),
        # new gemini run
        ("gemini_flash_20-3023eeeaa3ba", "gemini_flash_20"),
    ],
    "dpo": [
        ("gpt_4o_mini-gemini_flash_20-02524955a8f4", "gpt_4o_mini"),
        ("gpt_4o_mini-gemini_flash_20-02524955a8f4", "gemini_flash_20"),
        ("gpt_4o_mini-gemini_flash_20-7b763e3793a5", "gpt_4o_mini"),
        ("gpt_4o_mini-gemini_flash_20-7b763e3793a5", "gemini_flash_20"),
        ("gpt_4o_mini-gpt_4o_mini_2-99a779f80e93", "gpt_4o_mini"),
        ("gpt_4o_mini-gpt_4o_mini_2-99a779f80e93", "gpt_4o_mini_2"),
        ("gemini_flash_20-da644a7801ef", "gemini_flash_20"),
    ],
    "soup": [
        ("gpt_4o_mini-gemini_flash_20-5d04426ae5c0", "gpt_4o_mini"),
        ("gpt_4o_mini-gemini_flash_20-5d04426ae5c0", "gemini_flash_20"),
        ("gpt_4o_mini-gemini_flash_20-61af2c26f795", "gpt_4o_mini"),
        ("gpt_4o_mini-gemini_flash_20-61af2c26f795", "gemini_flash_20"),
        ("gpt_4o_mini-gpt_4o_mini_2-c4bf45cc735d", "gpt_4o_mini"),
        ("gpt_4o_mini-gpt_4o_mini_2-c4bf45cc735d", "gpt_4o_mini_2"),
        ("gemini_flash_20-6fee2dd68d9a", "gemini_flash_20"),
    ],
    "old": [
        ("gpt_4o_mini-gemini_flash_20-30211aa3a32c", "gpt_4o_mini"),
        ("gpt_4o_mini-gemini_flash_20-30211aa3a32c", "gemini_flash_20"),
        ("gpt_4o_mini-gemini_flash_20-bfa0ba2e2f8b", "gpt_4o_mini"),
        ("gpt_4o_mini-gemini_flash_20-bfa0ba2e2f8b", "gemini_flash_20"),
        ("gpt_4o_mini-gpt_4o_mini_2-18c0fe6a4e1d", "gpt_4o_mini"),
        ("gpt_4o_mini-gpt_4o_mini_2-18c0fe6a4e1d", "gpt_4o_mini_2"),
        ("gemini_flash_20-da079785aee6", "gemini_flash_20"),
    ],
}

# GPT rater_ids that are actually gpt-4o-mini (broken runs mislabelled)
_GPT_RATER_IDS = {"gpt_4o_mini", "gpt_4o_mini_2", "gemini_flash_20_as_gpt"}
# The gemini_flash_20 rater_id in broken runs needs special treatment — we
# detect it by checking the run dir name.
_BROKEN_RUNS = {
    "gpt_4o_mini-gemini_flash_20-350b967bc906",
    "gpt_4o_mini-gemini_flash_20-e4e206586c8f",
    "gpt_4o_mini-gemini_flash_20-02524955a8f4",
    "gpt_4o_mini-gemini_flash_20-7b763e3793a5",
    "gpt_4o_mini-gemini_flash_20-5d04426ae5c0",
    "gpt_4o_mini-gemini_flash_20-61af2c26f795",
    "gpt_4o_mini-gemini_flash_20-30211aa3a32c",
    "gpt_4o_mini-gemini_flash_20-bfa0ba2e2f8b",
}

JUDGE_RUNS_ROOT = Path("scratch/judge_runs")
OUTPUT_DIR = Path("scratch/coherence_analysis")
HF_REPO_ID = "persona-shattering-lasr/ocean_judge_runs"
HF_PATH = "coherence_lora_sweep/analysis"

_SCALE_RE = re.compile(r"@scale_([+-]?\d+(?:\.\d+)?)")

# ── Data loading ───────────────────────────────────────────────────────────────


def _is_gpt_rater(run_dir_name: str, rater_id: str) -> bool:
    if run_dir_name in _BROKEN_RUNS and rater_id == "gemini_flash_20":
        return True
    return rater_id in {"gpt_4o_mini", "gpt_4o_mini_2"}


def load_scores_for_adapter(
    adapter: str,
) -> dict[str, dict[float, list[float]]]:
    """Load and merge scores from all runs for an adapter.

    Returns:
        ``{"gpt_4o_mini": {scale: [scores]}, "gemini_flash_20": {scale: [scores]}}``
    """
    merged: dict[str, dict[float, list[float]]] = {
        "gpt_4o_mini": defaultdict(list),
        "gemini_flash_20": defaultdict(list),
    }

    for run_dir_name, rater_id in ADAPTER_RUNS[adapter]:
        run_dir = JUDGE_RUNS_ROOT / run_dir_name
        raw_file = run_dir / "judge_calls" / "raw" / f"{rater_id}.jsonl"
        if not raw_file.exists():
            print(f"  WARNING: missing {raw_file}")
            continue

        logical = "gpt_4o_mini" if _is_gpt_rater(run_dir_name, rater_id) else "gemini_flash_20"

        for line in raw_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") != "success" or rec.get("score") is None:
                continue
            m = _SCALE_RE.search(rec.get("condition", ""))
            if m:
                merged[logical][float(m.group(1))].append(float(rec["score"]))

    return {k: dict(v) for k, v in merged.items()}


# ── Stats ──────────────────────────────────────────────────────────────────────


def _std(scores: list[float]) -> float:
    n = len(scores)
    if n < 2:
        return 0.0
    mean = sum(scores) / n
    return math.sqrt(sum((x - mean) ** 2 for x in scores) / (n - 1))


def compute_scale_stats(
    scores_by_scale: dict[float, list[float]],
) -> dict[float, dict]:
    return {
        scale: {
            "n": len(scores),
            "mean": sum(scores) / len(scores),
            "median": sorted(scores)[len(scores) // 2],
            "std": _std(scores),
        }
        for scale, scores in sorted(scores_by_scale.items())
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

_RATER_STYLE = {
    "gpt_4o_mini":    {"color": "#4363d8", "marker": "o", "linestyle": "-",  "label": "GPT-4o-mini (×6)"},
    "gemini_flash_20": {"color": "#e6194b", "marker": "s", "linestyle": "--", "label": "Gemini Flash 2.0 (×1)"},
}
_ADAPTER_ORDER = ["sft", "dpo", "soup", "old"]


def plot_all_adapters(
    all_scores: dict[str, dict[str, dict[float, list[float]]]],
    output: Path,
    title: str = "Neuroticism coherence sweep — all adapters",
) -> Path:
    n = len(_ADAPTER_ORDER)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

    for i, adapter in enumerate(_ADAPTER_ORDER):
        ax = axes[i // ncols][i % ncols]
        data = all_scores[adapter]

        for rater_id, style in _RATER_STYLE.items():
            scores_by_scale = data.get(rater_id, {})
            if not scores_by_scale:
                continue
            scales = sorted(scores_by_scale)
            means = [sum(scores_by_scale[s]) / len(scores_by_scale[s]) for s in scales]
            cis = [_std(scores_by_scale[s]) for s in scales]

            ax.plot(scales, means, marker=style["marker"], linestyle=style["linestyle"],
                    color=style["color"], label=style["label"], linewidth=2, markersize=5)
            if any(ci > 0 for ci in cis):
                ax.errorbar(scales, means, yerr=cis, fmt="none", color=style["color"],
                            capsize=3, capthick=1, elinewidth=1, alpha=0.5)

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("LoRA scale factor", fontsize=10)
        ax.set_ylabel("Coherence score (mean ± 1 std)", fontsize=10)
        ax.set_title(f"Adapter: {adapter}", fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output}")
    return output


# ── Analysis markdown ──────────────────────────────────────────────────────────


def _monotonicity(means: list[float]) -> float:
    """Fraction of consecutive pairs that are non-decreasing."""
    if len(means) < 2:
        return float("nan")
    pairs = [(means[i], means[i + 1]) for i in range(len(means) - 1)]
    return sum(a <= b for a, b in pairs) / len(pairs)


def write_analysis_md(
    all_scores: dict[str, dict[str, dict[float, list[float]]]],
    output: Path,
) -> None:
    lines = [
        "# Coherence sweep analysis — neuroticism LoRA adapters",
        "",
        "Coherence scores from the LLM judge panel across 17 LoRA scale points (−2.0 to +2.0, step 0.25).",
        "Higher scale = stronger neuroticism persona injection.",
        "",
        "## Judge panel",
        "",
        "| Rater | Model | Runs | Repeats/run |",
        "|-------|-------|------|-------------|",
        "| GPT-4o-mini | openai/gpt-4o-mini via OpenRouter | 6 | 3 (temp=0) |",
        "| Gemini Flash 2.0 | google/gemini-2.0-flash-001 via OpenRouter | 1 | 1 (temp=0) |",
        "",
        "Note: early runs mislabelled the Gemini rater but actually used GPT — all 6 GPT runs confirmed to use the same model.",
        "",
        "## Per-adapter summary",
        "",
    ]

    for adapter in _ADAPTER_ORDER:
        data = all_scores[adapter]
        lines.append(f"### {adapter}")
        lines.append("")
        lines.append("| Rater | Scale | N | Mean | Median | Std |")
        lines.append("|-------|-------|---|------|--------|-----|")

        for rater_id, label in [("gpt_4o_mini", "GPT-4o-mini"), ("gemini_flash_20", "Gemini Flash 2.0")]:
            stats = compute_scale_stats(data.get(rater_id, {}))
            for scale, s in sorted(stats.items()):
                lines.append(
                    f"| {label} | {scale:+.2f} | {s['n']} "
                    f"| {s['mean']:.3f} | {s['median']:.3f} "
                    f"| {s['std']:.3f} |"
                )
        lines.append("")

        # Trend summary
        lines.append("**Trend (scale monotonicity):**")
        lines.append("")
        for rater_id, label in [("gpt_4o_mini", "GPT-4o-mini"), ("gemini_flash_20", "Gemini Flash 2.0")]:
            stats = compute_scale_stats(data.get(rater_id, {}))
            if not stats:
                continue
            sorted_scales = sorted(stats)
            means = [stats[s]["mean"] for s in sorted_scales]
            mono = _monotonicity(means)
            mean_at_neg2 = stats.get(-2.0, {}).get("mean", float("nan"))
            mean_at_0 = stats.get(0.0, {}).get("mean", float("nan"))
            mean_at_pos2 = stats.get(2.0, {}).get("mean", float("nan"))
            lines.append(
                f"- **{label}**: monotonicity={mono:.0%}, "
                f"score @ −2.0={mean_at_neg2:.2f}, @ 0.0={mean_at_0:.2f}, @ +2.0={mean_at_pos2:.2f}"
            )
        lines.append("")

    # Cross-adapter comparison
    lines += [
        "## Cross-adapter comparison",
        "",
        "Mean coherence score at scale=0 (neutral persona) and scale=+2.0 (max neuroticism):",
        "",
        "| Adapter | Rater | Score @ 0.0 | Score @ +2.0 | Drop |",
        "|---------|-------|-------------|--------------|------|",
    ]
    for adapter in _ADAPTER_ORDER:
        data = all_scores[adapter]
        for rater_id, label in [("gpt_4o_mini", "GPT"), ("gemini_flash_20", "Gemini")]:
            stats = compute_scale_stats(data.get(rater_id, {}))
            s0 = stats.get(0.0, {}).get("mean", float("nan"))
            s2 = stats.get(2.0, {}).get("mean", float("nan"))
            drop = s2 - s0 if not (math.isnan(s0) or math.isnan(s2)) else float("nan")
            lines.append(f"| {adapter} | {label} | {s0:.3f} | {s2:.3f} | {drop:+.3f} |")
    lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"Saved analysis to {output}")


# ── Upload ─────────────────────────────────────────────────────────────────────


def upload_analysis(output_dir: Path) -> str:
    from src_dev.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo
    login_from_env()
    url = upload_folder_to_dataset_repo(
        local_dir=output_dir,
        repo_id=HF_REPO_ID,
        path_in_repo=HF_PATH,
        commit_message="Upload neuroticism coherence sweep analysis",
    )
    print(f"Uploaded to: {url}")
    return url


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce combined coherence plot and analysis for neuroticism sweep."
    )
    parser.add_argument("--upload", action="store_true", help=f"Upload to HF ({HF_REPO_ID}/{HF_PATH})")
    args = parser.parse_args()

    load_dotenv()

    print("Loading scores...")
    all_scores = {adapter: load_scores_for_adapter(adapter) for adapter in _ADAPTER_ORDER}

    # Print quick summary
    for adapter in _ADAPTER_ORDER:
        for rater, scores_by_scale in all_scores[adapter].items():
            total = sum(len(v) for v in scores_by_scale.values())
            print(f"  {adapter}/{rater}: {len(scores_by_scale)} scales, {total} scores total")

    plot_output = OUTPUT_DIR / "coherence_sweep_all_adapters.png"
    plot_all_adapters(all_scores, plot_output)

    md_output = OUTPUT_DIR / "coherence_analysis.md"
    write_analysis_md(all_scores, md_output)

    if args.upload:
        upload_analysis(OUTPUT_DIR)


if __name__ == "__main__":
    main()
