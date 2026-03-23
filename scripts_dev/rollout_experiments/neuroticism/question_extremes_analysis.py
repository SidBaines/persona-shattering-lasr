#!/usr/bin/env python3
"""Question-level extremes analysis for neuroticism/coherence sweeps.

Joins judge scores back to the judge dataset (which contains question + response
text) to identify which questions/responses receive the highest and lowest scores
at each LoRA scale.

Two analyses:
1. **Cross-scale stability**: which questions consistently score high/low across
   all scales? These reveal questions that are inherently more/less trait-expressive
   regardless of the persona injection.
2. **Scale-sensitive extremes**: at a given scale (default: +2.0), which
   questions produce the most/least trait-expressive responses? Prints the
   question + response text for inspection.

Usage::

    # Full analysis — both metrics, old adapter
    uv run python -m scripts_dev.rollout_experiments.neuroticism.question_extremes_analysis

    # Specific adapter and metric
    uv run python -m scripts_dev.rollout_experiments.neuroticism.question_extremes_analysis --adapter old --metric coherence

    # Inspect more extremes
    uv run python -m scripts_dev.rollout_experiments.neuroticism.question_extremes_analysis --top-n 10

    # Save full HTML report
    uv run python -m scripts_dev.rollout_experiments.neuroticism.question_extremes_analysis --save
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

# ── Configuration ──────────────────────────────────────────────────────────────

JUDGE_RUNS_ROOT = Path("scratch/judge_runs")
JUDGE_DATASETS: dict[str, Path] = {
    "sft":  Path("scratch/judge_datasets/neuroticism_sft_sweep.jsonl"),
    "dpo":  Path("scratch/judge_datasets/neuroticism_dpo_sweep.jsonl"),
    "soup": Path("scratch/judge_datasets/neuroticism_soup_sweep.jsonl"),
    "old":  Path("scratch/judge_datasets/neuroticism_old_sweep.jsonl"),
}
OUTPUT_DIR = Path("scratch/question_extremes_analysis")
HF_REPO_ID = "persona-shattering-lasr/ocean_judge_runs"
HF_PATH = "question_extremes_analysis"

_SCALE_RE = re.compile(r"@scale_([+-]?\d+(?:\.\d+)?)")
_ADAPTER_ORDER = ["sft", "dpo", "soup", "old"]

# Best single-rater runs per (adapter, metric) — prefer multi-repeat runs for
# averaging, fall back to single-repeat runs.
# These are the same runs used in coherence_analysis.py for Gemini (single rater,
# clean signal) plus the neuroticism gpt runs.
SCORE_RUNS: dict[str, dict[str, list[tuple[str, str]]]] = {
    "coherence": {
        # Gemini run: single file, 1 repeat, clean rater
        "sft":  [("gemini_flash_20-3023eeeaa3ba", "gemini_flash_20")],
        "dpo":  [("gemini_flash_20-da644a7801ef", "gemini_flash_20")],
        "soup": [("gemini_flash_20-6fee2dd68d9a", "gemini_flash_20")],
        "old":  [("gemini_flash_20-da079785aee6", "gemini_flash_20")],
    },
    "neuroticism": {
        # GPT runs (both raters are actually GPT in these broken-label runs)
        "sft":  [("gpt_4o_mini-gemini_flash_20-350b967bc906", "gpt_4o_mini"),
                 ("gpt_4o_mini-gemini_flash_20-350b967bc906", "gemini_flash_20")],
        "dpo":  [("gpt_4o_mini-gemini_flash_20-02524955a8f4", "gpt_4o_mini"),
                 ("gpt_4o_mini-gemini_flash_20-02524955a8f4", "gemini_flash_20")],
        "soup": [("gpt_4o_mini-gemini_flash_20-5d04426ae5c0", "gpt_4o_mini"),
                 ("gpt_4o_mini-gemini_flash_20-5d04426ae5c0", "gemini_flash_20")],
        "old":  [("gpt_4o_mini-gemini_flash_20-30211aa3a32c", "gpt_4o_mini"),
                 ("gpt_4o_mini-gemini_flash_20-30211aa3a32c", "gemini_flash_20")],
    },
}


# ── Data loading ───────────────────────────────────────────────────────────────


def load_judge_dataset(adapter: str) -> dict[str, dict]:
    """Load judge dataset as {response_id: record}."""
    path = JUDGE_DATASETS[adapter]
    if not path.exists():
        raise FileNotFoundError(f"Judge dataset not found: {path}")
    result = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            result[rec["response_id"]] = rec
    return result


def load_scores(adapter: str, metric: str) -> dict[str, float]:
    """Load mean score per response_id (averaged across all runs/repeats)."""
    runs = SCORE_RUNS[metric][adapter]
    by_response: dict[str, list[float]] = defaultdict(list)

    for run_dir, rater_id in runs:
        path = JUDGE_RUNS_ROOT / run_dir / "judge_calls" / "raw" / f"{rater_id}.jsonl"
        if not path.exists():
            print(f"  WARNING: missing {path}")
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") != "success" or rec.get("score") is None:
                continue
            by_response[rec["response_id"]].append(float(rec["score"]))

    return {
        resp_id: sum(scores) / len(scores)
        for resp_id, scores in by_response.items()
    }


# ── Analysis ───────────────────────────────────────────────────────────────────


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))


def build_question_profiles(
    scores: dict[str, float],
    dataset: dict[str, dict],
) -> dict[str, dict]:
    """Build per-question (sample_id) score profiles across scales.

    Returns:
        {sample_id: {
            "question": str,
            "by_scale": {scale: score},
            "mean_score": float,
            "std_score": float,
            "scale_sensitivity": float,  # max - min across scales
        }}
    """
    # Group by sample_id
    by_sample: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))

    for resp_id, score in scores.items():
        m = _SCALE_RE.search(resp_id)
        if not m:
            continue
        scale = float(m.group(1))
        ds_rec = dataset.get(resp_id)
        if ds_rec is None:
            continue
        sample_id = ds_rec["sample_id"]
        by_sample[sample_id][scale].append(score)

    profiles: dict[str, dict] = {}
    for sample_id, scale_map in by_sample.items():
        # Get question from any record with this sample_id
        question = ""
        for resp_id, ds_rec in dataset.items():
            if ds_rec["sample_id"] == sample_id:
                question = ds_rec.get("question", "")
                break

        by_scale = {s: sum(v) / len(v) for s, v in scale_map.items()}
        all_scores = list(by_scale.values())
        profiles[sample_id] = {
            "question": question,
            "by_scale": by_scale,
            "mean_score": sum(all_scores) / len(all_scores),
            "std_score": _std(all_scores),
            "scale_sensitivity": max(all_scores) - min(all_scores),
        }

    return profiles


def get_responses_at_scale(
    scale: float,
    scores: dict[str, float],
    dataset: dict[str, dict],
    n: int = 5,
    highest: bool = True,
) -> list[dict]:
    """Return the top-n or bottom-n scored responses at a given scale."""
    candidates = []
    for resp_id, score in scores.items():
        m = _SCALE_RE.search(resp_id)
        if not m or abs(float(m.group(1)) - scale) > 0.01:
            continue
        ds_rec = dataset.get(resp_id)
        if ds_rec is None:
            continue
        candidates.append({
            "response_id": resp_id,
            "score": score,
            "question": ds_rec.get("question", ""),
            "response": ds_rec.get("response", ""),
            "scale": scale,
        })

    candidates.sort(key=lambda x: x["score"], reverse=highest)
    return candidates[:n]


# ── Plotting ───────────────────────────────────────────────────────────────────


def plot_score_distribution_by_question_rank(
    profiles: dict[str, dict],
    adapter: str,
    metric: str,
    output: Path,
    top_n: int = 10,
) -> Path:
    """Plot score-vs-scale curves for highest/lowest baseline questions."""
    # Rank questions by their mean score at scale=0
    at_zero = {
        sid: p["by_scale"].get(0.0, float("nan"))
        for sid, p in profiles.items()
        if not math.isnan(p["by_scale"].get(0.0, float("nan")))
    }
    sorted_sids = sorted(at_zero, key=lambda s: at_zero[s], reverse=True)
    top_sids = sorted_sids[:top_n]
    bottom_sids = sorted_sids[-top_n:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, group, label, color in [
        (axes[0], top_sids, f"Top-{top_n} (highest @ scale=0)", "#e6194b"),
        (axes[1], bottom_sids, f"Bottom-{top_n} (lowest @ scale=0)", "#4363d8"),
    ]:
        for sid in group:
            p = profiles[sid]
            scales = sorted(p["by_scale"])
            vals = [p["by_scale"][s] for s in scales]
            q_short = p["question"][:40] + "..." if len(p["question"]) > 40 else p["question"]
            ax.plot(scales, vals, marker=".", linewidth=1, alpha=0.7, label=q_short)

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("LoRA scale")
        ax.set_ylabel(f"{metric} score")
        ax.set_title(f"{adapter} / {metric} — {label}")
        ax.grid(True, alpha=0.3)
        # Skip legend (too crowded) — questions are in the text report

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved question profile plot to {output}")
    return output


# ── Text report ────────────────────────────────────────────────────────────────


def _truncate(s: str, n: int = 600) -> str:
    return s if len(s) <= n else s[:n] + "…"


def write_extremes_report(
    scores: dict[str, float],
    dataset: dict[str, dict],
    profiles: dict[str, dict],
    adapter: str,
    metric: str,
    output: Path,
    top_n: int = 5,
    inspect_scales: tuple[float, ...] = (-2.0, 0.0, 2.0),
) -> None:
    lines = [
        f"# Question extremes — {adapter} / {metric}",
        "",
        "Joins judge scores back to question+response text to find which questions produce",
        "the most/least trait-expressive responses.",
        "",
    ]

    # 1. Most and least scale-sensitive questions
    sorted_by_sensitivity = sorted(
        profiles.items(),
        key=lambda kv: kv[1]["scale_sensitivity"],
        reverse=True,
    )
    lines += [
        "## Most scale-sensitive questions",
        "",
        "These questions show the largest swing in score from scale −2.0 to +2.0.",
        "They are the best candidates for understanding what makes a response more/less neurotic.",
        "",
    ]
    for sid, p in sorted_by_sensitivity[:top_n]:
        lines.append(f"### Q: {p['question']}")
        lines.append(f"- Scale sensitivity (max−min): **{p['scale_sensitivity']:.1f}**")
        lines.append(f"- Mean score across scales: {p['mean_score']:.1f}  |  Std: {p['std_score']:.1f}")
        lines.append(f"- Score by scale: " + ", ".join(
            f"{s:+.2f}→{p['by_scale'][s]:.0f}" for s in sorted(p["by_scale"])
        ))
        lines.append("")

    lines += [
        "## Least scale-sensitive questions",
        "",
        "These questions show almost no change across scales — trait injection barely affects them.",
        "",
    ]
    for sid, p in sorted_by_sensitivity[-top_n:]:
        lines.append(f"### Q: {p['question']}")
        lines.append(f"- Scale sensitivity (max−min): **{p['scale_sensitivity']:.1f}**")
        lines.append(f"- Mean score across scales: {p['mean_score']:.1f}  |  Std: {p['std_score']:.1f}")
        lines.append("")

    # 2. Top/bottom at specific scales
    for scale in inspect_scales:
        lines += [
            f"## Scale {scale:+.1f} — highest and lowest scoring responses",
            "",
        ]

        for extreme, label in [(True, f"Highest {metric} @ {scale:+.1f}"), (False, f"Lowest {metric} @ {scale:+.1f}")]:
            items = get_responses_at_scale(scale, scores, dataset, n=top_n, highest=extreme)
            lines.append(f"### {label}")
            lines.append("")
            for item in items:
                lines.append(f"**Score: {item['score']:.0f}**  |  Q: {item['question']}")
                lines.append("")
                lines.append(f"> {_truncate(item['response'])}")
                lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"Saved extremes report to {output}")


# ── Upload ─────────────────────────────────────────────────────────────────────


def upload_analysis(output_dir: Path) -> str:
    from src_dev.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo
    login_from_env()
    url = upload_folder_to_dataset_repo(
        local_dir=output_dir,
        repo_id=HF_REPO_ID,
        path_in_repo=HF_PATH,
        commit_message="Upload question extremes analysis",
    )
    print(f"Uploaded to: {url}")
    return url


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Question-level extremes analysis for sweep judge scores."
    )
    parser.add_argument(
        "--adapter", choices=list(JUDGE_DATASETS.keys()) + ["all"], default="all",
        help="Adapter(s) to analyse (default: all).",
    )
    parser.add_argument(
        "--metric", choices=["neuroticism", "coherence", "both"], default="both",
        help="Metric to analyse (default: both).",
    )
    parser.add_argument(
        "--top-n", type=int, default=5,
        help="Number of extreme examples to show (default: 5).",
    )
    parser.add_argument("--save", action="store_true",
                        help="Save all outputs to disk (plots + reports).")
    parser.add_argument("--upload", action="store_true",
                        help=f"Upload saved outputs to HF ({HF_REPO_ID}/{HF_PATH}).")
    args = parser.parse_args()

    load_dotenv()

    adapters = list(JUDGE_DATASETS.keys()) if args.adapter == "all" else [args.adapter]
    metrics = ["neuroticism", "coherence"] if args.metric == "both" else [args.metric]

    for adapter in adapters:
        print(f"\nLoading dataset for {adapter}...")
        dataset = load_judge_dataset(adapter)
        print(f"  {len(dataset)} records")

        for metric in metrics:
            print(f"\n  Loading {metric} scores for {adapter}...")
            scores = load_scores(adapter, metric)
            if not scores:
                print(f"  No scores found — skipping.")
                continue
            print(f"  {len(scores)} scored responses")

            profiles = build_question_profiles(scores, dataset)
            print(f"  {len(profiles)} unique questions")

            if args.save:
                plot_output = OUTPUT_DIR / f"{adapter}_{metric}_question_profiles.png"
                plot_score_distribution_by_question_rank(
                    profiles, adapter, metric, plot_output, top_n=args.top_n
                )

                report_output = OUTPUT_DIR / f"{adapter}_{metric}_extremes.md"
                write_extremes_report(
                    scores, dataset, profiles, adapter, metric, report_output,
                    top_n=args.top_n,
                )
            else:
                # Print a quick terminal summary
                sorted_by_sensitivity = sorted(
                    profiles.items(),
                    key=lambda kv: kv[1]["scale_sensitivity"],
                    reverse=True,
                )
                print(f"\n  === {adapter} / {metric} — Top-{args.top_n} most scale-sensitive questions ===")
                for sid, p in sorted_by_sensitivity[:args.top_n]:
                    print(f"    sensitivity={p['scale_sensitivity']:.1f}  mean={p['mean_score']:.1f}  Q: {p['question'][:80]}")

                print(f"\n  === {adapter} / {metric} — Highest {metric} @ scale=+2.0 ===")
                for item in get_responses_at_scale(2.0, scores, dataset, n=args.top_n, highest=True):
                    print(f"    score={item['score']:.0f}  Q: {item['question'][:60]}")
                    print(f"    R: {item['response'][:120].replace(chr(10), ' ')}...")

                print(f"\n  === {adapter} / {metric} — Lowest {metric} @ scale=+2.0 ===")
                for item in get_responses_at_scale(2.0, scores, dataset, n=args.top_n, highest=False):
                    print(f"    score={item['score']:.0f}  Q: {item['question'][:60]}")
                    print(f"    R: {item['response'][:120].replace(chr(10), ' ')}...")

    if args.upload and args.save:
        upload_analysis(OUTPUT_DIR)
    elif args.upload and not args.save:
        print("WARNING: --upload requires --save. Add --save to persist outputs first.")


if __name__ == "__main__":
    main()
