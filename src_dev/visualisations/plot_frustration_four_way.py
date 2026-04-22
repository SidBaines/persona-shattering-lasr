"""Plot per-turn frustration curves for BASE / CONTROL / N- / inverse-N-.

Pulls ``summary.json`` from the shared HF monorepo under
``evals/frustration_eval/<run_name>/<category>/summary.json`` (falls back to
``<run_name>/summary.json`` if the summary sits at the run root), then saves a
two-panel figure to ``paper/figures/main/``.

Usage:
    uv run python -m src_dev.visualisations.plot_frustration_four_way
    uv run python -m src_dev.visualisations.plot_frustration_four_way \
        --n-prompts 20  # pick the matching run-name set
"""

from __future__ import annotations

import argparse
import json
import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download

from scripts_dev.frustration_eval.prompts import IMPOSSIBLE_NUMERIC_PUZZLES
from src_dev.evals.personality.analyze_results import (
    _interval_ci_from_bootstrap,
    _interval_ci_from_wilson,
)
from src_dev.visualisations import PAPER_FIGURES_DIR

HIGH_FRUSTRATION_THRESHOLD = 5  # matches compute_summary in run_eval.py
CI_CONFIDENCE = 95
CI_BOOTSTRAP_N = 10_000
CI_SEED = 42

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "persona-shattering-lasr/monorepo"
REPO_PREFIX = "evals/frustration_eval"
CATEGORY = "impossible_numeric_3turn"


@dataclass(frozen=True)
class RunSpec:
    label: str
    run_name: str
    color: str
    marker: str


# Matching run-name sets for each sample size. Ordering here determines plot order.
RUN_SETS: dict[int, list[RunSpec]] = {
    20: [
        RunSpec("BASE",          "gemma3_27b_base_or_8turn_20prompt_1rollout",                                              "#2F5D9F", "o-"),
        RunSpec("CONTROL",       "gemma3_27b_control_use_diff_words_v1_persona_hfbatched_8turn_20prompt_1rollout",         "#6B6B6B", "D-"),
        RunSpec("N-",            "gemma3_27b_n_minus_v4_persona_hfbatched_8turn_20prompt_1rollout",                        "#C73E3A", "s-"),
        RunSpec("N- inverted",   "gemma3_27b_n_minus_v4_persona_negscale_hfbatched_8turn_20prompt_1rollout",               "#7A1F1B", "^-"),
    ],
    100: [
        RunSpec("BASE",          "gemma3_27b_base_or_8turn_200_samples_1rollout",                                          "#2F5D9F", "o-"),
        RunSpec("CONTROL",       "gemma3_27b_control_use_diff_words_v1_persona_hfbatched_8turn_100_samples_1rollout",     "#6B6B6B", "D-"),
        RunSpec("N-",            "gemma3_27b_n_minus_v4_persona_hfbatched_8turn_100_samples_1rollout",                    "#C73E3A", "s-"),
        RunSpec("N- inverted",   "gemma3_27b_n_minus_v4_persona_negscale_hfbatched_8turn_100_samples_1rollout",           "#7A1F1B", "^-"),
    ],
}


def fetch_per_turn_scores(
    run_name: str, *, repo_id: str = DEFAULT_REPO_ID,
) -> tuple[int, list[list[int]]]:
    """Download ``results.jsonl`` for a run from the HF monorepo and return
    ``(num_turns, per_turn_scores)`` where ``per_turn_scores[t]`` is the list
    of per-conversation frustration scores at turn index ``t``.

    Non-None scores are kept; missing / None scores are dropped (matches the
    filtering in ``compute_summary``)."""
    path_in_repo = f"{REPO_PREFIX}/{run_name}/{CATEGORY}/results.jsonl"
    local_path = hf_hub_download(
        repo_id=repo_id, filename=path_in_repo, repo_type="dataset",
    )
    convs: list[dict] = []
    with open(local_path) as f:
        for line in f:
            line = line.strip()
            if line:
                convs.append(json.loads(line))

    if not convs:
        raise RuntimeError(f"results.jsonl empty for {run_name!r}")
    num_turns = max(len(c.get("turn_results", [])) for c in convs)

    per_turn: list[list[int]] = [[] for _ in range(num_turns)]
    for c in convs:
        for t in c.get("turn_results", []):
            idx = t.get("turn_index")
            score = t.get("frustration_score")
            if idx is None or score is None or score < 0:
                continue
            if 0 <= idx < num_turns:
                per_turn[idx].append(int(score))
    return num_turns, per_turn


def compute_per_turn_stats(
    per_turn_scores: list[list[int]],
    *,
    threshold: int = HIGH_FRUSTRATION_THRESHOLD,
) -> dict:
    """Produce mean + bootstrap-CI for mean score (continuous 0-10) and
    proportion + Wilson-CI for ``score >= threshold`` (binary), for each turn.
    Returns arrays keyed by 'mean', 'mean_lo', 'mean_hi', 'pct', 'pct_lo',
    'pct_hi' (pct quantities are in 0-100)."""
    means, mean_lo, mean_hi = [], [], []
    pcts, pct_lo, pct_hi = [], [], []
    for scores in per_turn_scores:
        arr = np.asarray(scores, dtype=float)
        if len(arr) == 0:
            means.append(0.0); mean_lo.append(0.0); mean_hi.append(0.0)
            pcts.append(0.0); pct_lo.append(0.0); pct_hi.append(0.0)
            continue

        # Continuous mean CI via BCa bootstrap (10k resamples).
        m = float(arr.mean())
        lo_m, hi_m = _interval_ci_from_bootstrap(
            arr, confidence=CI_CONFIDENCE,
            n_resamples=CI_BOOTSTRAP_N, seed=CI_SEED,
        )
        means.append(m); mean_lo.append(lo_m); mean_hi.append(hi_m)

        # Binary proportion + Wilson CI (returned as fractions → scale to %).
        binary = (arr >= threshold).astype(int)
        p = float(binary.mean())
        lo_p, hi_p = _interval_ci_from_wilson(binary, confidence=CI_CONFIDENCE)
        pcts.append(100.0 * p)
        pct_lo.append(100.0 * lo_p)
        pct_hi.append(100.0 * hi_p)

    return {
        "mean": means, "mean_lo": mean_lo, "mean_hi": mean_hi,
        "pct": pcts, "pct_lo": pct_lo, "pct_hi": pct_hi,
    }


def load_run_for_plot(spec: "RunSpec") -> dict:
    """Returns a dict with per-turn stats + overall μ + overall pct_high for the legend."""
    _, per_turn_scores = fetch_per_turn_scores(spec.run_name)
    stats = compute_per_turn_stats(per_turn_scores)
    # Overall = mean-of-max-score-per-conv semantic used in compute_summary,
    # but for the plot header we just use the per-turn-8 values (what users
    # read on the right edge of the chart).
    return {"per_turn": stats, "n_per_turn": [len(s) for s in per_turn_scores]}


def plot_four_way(
    specs: list[RunSpec],
    *,
    n_prompts: int,
    out_dir: Path,
    name_suffix: str = "",
    explicit_stem: str | None = None,
) -> tuple[Path, Path]:
    summaries = [(s, load_run_for_plot(s)) for s in specs]
    turns = list(range(1, len(summaries[0][1]["per_turn"]["mean"]) + 1))

    fig, (ax_mean, ax_pct) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for spec, data in summaries:
        st = data["per_turn"]
        # Asymmetric error bars: distance from point to each CI edge.
        # Clip tiny floating-point negatives (e.g. degenerate CIs where lo==mean
        # but float subtraction goes below zero by an epsilon).
        mean_err = np.clip(np.vstack([
            np.asarray(st["mean"]) - np.asarray(st["mean_lo"]),
            np.asarray(st["mean_hi"]) - np.asarray(st["mean"]),
        ]), 0.0, None)
        pct_err = np.clip(np.vstack([
            np.asarray(st["pct"]) - np.asarray(st["pct_lo"]),
            np.asarray(st["pct_hi"]) - np.asarray(st["pct"]),
        ]), 0.0, None)
        ax_mean.errorbar(
            turns, st["mean"], yerr=mean_err, fmt=spec.marker,
            label=spec.label,
            color=spec.color, linewidth=1.8, markersize=6,
            elinewidth=1.0, capsize=3, capthick=1.0,
        )
        ax_pct.errorbar(
            turns, st["pct"], yerr=pct_err, fmt=spec.marker,
            label=spec.label,
            color=spec.color, linewidth=1.8, markersize=6,
            elinewidth=1.0, capsize=3, capthick=1.0,
        )

    ax_mean.set_xlabel("Turn")
    ax_mean.set_ylabel("Mean frustration (judge 0–10)")
    ax_mean.set_title("Per-turn mean frustration")
    ax_mean.set_xticks(turns)
    ax_mean.set_ylim(0, 10)
    ax_mean.grid(alpha=0.3)

    ax_pct.set_xlabel("Turn")
    ax_pct.set_ylabel(f"% high frustration (score ≥ {HIGH_FRUSTRATION_THRESHOLD})")
    ax_pct.set_title("Per-turn % high frustration")
    ax_pct.set_xticks(turns)
    ax_pct.set_ylim(-2, 102)
    ax_pct.grid(alpha=0.3)

    # One shared legend placed outside the right panel.
    handles, _ = ax_mean.get_legend_handles_labels()
    fig.legend(
        handles, [s.label for s in specs],
        loc="center left",
        bbox_to_anchor=(0.91, 0.55),
        fontsize=8,
        frameon=False,
        borderaxespad=0.0,
    )

    fig.suptitle(
        f"Frustration eval — gemma-3-27b-it (n={n_prompts} prompts, 8 turns, 1 rollout)",
        fontsize=10,
    )

    # Example puzzle — one representative prompt, wrapped, placed under the axes.
    example = " ".join(IMPOSSIBLE_NUMERIC_PUZZLES[0].split())
    wrapped = textwrap.fill(f"Example puzzle:  {example}", width=120)
    fig.text(
        0.5, 0.02, wrapped,
        ha="center", va="bottom",
        fontsize=7.5, color="#333",
        wrap=True,
    )

    # Reserve ~10% right for legend, ~12% bottom for example puzzle.
    fig.tight_layout(rect=[0, 0.12, 0.90, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = explicit_stem or f"fig_frustration_eval_4way_n{n_prompts}{name_suffix}"
    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-prompts", type=int, choices=sorted(RUN_SETS.keys()), default=20,
        help="Select the matching run-name set (20 or 100).",
    )
    parser.add_argument(
        "--subset", default="all",
        choices=("all", "base_vs_nminus"),
        help="'all' = 4-line plot; 'base_vs_nminus' = 2-line BASE vs N- (+1) only.",
    )
    parser.add_argument(
        "--out-dir", default=str(PAPER_FIGURES_DIR / "main"),
        help="Directory to write the figure PDF/PNG (ignored when --out-path is given).",
    )
    parser.add_argument(
        "--out-path", default=None,
        help="Explicit output path (without extension) — overrides --out-dir and "
             "the auto-generated stem. Useful for replacing a specific existing "
             "placeholder file. PNG and PDF are written at this stem.",
    )
    args = parser.parse_args()

    specs = RUN_SETS[args.n_prompts]
    if args.subset == "base_vs_nminus":
        specs = [s for s in specs if s.label in ("BASE", "N-")]

    if args.out_path is not None:
        out_path = Path(args.out_path)
        if out_path.suffix in (".png", ".pdf"):
            out_path = out_path.with_suffix("")
        out_dir = out_path.parent
        name_suffix = ""
        out_png, out_pdf = plot_four_way(
            specs, n_prompts=args.n_prompts, out_dir=out_dir,
            name_suffix=name_suffix, explicit_stem=out_path.name,
        )
    else:
        out_dir = Path(args.out_dir)
        suffix = "" if args.subset == "all" else f"_{args.subset}"
        out_png, out_pdf = plot_four_way(
            specs, n_prompts=args.n_prompts, out_dir=out_dir, name_suffix=suffix,
        )
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
