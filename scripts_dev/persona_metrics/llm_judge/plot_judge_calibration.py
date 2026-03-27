#!/usr/bin/env python3
"""Plot judge calibration figures from HuggingFace-hosted calibration logs.

Produces two figures:

  Figure 1 — Inter- and intra-model agreement
    A grouped bar chart showing, for each judge model × OCEAN trait:
      • Intra-model consistency (Krippendorff's alpha across 3 repeated runs)
      • Gold-vs-judge agreement (Spearman r, QWK) for each of the 3 models

  Figure 2 — Human-label vs judge heatmap (chosen model)
    A grid of confusion-matrix-style heatmaps (one per OCEAN trait) comparing
    the human gold scores to the chosen judge's median scores.  Each cell
    (i, j) shows the count of items where gold_score=i and judge_median=j.

Usage::

    python -m scripts_dev.persona_metrics.llm_judge.plot_judge_calibration \\
        --hf-base https://huggingface.co/datasets/persona-shattering-lasr/monorepo/resolve/main/judge_calibration \\
        --chosen-model google/gemini-2.0-flash-001 \\
        --output-dir scratch/judge_calibration_plots

    # Or with local data
    python -m scripts_dev.persona_metrics.llm_judge.plot_judge_calibration \\
        --local-dir scratch/judge_calibration \\
        --chosen-model google/gemini-2.0-flash-001 \\
        --output-dir scratch/judge_calibration_plots
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

OCEAN_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
TRAIT_LABELS = {
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "neuroticism": "Neuroticism",
}

# Short model display names
_MODEL_SHORT = {
    "google/gemini-2.0-flash-001": "Gemini 2.0 Flash",
    "openai/gpt-5-mini": "GPT-5 Mini",
    "moonshotai/kimi-k2": "Kimi K2",
}

# Run-key → model mapping (from the directory names on HF)
_RUNKEY_TO_MODEL = {
    "google_gemini-2.0-flash-001__r3__20260326T203008": "google/gemini-2.0-flash-001",
    "openai_gpt-5-mini__r3__20260326T220614": "openai/gpt-5-mini",
    "moonshotai_kimi-k2__r3__20260326T221255": "moonshotai/kimi-k2",
}

_MODEL_COLOURS = {
    "google/gemini-2.0-flash-001": "#4363d8",
    "openai/gpt-5-mini": "#e6194b",
    "moonshotai/kimi-k2": "#3cb44b",
}


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_json(path_or_url: str) -> dict:
    """Load JSON from a local file path or HTTP URL."""
    if path_or_url.startswith("http"):
        with urllib.request.urlopen(path_or_url) as resp:
            return json.loads(resp.read().decode())
    return json.loads(Path(path_or_url).read_text())


def _resolve(base: str, *parts: str) -> str:
    """Join base URL or path with subpath components."""
    sep = "/" if base.startswith("http") else None
    if sep:
        return base.rstrip("/") + "/" + "/".join(parts)
    return str(Path(base).joinpath(*parts))


def load_all_summaries(base: str) -> dict[str, dict]:
    """Load combined_summary.json for each run key.

    Args:
        base: HF resolve base URL or local directory containing run-key subdirs.

    Returns:
        ``{run_key: combined_summary_dict}``
    """
    summaries: dict[str, dict] = {}
    for run_key in _RUNKEY_TO_MODEL:
        url = _resolve(base, run_key, "analysis", "combined_summary.json")
        try:
            summaries[run_key] = _load_json(url)
        except Exception as exc:
            print(f"Warning: could not load {url}: {exc}")
    return summaries


def load_trait_summaries(base: str, run_key: str, traits: list[str]) -> dict[str, dict]:
    """Load per-trait summary JSON for a single run.

    Args:
        base: HF resolve base URL or local directory.
        run_key: Run key subdirectory name.
        traits: List of trait names to load.

    Returns:
        ``{trait: trait_summary_dict}``
    """
    result: dict[str, dict] = {}
    for trait in traits:
        url = _resolve(base, run_key, "analysis", f"{trait}_summary.json")
        try:
            result[trait] = _load_json(url)
        except Exception as exc:
            print(f"Warning: could not load {url}: {exc}")
    return result


# ── Figure 1: inter- and intra-model agreement ────────────────────────────────


def _short(model: str) -> str:
    return _MODEL_SHORT.get(model, model.split("/")[-1])


def plot_agreement_figure(
    summaries: dict[str, dict],
    output: Path,
) -> None:
    """Figure 1: grouped bar chart of intra- and inter-model agreement metrics.

    Two panels:
      Left  — Intra-model self-consistency (Krippendorff's alpha, 3 repeated runs)
      Right — Gold-vs-judge Spearman correlation (judge vs human labels)

    Each panel shows one bar group per trait, with one bar per model.

    Args:
        summaries: ``{run_key: combined_summary}`` as returned by load_all_summaries.
        output: Output PNG path.
    """
    models_ordered = [m for m in _MODEL_SHORT if any(
        _RUNKEY_TO_MODEL.get(rk) == m for rk in summaries
    )]
    traits = OCEAN_TRAITS

    # Collect arrays: shape (n_models, n_traits)
    alpha_matrix = np.full((len(models_ordered), len(traits)), np.nan)
    spearman_matrix = np.full((len(models_ordered), len(traits)), np.nan)
    qwk_matrix = np.full((len(models_ordered), len(traits)), np.nan)

    for rk, summary in summaries.items():
        model = _RUNKEY_TO_MODEL.get(rk)
        if model not in models_ordered:
            continue
        mi = models_ordered.index(model)
        for ti, trait in enumerate(traits):
            td = summary.get("per_trait", {}).get(trait)
            if td is None:
                continue
            alpha_matrix[mi, ti] = td.get("self_consistency_alpha", np.nan)
            spearman_matrix[mi, ti] = td.get("gold_vs_median_judge", {}).get("spearman", np.nan)
            qwk_matrix[mi, ti] = td.get("gold_vs_median_judge", {}).get("qwk", np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(traits))
    n_models = len(models_ordered)
    bar_w = 0.22
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_w

    def _draw_grouped_bars(ax, matrix, ylabel, title, ylim=(0.0, 1.05)):
        for mi, model in enumerate(models_ordered):
            colour = _MODEL_COLOURS.get(model, "#888888")
            vals = matrix[mi]
            bars = ax.bar(
                x + offsets[mi],
                vals,
                width=bar_w,
                color=colour,
                alpha=0.85,
                label=_short(model),
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([TRAIT_LABELS[t] for t in traits], fontsize=10, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(axis="y", alpha=0.35, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    _draw_grouped_bars(
        axes[0],
        alpha_matrix,
        ylabel="Krippendorff's α",
        title="Intra-model consistency\n(3 repeated runs)",
        ylim=(0.9, 1.02),
    )

    _draw_grouped_bars(
        axes[1],
        spearman_matrix,
        ylabel="Spearman ρ",
        title="Gold-vs-judge agreement\n(Spearman correlation)",
        ylim=(0.88, 1.02),
    )

    fig.suptitle(
        "Figure 1 — Judge calibration: inter- and intra-model agreement (OCEAN traits)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1 → {output}")


# ── Figure 2: human-label vs judge heatmap ────────────────────────────────────


def plot_human_vs_judge_heatmap(
    trait_summaries: dict[str, dict],
    model_name: str,
    output: Path,
) -> None:
    """Figure 2: confusion-matrix heatmaps of human gold scores vs judge medians.

    One subplot per OCEAN trait.  Each cell (row=gold, col=judge_median) is
    coloured by count.  Diagonal = perfect agreement.

    Args:
        trait_summaries: ``{trait: trait_summary_dict}`` for the chosen model.
        model_name: Display name of the chosen judge model.
        output: Output PNG path.
    """
    score_range = list(range(-4, 5))  # -4 … +4
    n_scores = len(score_range)
    idx_of = {s: i for i, s in enumerate(score_range)}

    traits = [t for t in OCEAN_TRAITS if t in trait_summaries]
    n_traits = len(traits)
    ncols = 3
    nrows = (n_traits + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes_flat = axes.flatten() if n_traits > 1 else [axes]

    vmax_global = 0
    mats = {}
    for trait in traits:
        mat = np.zeros((n_scores, n_scores), dtype=int)
        for item in trait_summaries[trait].get("per_item", []):
            g = item.get("gold_score")
            j = item.get("judge_median")
            if g is None or j is None:
                continue
            ri, ci = idx_of.get(g), idx_of.get(j)
            if ri is not None and ci is not None:
                mat[ri, ci] += 1
        mats[trait] = mat
        vmax_global = max(vmax_global, mat.max())

    for ti, trait in enumerate(traits):
        ax = axes_flat[ti]
        mat = mats[trait]

        im = ax.imshow(
            mat,
            aspect="equal",
            cmap="Blues",
            vmin=0,
            vmax=max(vmax_global, 1),
            origin="upper",
        )

        # Annotate each cell with count
        for ri in range(n_scores):
            for ci in range(n_scores):
                v = mat[ri, ci]
                if v > 0:
                    text_col = "white" if v > vmax_global * 0.55 else "#333333"
                    ax.text(ci, ri, str(v), ha="center", va="center", fontsize=8, color=text_col)

        ax.set_xticks(range(n_scores))
        ax.set_yticks(range(n_scores))
        ax.set_xticklabels(score_range, fontsize=8)
        ax.set_yticklabels(score_range, fontsize=8)
        ax.set_xlabel("Judge median score", fontsize=9)
        ax.set_ylabel("Human gold score", fontsize=9)
        ax.set_title(TRAIT_LABELS[trait], fontsize=11, fontweight="bold")

        # Diagonal guide
        ax.plot(
            [0 - 0.5, n_scores - 0.5],
            [0 - 0.5, n_scores - 0.5],
            color="#e6194b",
            linewidth=1.2,
            linestyle="--",
            alpha=0.6,
        )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")

    # Hide unused axes
    for ti in range(n_traits, nrows * ncols):
        axes_flat[ti].set_visible(False)

    # Compute summary stats for subtitle
    n_items = sum(
        len(trait_summaries[t].get("per_item", [])) for t in traits
    )
    mean_spearman = np.nanmean([
        trait_summaries[t].get("gold_vs_median_judge", {}).get("spearman", np.nan)
        for t in traits
    ])

    fig.suptitle(
        f"Figure 2 — Human-label vs judge agreement: {_short(model_name)}\n"
        f"({n_items} items across {len(traits)} OCEAN traits · "
        f"mean Spearman ρ = {mean_spearman:.3f})",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 2 → {output}")


# ── CLI ────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot judge calibration figures from HF-hosted calibration logs."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--hf-base",
        metavar="URL",
        help=(
            "HuggingFace resolve base URL, e.g. "
            "https://huggingface.co/datasets/persona-shattering-lasr/monorepo/resolve/main/judge_calibration"
        ),
    )
    source.add_argument(
        "--local-dir",
        type=Path,
        metavar="DIR",
        help="Local directory mirroring the judge_calibration folder structure.",
    )
    parser.add_argument(
        "--chosen-model",
        default="google/gemini-2.0-flash-001",
        help="Model ID to use for Figure 2 (default: google/gemini-2.0-flash-001).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scratch/judge_calibration_plots"),
        help="Directory to write PNG files (default: scratch/judge_calibration_plots).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base = args.hf_base if args.hf_base else str(args.local_dir)

    output_dir: Path = args.output_dir

    # ── Figure 1 ──────────────────────────────────────────────────────────────
    print("Loading combined summaries for all models …")
    summaries = load_all_summaries(base)
    if not summaries:
        raise RuntimeError("No summaries loaded — check --hf-base / --local-dir.")
    plot_agreement_figure(
        summaries=summaries,
        output=output_dir / "figure1_agreement.png",
    )

    # ── Figure 2 ──────────────────────────────────────────────────────────────
    chosen_run_key = next(
        (rk for rk, m in _RUNKEY_TO_MODEL.items() if m == args.chosen_model), None
    )
    if chosen_run_key is None:
        raise ValueError(
            f"Chosen model '{args.chosen_model}' not found in known run keys. "
            f"Known models: {list(_RUNKEY_TO_MODEL.values())}"
        )

    print(f"Loading per-trait summaries for '{args.chosen_model}' …")
    trait_summaries = load_trait_summaries(base, chosen_run_key, OCEAN_TRAITS)
    if not trait_summaries:
        raise RuntimeError("No trait summaries loaded.")
    plot_human_vs_judge_heatmap(
        trait_summaries=trait_summaries,
        model_name=args.chosen_model,
        output=output_dir / "figure2_human_vs_judge.png",
    )


if __name__ == "__main__":
    main()
