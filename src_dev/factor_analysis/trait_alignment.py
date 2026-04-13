"""Trait-oriented FA matrix + factor-trait alignment analysis.

Purpose:
- Build a response matrix where each cell is a *trait-direction* score, not a
  shuffled letter-rank. For each (persona k, trait_mcq item i) cell we compute
  ``Σ_letter P(letter) * answer_mapping[letter]`` when ``probs`` is available
  (logprob mode), or the binary ``answer_mapping[parsed_choice]`` otherwise.
  This sidesteps the shuffling-invalidates-sign issue in the canonical
  letter-encoded matrix (A=1..D=4) because each item's answer_mapping already
  re-orients letters to trait direction.
- Produce a factor-trait alignment summary from FA loadings obtained on that
  matrix: top-K items per factor (counts by trait) and mean-loading
  per trait per factor (signed + unsigned).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src_dev.factor_analysis.trait_scoring import _load_questionnaire_trait_mcq_items


@dataclass
class TraitOrientedMatrix:
    """Trait-oriented response matrix plus item metadata aligned to columns.

    Attributes:
        matrix: [n_personas × n_items] float, NaN for missing cells.
        k_index: list of persona indices (row order of ``matrix``).
        item_ids: list of item IDs (column order).
        item_dims: list of primary_dimension strings (one per column).
        trait_order: canonical trait ordering (OCEAN).
    """

    matrix: np.ndarray
    k_index: list[int]
    item_ids: list[str]
    item_dims: list[str]
    trait_order: list[str]


def build_trait_oriented_matrix(
    raw_responses_path: Path | str,
    questionnaire_path: Path | str,
    *,
    trait_order: list[str] | None = None,
) -> TraitOrientedMatrix:
    """Build a [n_personas × n_items] trait-oriented matrix from raw responses.

    Each cell = ``Σ P(letter) · answer_mapping[letter]`` when ``probs`` is
    present (logprob mode), else ``answer_mapping[parsed_choice]`` (binary).
    Non-trait_mcq entries and entries with un-mappable choices are skipped.

    Args:
        raw_responses_path: Path to stage-2 ``raw_responses.jsonl``.
        questionnaire_path: Path to questionnaire JSON (for answer_mapping +
            primary_dimension).
        trait_order: Optional explicit trait ordering. Defaults to OCEAN order
            restricted to traits present.

    Returns:
        ``TraitOrientedMatrix``.
    """
    raw_responses_path = Path(raw_responses_path)
    questionnaire_path = Path(questionnaire_path)

    items = _load_questionnaire_trait_mcq_items(questionnaire_path)
    if not items:
        raise ValueError(f"No trait_mcq items found in {questionnaire_path}.")

    # Preserve questionnaire JSON ordering for item columns so the output has
    # a reproducible column order independent of raw_responses traversal.
    with open(questionnaire_path, "r", encoding="utf-8") as f:
        q_data = json.load(f)
    ordered_ids: list[str] = [
        str(it["id"]) for it in q_data.get("block_4_trait_mcq", {}).get("items", [])
    ]
    item_dims = [items[iid]["primary_dimension"] for iid in ordered_ids]
    item_idx = {iid: i for i, iid in enumerate(ordered_ids)}

    present_traits = list(dict.fromkeys(item_dims))  # de-dup, preserve order
    if trait_order is None:
        ocean = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        trait_order = [t for t in ocean if t in present_traits]
        trait_order += sorted(set(present_traits) - set(trait_order))

    # Two-pass: collect unique k values, then fill matrix.
    per_cell: dict[tuple[int, str], float] = {}
    with open(raw_responses_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("item_type") != "trait_mcq":
                continue
            iid = str(entry["item_id"])
            if iid not in items:
                continue
            mapping = items[iid]["answer_mapping"]

            probs = entry.get("probs")
            value: float | None = None
            if isinstance(probs, dict) and probs:
                v = 0.0
                for letter, p in probs.items():
                    if str(letter) in mapping:
                        v += float(p) * float(mapping[str(letter)])
                value = v
            else:
                choice = entry.get("parsed_choice")
                if choice is not None and str(choice) in mapping:
                    value = float(mapping[str(choice)])

            if value is None:
                continue
            per_cell[(int(entry["k"]), iid)] = value

    if not per_cell:
        raise ValueError(f"No parseable trait_mcq responses in {raw_responses_path}.")

    k_values = sorted({k for k, _ in per_cell})
    k_idx = {k: i for i, k in enumerate(k_values)}

    matrix = np.full((len(k_values), len(ordered_ids)), np.nan, dtype=np.float64)
    for (k, iid), v in per_cell.items():
        matrix[k_idx[k], item_idx[iid]] = v

    return TraitOrientedMatrix(
        matrix=matrix,
        k_index=k_values,
        item_ids=ordered_ids,
        item_dims=item_dims,
        trait_order=trait_order,
    )


@dataclass
class FactorTraitAlignment:
    """Alignment between FA factors and known traits.

    Attributes:
        top_k_count: [n_factors × n_traits] int — count of top-K loading items
            that belong to each trait. Unsigned (ranked by |loading|).
        top_k_count_pos: [n_factors × n_traits] int — count of top-K items
            with positive loading that belong to each trait.
        top_k_count_neg: [n_factors × n_traits] int — analogous for negative.
        mean_abs_loading: [n_factors × n_traits] — mean |loading| over all
            items in that trait block, per factor.
        mean_signed_loading: [n_factors × n_traits] — mean signed loading over
            all items in that trait block, per factor.
        trait_order: Canonical column order.
        factor_labels: Row labels (e.g. "F1", "F2", ...).
        top_k: The K used for top-K analysis.
    """

    top_k_count: np.ndarray
    top_k_count_pos: np.ndarray
    top_k_count_neg: np.ndarray
    mean_abs_loading: np.ndarray
    mean_signed_loading: np.ndarray
    trait_order: list[str]
    factor_labels: list[str]
    top_k: int


def compute_factor_trait_alignment(
    loadings: np.ndarray,
    item_dims: list[str],
    *,
    trait_order: list[str] | None = None,
    top_k: int = 20,
) -> FactorTraitAlignment:
    """Summarise factor-trait alignment from a trait-oriented FA loadings matrix.

    The matrix must have been produced on a trait-oriented response matrix —
    otherwise ``mean_signed_loading`` is not trait-interpretable (see
    ``build_trait_oriented_matrix``).

    Args:
        loadings: [n_items × n_factors] float.
        item_dims: ``primary_dimension`` for each row of ``loadings`` (same
            order as columns of the matrix that produced the FA).
        trait_order: Optional canonical trait ordering. Defaults to the order
            of first appearance in ``item_dims``.
        top_k: K for top-K-by-|loading| analysis.

    Returns:
        ``FactorTraitAlignment``.
    """
    loadings = np.asarray(loadings)
    n_items, n_factors = loadings.shape
    if len(item_dims) != n_items:
        raise ValueError(
            f"item_dims length {len(item_dims)} != loadings rows {n_items}"
        )

    if trait_order is None:
        trait_order = list(dict.fromkeys(item_dims))
    trait_to_col = {t: j for j, t in enumerate(trait_order)}
    n_traits = len(trait_order)

    dims_arr = np.array(item_dims)
    top_k_count = np.zeros((n_factors, n_traits), dtype=int)
    top_k_count_pos = np.zeros((n_factors, n_traits), dtype=int)
    top_k_count_neg = np.zeros((n_factors, n_traits), dtype=int)
    mean_abs_loading = np.zeros((n_factors, n_traits))
    mean_signed_loading = np.zeros((n_factors, n_traits))

    effective_k = min(top_k, n_items)
    for f in range(n_factors):
        col = loadings[:, f]
        order = np.argsort(-np.abs(col))[:effective_k]
        for i in order:
            t = dims_arr[i]
            if t not in trait_to_col:
                continue
            j = trait_to_col[t]
            top_k_count[f, j] += 1
            if col[i] >= 0:
                top_k_count_pos[f, j] += 1
            else:
                top_k_count_neg[f, j] += 1

        for t, j in trait_to_col.items():
            mask = dims_arr == t
            if mask.any():
                mean_abs_loading[f, j] = float(np.mean(np.abs(col[mask])))
                mean_signed_loading[f, j] = float(np.mean(col[mask]))

    factor_labels = [f"F{f+1}" for f in range(n_factors)]

    return FactorTraitAlignment(
        top_k_count=top_k_count,
        top_k_count_pos=top_k_count_pos,
        top_k_count_neg=top_k_count_neg,
        mean_abs_loading=mean_abs_loading,
        mean_signed_loading=mean_signed_loading,
        trait_order=trait_order,
        factor_labels=factor_labels,
        top_k=effective_k,
    )


def save_alignment(
    alignment: FactorTraitAlignment,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Save alignment tables as CSVs and a summary JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def _df(arr: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(arr, index=alignment.factor_labels, columns=alignment.trait_order)

    for name, arr in [
        ("top_k_count", alignment.top_k_count),
        ("top_k_count_pos", alignment.top_k_count_pos),
        ("top_k_count_neg", alignment.top_k_count_neg),
        ("mean_abs_loading", alignment.mean_abs_loading),
        ("mean_signed_loading", alignment.mean_signed_loading),
    ]:
        p = output_dir / f"alignment_{name}.csv"
        _df(arr).to_csv(p)
        written[name] = p

    # Per-factor "winner" trait by top-K count.
    winners = {}
    for f, label in enumerate(alignment.factor_labels):
        counts = alignment.top_k_count[f]
        best_j = int(np.argmax(counts))
        winners[label] = {
            "trait": alignment.trait_order[best_j],
            "count": int(counts[best_j]),
            "share": float(counts[best_j] / max(alignment.top_k, 1)),
        }
    summary = {
        "top_k": alignment.top_k,
        "trait_order": alignment.trait_order,
        "factor_labels": alignment.factor_labels,
        "factor_winners": winners,
    }
    summary_path = output_dir / "alignment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    written["summary"] = summary_path
    return written


def plot_alignment_bars(
    alignment: FactorTraitAlignment,
    save_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """Grouped bar chart: per-factor counts of top-K items per trait.

    One subplot per factor; bars show top-K composition across traits.
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_factors = len(alignment.factor_labels)
    ncols = min(4, n_factors)
    nrows = int(np.ceil(n_factors / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), squeeze=False)
    x = np.arange(len(alignment.trait_order))
    for ax, f in zip(axes.flat, range(n_factors)):
        counts = alignment.top_k_count[f]
        bars = ax.bar(x, counts, color="#4c78a8", edgecolor="white")
        best = int(np.argmax(counts))
        bars[best].set_color("#e45756")
        ax.set_title(
            f"{alignment.factor_labels[f]}  "
            f"(top-{alignment.top_k}, winner={alignment.trait_order[best]} "
            f"{counts[best]}/{alignment.top_k})"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [t[:4] for t in alignment.trait_order], rotation=0
        )
        ax.set_ylim(0, alignment.top_k)
        ax.set_ylabel("top-K count")
        for j, c in enumerate(counts):
            if c > 0:
                ax.text(j, c + 0.2, str(int(c)), ha="center", va="bottom", fontsize=8)
    for ax in axes.flat[n_factors:]:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_alignment_heatmap(
    alignment: FactorTraitAlignment,
    save_path: Path | str,
    *,
    kind: str = "mean_abs",
    title: str | None = None,
) -> Path:
    """Heatmap over (factor × trait).

    Args:
        kind: ``"mean_abs"`` (|loading|, symmetric colormap 0..vmax) or
            ``"mean_signed"`` (signed, symmetric around 0) or
            ``"top_k_count"`` (integer counts).
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if kind == "mean_abs":
        m = alignment.mean_abs_loading
        cmap, vmin, vmax = "viridis", 0.0, float(np.max(np.abs(m))) or 1e-9
        cbar_label = "mean |loading|"
        fmt = "{:.2f}"
    elif kind == "mean_signed":
        m = alignment.mean_signed_loading
        vmax = float(np.max(np.abs(m))) or 1e-9
        cmap, vmin = "RdBu_r", -vmax
        cbar_label = "mean signed loading"
        fmt = "{:+.2f}"
    elif kind == "top_k_count":
        m = alignment.top_k_count
        cmap, vmin, vmax = "viridis", 0.0, float(alignment.top_k)
        cbar_label = f"top-{alignment.top_k} count"
        fmt = "{:d}"
    else:
        raise ValueError(f"Unknown kind: {kind!r}")

    fig, ax = plt.subplots(
        figsize=(0.7 * len(alignment.trait_order) + 2.5,
                 0.45 * len(alignment.factor_labels) + 1.4)
    )
    im = ax.imshow(m, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(alignment.trait_order)))
    ax.set_xticklabels(alignment.trait_order, rotation=30, ha="right")
    ax.set_yticks(range(len(alignment.factor_labels)))
    ax.set_yticklabels(alignment.factor_labels)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            text = fmt.format(int(v) if kind == "top_k_count" else float(v))
            if kind == "mean_signed":
                color = "white" if abs(v) > 0.5 * vmax else "black"
            else:
                color = "white" if v > 0.5 * vmax else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label(cbar_label)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_all_alignment(
    alignment: FactorTraitAlignment,
    output_dir: Path | str,
    *,
    title_prefix: str | None = None,
) -> dict[str, Path]:
    """Write bar chart + three heatmaps (|loading|, signed loading, top-K count)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{title_prefix} — " if title_prefix else ""
    out: dict[str, Path] = {}
    out["bars"] = plot_alignment_bars(
        alignment, output_dir / "alignment_top_k_bars.pdf",
        title=f"{prefix}Top-{alignment.top_k} loading items per factor (by trait)",
    )
    out["heatmap_abs"] = plot_alignment_heatmap(
        alignment, output_dir / "alignment_mean_abs_heatmap.pdf",
        kind="mean_abs",
        title=f"{prefix}Factor × trait: mean |loading|",
    )
    out["heatmap_signed"] = plot_alignment_heatmap(
        alignment, output_dir / "alignment_mean_signed_heatmap.pdf",
        kind="mean_signed",
        title=f"{prefix}Factor × trait: mean signed loading (trait-oriented)",
    )
    out["heatmap_count"] = plot_alignment_heatmap(
        alignment, output_dir / "alignment_top_k_count_heatmap.pdf",
        kind="top_k_count",
        title=f"{prefix}Factor × trait: top-{alignment.top_k} count",
    )
    return out
