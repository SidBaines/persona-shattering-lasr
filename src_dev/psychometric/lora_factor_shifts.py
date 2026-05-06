"""Per-LoRA factor-score-shift summaries for the paper.

Reads ``<label>_summary.json`` produced by
``scripts_dev/oct_pipeline/unsup_k4_v7_pf3/validate_lora.py`` (one per
trained LoRA × validation run), and lays out a wide cross-LoRA table /
heatmap of per-factor mean shifts.

The summaries live on the shared HuggingFace monorepo
(``persona-shattering-lasr/monorepo``) under the path layout

    fine_tuning/llama-3.1-8b-it/unsupervised/<factor>/<direction>/
        v<train_recipe>/evals/factor_validate/<label>/
            <label>_summary.json    # aggregate per-factor mean_diff + CI
            <label>_scores.npz      # per-persona baseline + LoRA factor scores
            <label>_paired_diff.png

This module pulls the summary + scores files into a local mirror dir on
demand (single-file ``hf_hub_download`` calls — small, fast) and
provides utilities to fold them into a wide DataFrame-like dict and a
heatmap-style figure.

Heavier sibling files (``questionnaire_v7_fc_pair/`` raw response
matrices) are intentionally not hydrated here; pull them separately
when you need to do your own analysis on the per-item data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


__all__ = [
    "LoraValidation",
    "hydrate_validation_artifacts",
    "load_lora_factor_shifts",
    "compute_bucketed_shifts",
    "build_shift_matrix",
    "plot_factor_shift_heatmap",
    "plot_factor_shift_barchart",
]


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoraValidation:
    """One LoRA validation entry.

    Attributes:
        label: Short label used in the validate_lora.py run (e.g.
            ``initiative_amp``). Used as the row name in the heatmap.
        factor: Canonical factor name the LoRA targets (one of
            ``Initiative``, ``Warmth``, ``Pedagogy``, ``Hedging``).
            Used to mark the ``intended-effect'' cell on the heatmap.
        direction: ``"+"`` for amplifier, ``"-"`` for suppressor.
        hf_subdir: Repo path under ``persona-shattering-lasr/monorepo``
            holding the run dir (the dir containing
            ``<label>_summary.json``, ``<label>_scores.npz``, etc.).
        prefer_large_n: When True, if a sibling label like
            ``<label>_prefix1000`` exists on the same parent, use the
            larger-n variant instead. Lets the analysis auto-upgrade as
            larger runs land without an edit.
    """
    label: str
    factor: str
    direction: str
    hf_subdir: str
    prefer_large_n: bool = True


def hydrate_validation_artifacts(
    *,
    hf_repo_id: str,
    hf_subdir: str,
    label: str,
    local_root: Path,
    pull_scores: bool = True,
) -> dict[str, Path | None]:
    """Download ``<label>_summary.json`` (and optionally ``_scores.npz``)
    from HF into ``local_root / hf_subdir / ...`` if not already present.

    Returns a dict of fetched paths keyed by artifact name. Missing
    optional artifacts (e.g. scores.npz absent on HF for a label) come
    back as ``None``.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    out: dict[str, Path | None] = {"summary": None, "scores": None}

    summary_in_repo = f"{hf_subdir.rstrip('/')}/{label}_summary.json"
    summary_local = local_root / summary_in_repo
    if not summary_local.exists():
        log.info("[lora-shifts] hydrate %s", summary_in_repo)
        path = hf_hub_download(
            repo_id=hf_repo_id, repo_type="dataset",
            filename=summary_in_repo, local_dir=local_root,
        )
        summary_local = Path(path)
    out["summary"] = summary_local

    if pull_scores:
        scores_in_repo = f"{hf_subdir.rstrip('/')}/{label}_scores.npz"
        scores_local = local_root / scores_in_repo
        if not scores_local.exists():
            try:
                log.info("[lora-shifts] hydrate %s", scores_in_repo)
                path = hf_hub_download(
                    repo_id=hf_repo_id, repo_type="dataset",
                    filename=scores_in_repo, local_dir=local_root,
                )
                scores_local = Path(path)
            except EntryNotFoundError:
                scores_local = None
        out["scores"] = scores_local

    return out


def _resolve_label_with_large_n(
    *,
    hf_repo_id: str,
    base_subdir: str,
    base_label: str,
) -> tuple[str, str]:
    """If ``<base_subdir>/../<base_label>_prefix1000/`` exists on HF,
    return the (label, hf_subdir) pointing at the larger run; otherwise
    return the inputs unchanged.

    The validation-runs convention is ``<label>_prefix<N>`` for an N-
    persona variant of the same LoRA (typical: prefix1000 alongside the
    default 200-persona run).
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError, EntryNotFoundError

    parent = str(Path(base_subdir).parent)
    candidate_label = f"{base_label}_prefix1000"
    candidate_subdir = f"{parent}/{candidate_label}"
    candidate_summary = f"{candidate_subdir}/{candidate_label}_summary.json"
    api = HfApi()
    try:
        api.hf_hub_download(  # exists check via head
            repo_id=hf_repo_id, repo_type="dataset",
            filename=candidate_summary,
        ) if False else None
        # Use list_repo_tree on the parent — safer than hf_hub_download for an
        # existence test.
        for entry in api.list_repo_tree(
            repo_id=hf_repo_id, repo_type="dataset",
            path_in_repo=parent, recursive=False,
        ):
            if entry.path.rstrip("/").endswith("/" + candidate_label):
                return candidate_label, candidate_subdir
    except (RepositoryNotFoundError, EntryNotFoundError, Exception):
        pass
    return base_label, base_subdir


def load_lora_factor_shifts(
    validations: list[LoraValidation],
    *,
    hf_repo_id: str,
    local_root: Path,
    pull_scores: bool = True,
    canonical_factor_order: list[str] | None = None,
) -> dict:
    """Hydrate every entry's summary (+ scores.npz when present), build a
    matrix of per-LoRA per-factor mean shifts.

    Returns:
        dict with keys:
            ``rows`` — list of ``{label, factor, direction, n_personas,
                                  target_factor, summary_path,
                                  scores_path}``.
            ``factors`` — column factor name order.
            ``mean_diff`` — np.ndarray (n_lora, n_factor)
            ``ci_lo`` / ``ci_hi`` — same shape, 95% CI bounds
            ``cohen_dz`` — same shape
    """
    rows: list[dict] = []
    matrix_diff: list[list[float]] = []
    matrix_lo: list[list[float]] = []
    matrix_hi: list[list[float]] = []
    matrix_dz: list[list[float]] = []
    factors_seen: list[str] | None = canonical_factor_order

    for v in validations:
        label = v.label
        subdir = v.hf_subdir
        if v.prefer_large_n:
            try:
                label, subdir = _resolve_label_with_large_n(
                    hf_repo_id=hf_repo_id,
                    base_subdir=v.hf_subdir,
                    base_label=v.label,
                )
            except Exception as exc:
                log.warning("[lora-shifts] prefer_large_n probe failed for %s: %s",
                            v.label, exc)

        artifacts = hydrate_validation_artifacts(
            hf_repo_id=hf_repo_id,
            hf_subdir=subdir,
            label=label,
            local_root=local_root,
            pull_scores=pull_scores,
        )
        summary_path = artifacts["summary"]
        if summary_path is None or not Path(summary_path).exists():
            log.warning("[lora-shifts] skipping %s: no summary found", label)
            continue

        payload = json.loads(Path(summary_path).read_text())
        per_factor = payload.get("factor_summary") or []
        if not per_factor:
            log.warning("[lora-shifts] empty factor_summary in %s", summary_path)
            continue

        # Match by short factor name (strip F<digit>_ prefix). The
        # validate_lora.py summaries label factors with their pre-sort
        # F-index (e.g. "F1_Pedagogy") while the rest of the v2 paper
        # pipeline uses post-variance-sort indices (e.g. "F1_Warmth").
        # The factor identity (Initiative / Warmth / Pedagogy / Hedging)
        # is the same across both; only the index disagrees, so we key
        # on the short name and ignore F#.
        def _short_name(name: str) -> str:
            n = name.strip()
            if "_" in n and n[:1] == "F" and n[1:2].isdigit():
                return n.split("_", 1)[1]
            return n
        by_short = {_short_name(r["factor"]): r for r in per_factor}
        if factors_seen is None:
            # Default order if caller didn't specify: short-names from
            # this first summary's order.
            factors_seen = [_short_name(r["factor"]) for r in per_factor]
        # Reorder this run's per-factor entries to match factors_seen.
        per_factor = [by_short[f] for f in factors_seen if f in by_short]
        if len(per_factor) != len(factors_seen):
            missing = [f for f in factors_seen if f not in by_short]
            log.warning(
                "[lora-shifts] %s missing factors %s — skipping", label, missing,
            )
            continue

        # Bucketed (low/medium/high) per-factor shifts computed off the
        # paired scores npz when available. Used by build_shift_matrix to
        # construct the middling-only and headroom-conditioned heatmaps.
        bucketed: dict[str, dict] = {}
        scores_path = artifacts.get("scores")
        if scores_path is not None and Path(scores_path).exists():
            try:
                bucketed = compute_bucketed_shifts(
                    scores_path=scores_path,
                    summary_path=summary_path,
                    canonical_factor_order=factors_seen,
                )
            except Exception as exc:
                log.warning(
                    "[lora-shifts] %s bucketed shifts failed: %s", label, exc,
                )
                bucketed = {}

        rows.append({
            "label": label,
            "factor": v.factor,
            "direction": v.direction,
            "n_personas": int(payload.get("n_personas", 0)),
            "target_factor": payload.get("target_factor"),
            "summary_path": str(summary_path),
            "scores_path": str(artifacts["scores"]) if artifacts.get("scores") else None,
            "adapter": payload.get("adapter"),
            "bucketed": bucketed,
        })
        matrix_diff.append([float(r["mean_diff"]) for r in per_factor])
        matrix_lo.append([float(r["ci_95_lo"]) for r in per_factor])
        matrix_hi.append([float(r["ci_95_hi"]) for r in per_factor])
        matrix_dz.append([float(r.get("cohen_dz", float("nan"))) for r in per_factor])

    return {
        "rows": rows,
        "factors": factors_seen or [],
        "mean_diff": np.asarray(matrix_diff, dtype=float),
        "ci_lo": np.asarray(matrix_lo, dtype=float),
        "ci_hi": np.asarray(matrix_hi, dtype=float),
        "cohen_dz": np.asarray(matrix_dz, dtype=float),
    }


def _short_factor_name(name: str) -> str:
    n = name.strip()
    if "_" in n and n[:1] == "F" and n[1:2].isdigit():
        return n.split("_", 1)[1]
    return n


def compute_bucketed_shifts(
    *,
    scores_path: Path,
    summary_path: Path,
    canonical_factor_order: list[str],
    seed: int = 436,
    n_bootstrap: int = 2000,
) -> dict[str, dict]:
    """Load ``<label>_scores.npz`` and bucket personas by their *baseline*
    score on each factor (low / medium / high tertile), reporting mean
    paired shift in each bucket.

    Returns ``{short_factor_name -> {bucket_name -> {mean, ci_lo, ci_hi, n,
    baseline_min, baseline_max, baseline_mean}}}`` reordered to
    ``canonical_factor_order``. Bucketing uses the same factor's baseline
    column to bucket — i.e. for each (column factor) we bucket on that
    factor's baseline column and then read off the shift on that same
    factor's column. This is the "did personas with room to move on this
    factor actually move?" reading.
    """
    npz = np.load(Path(scores_path), allow_pickle=False)
    baseline = np.asarray(npz["baseline_scores"], dtype=float)
    lora = np.asarray(npz["lora_scores"], dtype=float)
    summary = json.loads(Path(summary_path).read_text())
    factor_names_npz = [r["factor"] for r in summary.get("factor_summary", [])]
    short_to_col = {_short_factor_name(n): i for i, n in enumerate(factor_names_npz)}

    out: dict[str, dict] = {}
    for f_short in canonical_factor_order:
        if f_short not in short_to_col:
            continue
        col = short_to_col[f_short]
        b = baseline[:, col]
        l = lora[:, col]
        d = l - b
        q1, q2 = np.quantile(b, [1.0 / 3.0, 2.0 / 3.0])
        masks = {
            "low":    b <= q1,
            "medium": (b > q1) & (b <= q2),
            "high":   b > q2,
        }
        per_bucket: dict[str, dict] = {}
        for bi, (name, m) in enumerate(masks.items()):
            d_m = d[m]
            b_m = b[m]
            if len(d_m) == 0:
                per_bucket[name] = {
                    "mean": float("nan"), "ci_lo": float("nan"),
                    "ci_hi": float("nan"), "n": 0,
                    "baseline_min": float("nan"),
                    "baseline_max": float("nan"),
                    "baseline_mean": float("nan"),
                }
                continue
            rng = np.random.default_rng(seed + bi)
            draws = rng.integers(0, len(d_m), size=(n_bootstrap, len(d_m)))
            boots = d_m[draws].mean(axis=1)
            ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
            per_bucket[name] = {
                "mean": float(d_m.mean()),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "n": int(len(d_m)),
                "baseline_min": float(b_m.min()),
                "baseline_max": float(b_m.max()),
                "baseline_mean": float(b_m.mean()),
            }
        out[f_short] = per_bucket
    return out


def build_shift_matrix(
    shifts: dict,
    *,
    selection: str,
) -> dict:
    """Build a (n_lora, n_factor) shift matrix using one of several bucket
    selection rules.

    ``selection`` choices:
        ``"naive"`` — full-sample mean shift (already in ``shifts['mean_diff']``).
        ``"middling"`` — mean shift over personas in the *medium* baseline
            tertile of the column factor (option A).
        ``"headroom"`` — mean shift over personas in the headroom tertile
            for the LoRA's intended direction: low-baseline tertile for
            amplifiers (``direction == "+"``), high-baseline tertile for
            suppressors (``direction == "-"``) (option B). Same rule
            applied to off-target columns: i.e. for an amplifier, every
            cell reports the upward-headroom-tertile shift on the column
            factor.

    Returns a dict with ``mean_diff``, ``ci_lo``, ``ci_hi`` (all
    ``np.ndarray`` of shape ``(n_lora, n_factor)``) and ``n_per_cell``.
    """
    rows = shifts["rows"]
    factors = shifts["factors"]
    n_lora = len(rows)
    n_factor = len(factors)

    if selection == "naive":
        return {
            "mean_diff": shifts["mean_diff"],
            "ci_lo": shifts["ci_lo"],
            "ci_hi": shifts["ci_hi"],
            "n_per_cell": np.full((n_lora, n_factor),
                                  rows[0]["n_personas"] if rows else 0, dtype=int),
            "selection": selection,
        }

    diff = np.full((n_lora, n_factor), np.nan, dtype=float)
    lo = np.full_like(diff, np.nan)
    hi = np.full_like(diff, np.nan)
    n = np.zeros((n_lora, n_factor), dtype=int)

    for i, r in enumerate(rows):
        bucketed = r.get("bucketed") or {}
        direction = r.get("direction", "+")
        if selection == "middling":
            bucket_name = "medium"
        elif selection == "headroom":
            bucket_name = "low" if direction == "+" else "high"
        else:
            raise ValueError(f"unknown selection: {selection!r}")
        for j, f in enumerate(factors):
            cell = bucketed.get(f, {}).get(bucket_name)
            if cell is None:
                continue
            diff[i, j] = cell["mean"]
            lo[i, j] = cell["ci_lo"]
            hi[i, j] = cell["ci_hi"]
            n[i, j] = cell["n"]
    return {
        "mean_diff": diff,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_per_cell": n,
        "selection": selection,
    }


def plot_factor_shift_heatmap(
    shifts: dict,
    *,
    save_path: Path,
    title: str = "Per-LoRA factor-score shift",
    factor_display_names: list[str] | None = None,
    annotate: str = "diff_with_ci",
    matrix_override: dict | None = None,
    row_label_factor_map: dict[str, str] | None = None,
) -> None:
    """Heatmap: rows = LoRAs, cols = factors, cells coloured by mean_diff.

    ``annotate``:
        ``"diff"``         - just the mean diff in the cell.
        ``"diff_with_ci"`` - mean diff with the half-CI as ± below.
        ``"dz"``           - Cohen's d_z.

    The intended-effect cell (LoRA factor matches column) is outlined
    in white so the reader can scan diagonal-vs-off-diagonal at a glance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if matrix_override is not None:
        diff = matrix_override["mean_diff"]
        lo = matrix_override["ci_lo"]
        hi = matrix_override["ci_hi"]
        n_per_cell = matrix_override.get("n_per_cell")
    else:
        diff = shifts["mean_diff"]
        lo = shifts["ci_lo"]
        hi = shifts["ci_hi"]
        n_per_cell = None
    dz = shifts["cohen_dz"]
    rows = shifts["rows"]
    factors = shifts["factors"]
    n_lora, n_factor = diff.shape
    if factor_display_names is None:
        factor_display_names = factors

    # Row labels e.g. "Initiative ↑". When a paper-display rename is in
    # effect, ``row_label_factor_map`` translates the internal short name
    # (e.g. "Warmth") to the display name (e.g. "Tone") before formatting.
    arrow = {"+": "↑", "-": "↓"}
    row_labels: list[str] = []
    for r in rows:
        a = arrow.get(r["direction"], r["direction"])
        f_display = (row_label_factor_map or {}).get(r["factor"], r["factor"])
        row_labels.append(f"{f_display} {a}")

    abs_diff = np.abs(diff)
    finite = abs_diff[np.isfinite(abs_diff)]
    vmax = float(finite.max()) if finite.size else 1.0
    if vmax == 0:
        vmax = 1.0
    fig, ax = plt.subplots(
        figsize=(1.2 * n_factor + 4, 0.55 * n_lora + 2),
    )
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    # Annotate cells.
    for i in range(n_lora):
        for j in range(n_factor):
            v = diff[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="grey")
                continue
            if annotate == "diff":
                txt = f"{v:+.2f}"
            elif annotate == "dz":
                txt = f"{dz[i, j]:+.2f}"
            else:
                half_ci = (hi[i, j] - lo[i, j]) / 2.0
                txt = f"{v:+.2f}\n±{half_ci:.2f}"
            color = "white" if abs(v) > 0.55 * vmax else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    # Outline target cells: LoRA's factor matches column factor.
    factor_to_col = {f: j for j, f in enumerate(factors)}
    # Match by stripping "F<n>_" prefix when present (e.g. "F0_Initiative" -> "Initiative").
    for i, r in enumerate(rows):
        for j, f in enumerate(factors):
            f_short = f.split("_", 1)[1] if "_" in f and f[1:2].isdigit() else f
            if r["factor"].lower() == f_short.lower():
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="white", linewidth=2.2,
                ))
                # Side note: target direction (+/-) — useful interpretation.
                # We don't recolor; the outline is enough.

    ax.set_xticks(range(n_factor))
    ax.set_xticklabels(factor_display_names, rotation=15, ha="right")
    ax.set_yticks(range(n_lora))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Behavioural Factor")
    ax.set_ylabel("LoRA target (behaviour + direction)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                 label=r"$\bar{F}^{\text{LoRA}} - \bar{F}^{\text{baseline}}$ (factor-score units)")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    fig.savefig(save_path.with_suffix(".pdf"))
    plt.close(fig)


def plot_factor_shift_barchart(
    shifts: dict,
    *,
    save_path: Path,
    title: str = "Per-LoRA factor-score shift",
    factor_display_names: list[str] | None = None,
    label_filter: list[str] | None = None,
    matrix_override: dict | None = None,
    direction_palette: dict[str, str] | None = None,
    row_label_factor_map: dict[str, str] | None = None,
) -> None:
    """Grouped bar chart: x = factors, bars within group = LoRAs.

    A focused alternative to ``plot_factor_shift_heatmap`` for the case
    where a small number of LoRAs (e.g. one factor's amp + sup pair)
    are being compared across the full set of factors. Each bar shows
    $\\Delta$ in factor-score units; error bars span the 95% bootstrap
    CI from ``matrix_override`` (or ``shifts``'s naive fields if None).

    Args:
        shifts: Output of ``load_lora_factor_shifts``.
        save_path: Output PNG path. A sibling PDF is also written.
        title: Plot title.
        factor_display_names: Override for x-axis tick labels; defaults
            to ``shifts['factors']``.
        label_filter: If set, restrict the LoRAs shown to those whose
            ``label`` is in this list (preserving order).
        matrix_override: As in ``plot_factor_shift_heatmap`` — if set,
            use these values (e.g. for the medium-tertile or headroom
            views) instead of the naive full-sample mean.
        direction_palette: Optional mapping ``{"+": colour, "-": colour}``
            so amp / sup get distinct fills. Defaults to a blue / orange
            pair.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if matrix_override is not None:
        diff = matrix_override["mean_diff"]
        lo = matrix_override["ci_lo"]
        hi = matrix_override["ci_hi"]
    else:
        diff = shifts["mean_diff"]
        lo = shifts["ci_lo"]
        hi = shifts["ci_hi"]
    rows = shifts["rows"]
    factors = shifts["factors"]
    if factor_display_names is None:
        factor_display_names = factors

    if label_filter is not None:
        wanted = list(label_filter)
        keep = []
        for w in wanted:
            for i, r in enumerate(rows):
                if r["label"] == w:
                    keep.append(i)
                    break
        if not keep:
            raise ValueError(f"None of label_filter={wanted} matched any loaded LoRA")
        rows = [rows[i] for i in keep]
        diff = diff[keep, :]
        lo = lo[keep, :]
        hi = hi[keep, :]

    n_lora, n_factor = diff.shape
    if direction_palette is None:
        direction_palette = {"+": "#2563eb", "-": "#ea580c"}

    arrow = {"+": "↑", "-": "↓"}
    bar_labels = [
        f"{(row_label_factor_map or {}).get(r['factor'], r['factor'])} "
        f"{arrow.get(r['direction'], r['direction'])}"
        for r in rows
    ]

    width = 0.8 / max(n_lora, 1)
    x = np.arange(n_factor, dtype=float)

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * n_factor + 2), 4.4))
    for i, r in enumerate(rows):
        offset = (i - (n_lora - 1) / 2.0) * width
        vals = diff[i]
        # Error bar ranges relative to the bar's mean (matplotlib expects
        # nonnegative error magnitudes).
        err_lo = np.maximum(vals - lo[i], 0.0)
        err_hi = np.maximum(hi[i] - vals, 0.0)
        ax.bar(
            x + offset, vals, width,
            color=direction_palette.get(r["direction"], "#6b7280"),
            edgecolor="#111", linewidth=0.4,
            label=bar_labels[i],
            yerr=np.vstack([err_lo, err_hi]),
            capsize=3, error_kw={"elinewidth": 0.9, "ecolor": "#374151"},
        )
        # Numeric annotation above (or below for negative) each bar.
        for xi, v in zip(x + offset, vals):
            if not np.isfinite(v):
                continue
            va = "bottom" if v >= 0 else "top"
            pad = 0.02 * max(np.nanmax(np.abs(diff)) or 1.0, 1e-3)
            y_text = v + pad if v >= 0 else v - pad
            ax.text(xi, y_text, f"{v:+.2f}",
                    ha="center", va=va, fontsize=8, color="#111")

    ax.axhline(0, color="#111", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(factor_display_names, fontsize=10)
    ax.set_xlabel("Behavioural Factor")
    ax.set_ylabel(
        r"$\bar{F}^{\text{LoRA}} - \bar{F}^{\text{baseline}}$ (factor-score units)"
    )
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    fig.savefig(save_path.with_suffix(".pdf"))
    plt.close(fig)
