#!/usr/bin/env python3
"""Analyze and visualize personality evaluation results from a sweep run.

Produces plots from a single run directory, one per eval type found in
``_PLOT_REGISTRY``. Each eval maps to one of the built-in plot styles or a
custom PlotFn. Evals not present in the registry are skipped with a warning.

Built-in plot styles:

  "trait"      — trait_sweep.png
    • Big Five from TRAIT benchmark (absolute 0–1, full color)
    • Dark Triad (Mach, Narc, Psychopathy) dimmed + dashed
    • 95% CI bands when n_runs > 1

  "bfi"        — bfi_sweep.png
    • Big Five from BFI benchmark (delta from baseline, y=0 = unmodified model)
    • y-axis zoomed to data range
    • 95% CI bands when n_runs > 1

  "capability" — <eval_name>_sweep.png
    • Accuracy vs. scale with baseline reference and allowed-drop band
    • Suitable for any accuracy-like eval: mmlu, gsm8k, truthfulqa, arc, etc.
    • 95% CI bands when n_runs > 1

  "generic"    — <eval_name>_sweep.png
    • All numeric metric columns on a single 0–1 axis
    • Useful as a quick-look fallback; register explicitly to opt in

To add a new eval, insert one line in ``_PLOT_REGISTRY``:
    "my_eval": "capability"        # for accuracy-like metrics
    "my_eval": "generic"           # for a quick look at any numeric columns
    "my_eval": plot_my_sweep       # for a fully custom plot function

Expected run directory layout (produced by the personality eval suite):

    scratch/evals/personality/{run_name}/
      {model_spec_name}/
        {eval_name}/          # e.g. bfi/, trait/, mmlu/
          run_info.json
          native/inspect_logs/*.json

Usage:
    uv run python -m scripts.evals.personality.analyze_results \\
        scratch/evals/personality/my_run --visualize

    # Save plots to a specific directory
    uv run python -m scripts.evals.personality.analyze_results \\
        scratch/evals/personality/my_run \\
        --output-dir scratch/evals/personality/my_run/figures \\
        --visualize

    # Mock sweep data for offline testing
    uv run python -m scripts.evals.personality.analyze_results --mock
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIG_FIVE = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
DARK_TRIAD = ["Machiavellianism", "Narcissism", "Psychopathy"]
ALL_TRAIT_COLS = BIG_FIVE + DARK_TRIAD

_OCEAN_ALIASES: dict[str, str] = {
    "O": "Openness",
    "C": "Conscientiousness",
    "E": "Extraversion",
    "A": "Agreeableness",
    "N": "Neuroticism",
}

BIG_FIVE_COLORS = {
    "Openness":          "#2196F3",
    "Conscientiousness": "#FF9800",
    "Extraversion":      "#4CAF50",
    "Agreeableness":     "#9C27B0",
    "Neuroticism":       "#F44336",
}
DARK_TRIAD_COLORS = {
    "Machiavellianism": "#795548",
    "Narcissism":       "#E91E63",
    "Psychopathy":      "#607D8B",
}

# Linestyle cycle used when overlaying multiple runs on a single plot.
# Index 0 = primary run (solid); subsequent indices = comparison runs.
_LINESTYLES = ["-", "--", "-.", ":"]

# Eval names that use the fallback answer parser (rescore_log) for scoring.
_PERSONALITY_EVALS = {"bfi", "trait"}

# Directories that are not model-spec dirs at the run root.
_NON_MODEL_DIRS = {"figures", "analysis"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SweepData:
    """DataFrames for every eval found in a sweep run directory.

    Each DataFrame has columns: scale (float), run (str), + metric columns.
    Keyed by eval name (the directory name used in the suite config, e.g.
    "bfi", "trait", "mmlu", or any custom name).

    Access via ``data.get("bfi")`` which returns None if the eval is absent.
    """
    evals: dict[str, pd.DataFrame] = field(default_factory=dict)

    def get(self, name: str) -> pd.DataFrame | None:
        return self.evals.get(name)

    def names(self) -> list[str]:
        return list(self.evals.keys())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extract_scores(log_path: Path) -> tuple[dict[str, float], float] | None:
    """Extract metric values and parse rate from an inspect log JSON.

    Returns:
        Tuple of (scores dict, parse_rate 0–1), or None if the log failed.
    """
    with open(log_path) as f:
        log = json.load(f)
    if log.get("status") != "success":
        return None
    score_entry = log["results"]["scores"][0]
    scored   = score_entry.get("scored_samples", 0)
    unscored = score_entry.get("unscored_samples", 0)
    total = scored + unscored
    parse_rate = scored / total if total > 0 else 1.0
    metrics = score_entry["metrics"]
    scores = {k: v["value"] for k, v in metrics.items() if isinstance(v, dict) and "value" in v}
    return scores, parse_rate


def _extract_scores_reparsed(log_path: Path, eval_type: str) -> tuple[dict[str, float], float] | None:
    """Like _extract_scores but recomputes trait scores using the fallback parser."""
    from scripts.evals.personality.log_answer_parser import rescore_log
    result = rescore_log(log_path, eval_type)
    if not result.scores:
        return None
    return result.scores, result.parse_rate


def _load_from_info(
    info_path: Path,
    model: str,
    run: str,
    reparse: bool = False,
    eval_type: str = "bfi",
) -> dict | None:
    with open(info_path) as f:
        info = json.load(f)
    if info.get("status") != "ok":
        print(f"  skip {model}/{run}: {info.get('error')}", file=sys.stderr)
        return None
    log_path = info.get("native", {}).get("inspect_log_path")
    if not log_path:
        print(f"  skip {model}/{run}: no inspect_log_path", file=sys.stderr)
        return None
    if reparse:
        result = _extract_scores_reparsed(Path(log_path), eval_type)
    else:
        result = _extract_scores(Path(log_path))
    if result is None:
        print(f"  skip {model}/{run}: log not success", file=sys.stderr)
        return None
    scores, parse_rate = result
    scale = info.get("scale")  # float | None; None = base model
    return {"model": model, "run": run, "scale": scale, "_parse_rate": parse_rate, **scores}


def _parse_scale(model_name: str) -> float | None:
    """Fallback: parse scale from model name string (e.g. lora_+1p25x -> 1.25)."""
    if model_name == "base":
        return 0.0
    m = re.match(r"lora_([+-]?)(\d+)p(\d+)x", model_name)
    if m:
        sign = -1.0 if m.group(1) == "-" else 1.0
        return sign * (int(m.group(2)) + int(m.group(3)) / 100.0)
    m = re.match(r"lora_([+-]?\d+)x", model_name)
    if m:
        return float(m.group(1))
    return None


def _normalise_scale_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has a numeric 'scale' column. Fills from name parsing if needed."""
    df = df.copy()
    if "scale" in df.columns and df["scale"].notna().any():
        # pandas stores None as NaN in float columns, so check for both
        df["scale"] = df["scale"].apply(lambda s: 0.0 if (s is None or (isinstance(s, float) and np.isnan(s))) else s)
    else:
        df["scale"] = [_parse_scale(m) for m in df["model"]]
    return df


def load_sweep_data(run_dir: Path, reparse: bool = False) -> SweepData:
    """Load results for all evals found in a sweep run directory.

    Walks ``run_dir/<model_spec>/<eval_name>/run_info.json`` and loads every
    eval directory present — no hardcoded whitelist. Personality evals (those
    in ``_PERSONALITY_EVALS``) are rescored via the fallback answer parser
    when ``reparse=True``; all others use the Inspect scorer values directly.

    Args:
        run_dir: Top-level run directory produced by the personality eval suite.
        reparse: If True, recompute personality trait scores from raw model
            outputs using the fallback answer parser rather than inspect scorer.

    Returns:
        SweepData with one DataFrame per eval type found (keyed by eval name).
    """
    records: dict[str, list[dict]] = {}

    for model_dir in sorted(run_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name in _NON_MODEL_DIRS:
            continue
        model = model_dir.name

        for eval_dir in sorted(model_dir.iterdir()):
            if not eval_dir.is_dir():
                continue
            eval_name = eval_dir.name
            if eval_name not in records:
                records[eval_name] = []

            # n_runs support: look for run_info.json directly or inside run_NN subdirs
            info_paths: list[tuple[Path, str]] = []
            direct = eval_dir / "run_info.json"
            if direct.exists():
                info_paths.append((direct, "run_1"))
            else:
                for run_subdir in sorted(eval_dir.iterdir()):
                    rip = run_subdir / "run_info.json"
                    if run_subdir.is_dir() and rip.exists():
                        info_paths.append((rip, run_subdir.name))

            is_personality = eval_name in _PERSONALITY_EVALS
            for info_path, run_label in info_paths:
                rec = _load_from_info(
                    info_path, model, run_label,
                    reparse=(reparse and is_personality),
                    eval_type=eval_name if is_personality else "bfi",
                )
                if rec:
                    records[eval_name].append(rec)

    def _to_df(recs: list[dict]) -> pd.DataFrame | None:
        if not recs:
            return None
        df = pd.DataFrame(recs)
        df = _normalise_scale_col(df)
        return df[df["scale"].notna()].sort_values(["scale", "run"]).reset_index(drop=True)

    return SweepData(evals={name: df for name, recs in records.items()
                            if (df := _to_df(recs)) is not None})


def load_data_from_logs(
    log_dir: Path,
    eval_type: str,
    reparse: bool = True,
) -> pd.DataFrame:
    """Fallback loader for bare log directories without run_info.json.

    Discovers logs via ``**/<eval_type>/native/inspect_logs/*.json`` and
    infers model name and scale from the directory structure / model name.
    """
    from scripts.evals.personality.log_answer_parser import rescore_log

    pattern = f"**/{eval_type}/native/inspect_logs/*.json"
    records = []
    for log_path in sorted(log_dir.glob(pattern)):
        try:
            parts = log_path.parts
            eval_idx = max(i for i, p in enumerate(parts) if p == eval_type)
            model = parts[eval_idx - 1]
        except (ValueError, IndexError):
            model = "unknown"

        if reparse:
            result = rescore_log(log_path, eval_type)
            scores = result.scores
        else:
            scores = _extract_scores(log_path)
        if scores:
            records.append({"model": model, "run": "run_1", "scale": None, **scores})

    if not records:
        raise ValueError(f"No logs found under {log_dir} for eval_type={eval_type!r}")
    df = pd.DataFrame(records)
    return _normalise_scale_col(df)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _resolve_highlight(highlight: list[str] | None) -> set[str]:
    """Resolve highlight list (full names or OCEAN letters) to a set of full trait names.

    Returns the full Big Five set when highlight is None or empty (all lit by default).
    """
    if not highlight:
        return set(BIG_FIVE)
    resolved = set()
    for h in highlight:
        resolved.add(_OCEAN_ALIASES.get(h, h))
    return resolved


def _ci95(values: np.ndarray) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df=n - 1) * values.std(ddof=1) / np.sqrt(n))
    except Exception:
        return float(1.96 * values.std(ddof=1) / np.sqrt(n))


def _agg_sweep(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Aggregate a sweep DataFrame to mean ± CI per scale point.

    Returns a DataFrame with columns: scale, {col}_mean, {col}_ci for each col.
    """
    rows = []
    for scale, grp in df.groupby("scale"):
        row: dict = {"scale": scale}
        for col in cols:
            if col not in grp.columns:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_ci"] = 0.0
                continue
            vals = grp[col].dropna().values
            row[f"{col}_mean"] = vals.mean() if len(vals) else float("nan")
            row[f"{col}_ci"] = _ci95(vals)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("scale").reset_index(drop=True)


def _metric_cols(df: pd.DataFrame) -> list[str]:
    """Return metric columns from a sweep DataFrame (everything except housekeeping cols)."""
    return [c for c in df.columns if c not in ("model", "run", "scale", "_parse_rate")]


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------

def print_sweep_table(agg: pd.DataFrame, cols: list[str], title: str) -> None:
    width = 10 + 20 * len(cols)
    print(f"\n{'=' * width}")
    print(title)
    print("=" * width)
    header = f"{'Scale':<10}" + "".join(f"{c:<20}" for c in cols)
    print(header)
    print("-" * width)
    for _, row in agg.iterrows():
        marker = " ← baseline" if abs(row["scale"]) < 0.01 else ""
        vals = "".join(f"{row[f'{c}_mean']:.4f}±{row[f'{c}_ci']:.4f}{'':>6}" for c in cols)
        print(f"{row['scale']:+7.2f}   {vals}{marker}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

def _mock_sweep_data() -> SweepData:
    """Generate realistic mock SweepData for offline testing."""
    rng = np.random.default_rng(42)
    scales = [s for s in np.arange(-2.0, 2.25, 0.25) if round(s, 10) != 0.0]
    base_scales = [0.0] + list(scales)

    def _df(records: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        return _normalise_scale_col(df).sort_values(["scale", "run"]).reset_index(drop=True)

    # BFI: Big Five only, 3 runs per scale
    bfi_base  = {"Openness": 0.65, "Conscientiousness": 0.70, "Extraversion": 0.50,
                 "Agreeableness": 0.60, "Neuroticism": 0.45}
    bfi_slope = {"Openness": 0.08, "Conscientiousness": -0.03, "Extraversion": 0.05,
                 "Agreeableness": -0.12, "Neuroticism": 0.06}
    def _mock_parse_rate(s: float) -> float:
        """Simulate parse degradation at extreme scales (perfect in the middle)."""
        return float(np.clip(1.0 - 0.08 * max(0.0, abs(s) - 1.0) + rng.normal(0, 0.02), 0, 1))

    bfi_records = []
    for s in base_scales:
        for run in ["run_1", "run_2", "run_3"]:
            scores = {t: float(np.clip(bfi_base[t] + bfi_slope[t] * s + rng.normal(0, 0.025), 0, 1))
                      for t in BIG_FIVE}
            bfi_records.append({"model": "base" if s == 0.0 else f"lora_{s:+.2f}x",
                                 "run": run, "scale": s, "_parse_rate": _mock_parse_rate(s), **scores})

    # TRAIT: Big Five + Dark Triad, 3 runs per scale
    trait_base  = {**bfi_base, "Machiavellianism": 0.40, "Narcissism": 0.45, "Psychopathy": 0.35}
    trait_slope = {**bfi_slope, "Machiavellianism": 0.03, "Narcissism": 0.02, "Psychopathy": 0.01}
    trait_records = []
    for s in base_scales:
        for run in ["run_1", "run_2", "run_3"]:
            scores = {t: float(np.clip(trait_base[t] + trait_slope[t] * s + rng.normal(0, 0.025), 0, 1))
                      for t in ALL_TRAIT_COLS}
            trait_records.append({"model": "base" if s == 0.0 else f"lora_{s:+.2f}x",
                                   "run": run, "scale": s, "_parse_rate": _mock_parse_rate(s), **scores})

    # MMLU: coarser grid, 3 runs
    mmlu_scales = [s for s in np.arange(-2.0, 2.25, 0.5) if round(s, 10) != 0.0]
    mmlu_records = []
    for s in [0.0] + list(mmlu_scales):
        for run in ["run_1", "run_2", "run_3"]:
            acc = float(np.clip(0.62 - 0.04 * abs(s) + rng.normal(0, 0.01), 0, 1))
            mmlu_records.append({"model": "base" if s == 0.0 else f"lora_{s:+.2f}x",
                                  "run": run, "scale": s, "_parse_rate": _mock_parse_rate(s), "accuracy": acc})

    return SweepData(evals={
        "bfi":   _df(bfi_records),
        "trait": _df(trait_records),
        "mmlu":  _df(mmlu_records),
    })


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _setup_matplotlib() -> None:
    import matplotlib
    matplotlib.use("Agg")


def _draw_error_bars(ax, scales, means, cis, color) -> None:
    """Draw vertical error bars (mean ± CI) at each scale point. No-op if all CIs are zero."""
    cis_arr = np.array(cis)
    if not any(cis_arr > 0):
        return
    ax.errorbar(scales, means, yerr=cis_arr,
                fmt="none", color=color, capsize=3, capthick=1.0,
                elinewidth=1.0, alpha=0.7, zorder=5)


def plot_trait_sweep(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
    highlight: list[str] | None = None,
) -> Path:
    """Primary research plot: TRAIT Big Five + Dark Triad + human baselines.

    Args:
        df: DataFrame with columns: scale, run, Openness, ..., Psychopathy.
        output_dir: Directory to save the figure.
        title_suffix: Optional suffix appended to the figure title.
        highlight: Traits to render at full brightness. Accepts full names or OCEAN
            single letters (O/C/E/A/N). Unlisted Big Five traits are dimmed.
            Dark Triad is always dimmed regardless. Defaults to all Big Five.

    Returns:
        Path to the saved figure.
    """
    import matplotlib.pyplot as plt

    lit = _resolve_highlight(highlight)
    trait_agg = _agg_sweep(df, ALL_TRAIT_COLS)
    scales = trait_agg["scale"].values

    fig, ax = plt.subplots(figsize=(12, 5.5))

    # --- Big Five: lit at full color, dimmed if not in highlight ---
    for trait in BIG_FIVE:
        color = BIG_FIVE_COLORS[trait]
        means = trait_agg[f"{trait}_mean"].values
        cis   = trait_agg[f"{trait}_ci"].values
        if trait in lit:
            ax.plot(scales, means, "o-", color=color, linewidth=2.2, markersize=6,
                    label=trait, zorder=4)
            _draw_error_bars(ax, scales, means, cis, color)
        else:
            ax.plot(scales, means, "o-", color=color, linewidth=1.4, markersize=4,
                    alpha=0.35, label=trait, zorder=3)

    # --- Dark Triad: always dimmed dashed, no error bars ---
    for trait in DARK_TRIAD:
        color = DARK_TRIAD_COLORS[trait]
        means = trait_agg[f"{trait}_mean"].values
        ax.plot(scales, means, "--", color=color, linewidth=1.4, markersize=4,
                alpha=0.35, label=trait, zorder=3)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    ax.set_xlabel("LoRA scaling factor", fontsize=11)
    ax.set_ylabel("Trait score (0–1)", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_xticks(scales)
    ax.grid(True, alpha=0.25)

    title = "TRAIT sweep: personality scores vs. LoRA scale"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=13, fontweight="bold")

    ax.errorbar([], [], yerr=1, fmt="none", color="gray", capsize=3, capthick=1.0,
                elinewidth=1.0, alpha=0.7, label="95% CI")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
              fontsize=9, ncol=6, framealpha=0.85)

    plt.tight_layout()
    out = output_dir / "trait_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


def plot_bfi_sweep(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
    highlight: list[str] | None = None,
) -> Path:
    """Sanity-check plot: BFI Big Five centred at baseline (delta from scale=0).

    Args:
        df: DataFrame with columns: scale, run, Openness, ..., Neuroticism.
        output_dir: Directory to save the figure.
        title_suffix: Optional suffix appended to the figure title.
        highlight: Traits to render at full brightness. Accepts full names or OCEAN
            single letters (O/C/E/A/N). Unlisted traits are dimmed.
            Defaults to all Big Five.

    Returns:
        Path to the saved figure.
    """
    import matplotlib.pyplot as plt

    lit = _resolve_highlight(highlight)
    bfi_agg = _agg_sweep(df, BIG_FIVE)

    # Compute per-trait baseline (scale=0) mean for delta calculation.
    baseline_row = bfi_agg[bfi_agg["scale"].abs() < 1e-9]
    if baseline_row.empty:
        baseline_row = bfi_agg.loc[[bfi_agg["scale"].abs().idxmin()]]

    scales = bfi_agg["scale"].values
    fig, ax = plt.subplots(figsize=(12, 5))

    all_delta_means: list[float] = []
    all_delta_cis:   list[float] = []

    for trait in BIG_FIVE:
        color = BIG_FIVE_COLORS[trait]
        baseline_val = float(baseline_row[f"{trait}_mean"].iloc[0])
        means = bfi_agg[f"{trait}_mean"].values - baseline_val
        cis   = bfi_agg[f"{trait}_ci"].values
        if trait in lit:
            ax.plot(scales, means, "o-", color=color, linewidth=2.2, markersize=6,
                    label=trait, zorder=4)
            _draw_error_bars(ax, scales, means, cis, color)
        else:
            ax.plot(scales, means, "o-", color=color, linewidth=1.4, markersize=4,
                    alpha=0.35, label=trait, zorder=3)
        # Use all traits for y-axis scaling regardless of highlight
        all_delta_means.extend(means[~np.isnan(means)].tolist())
        all_delta_cis.extend(cis.tolist())

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6,
               label="Baseline (s=0)", zorder=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.3, zorder=1)
    ax.set_xticks(scales)

    if all_delta_means:
        max_ci   = max(all_delta_cis) if all_delta_cis else 0.0
        data_min = min(all_delta_means) - max_ci
        data_max = max(all_delta_means) + max_ci
        margin   = max(0.03, (data_max - data_min) * 0.15)
        ax.set_ylim(data_min - margin, data_max + margin)

    ax.set_xlabel("LoRA scaling factor", fontsize=11)
    ax.set_ylabel("Δ trait score (vs. baseline)", fontsize=11)
    ax.grid(True, alpha=0.25)

    title = "BFI sweep: trait delta from baseline"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=13, fontweight="bold")

    ax.errorbar([], [], yerr=1, fmt="none", color="gray", capsize=3, capthick=1.0,
                elinewidth=1.0, alpha=0.7, label="95% CI")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
              fontsize=9, ncol=7, framealpha=0.85)

    plt.tight_layout()
    out = output_dir / "bfi_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


def plot_capability_sweep(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
    eval_name: str = "capability",
    allowed_drop: float = 0.05,
) -> Path:
    """Capability coherence plot: accuracy vs. LoRA scale with baseline and allowed-drop band.

    Suitable for any accuracy-like eval (mmlu, gsm8k, truthfulqa, arc, etc.).

    Args:
        df: DataFrame with columns: scale, run, accuracy.
        output_dir: Directory to save the figure.
        title_suffix: Optional suffix appended to the figure title.
        eval_name: Eval name used for the figure title and output filename.
        allowed_drop: Maximum acceptable accuracy drop from baseline (default 0.05 = 5 pp).

    Returns:
        Path to the saved figure.
    """
    import matplotlib.pyplot as plt

    cap_agg = _agg_sweep(df, ["accuracy"])
    scales = cap_agg["scale"].values
    means  = cap_agg["accuracy_mean"].values
    cis    = cap_agg["accuracy_ci"].values

    baseline_idx = int(np.argmin(np.abs(scales)))
    baseline_acc = float(means[baseline_idx])

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.axhspan(baseline_acc - allowed_drop, baseline_acc,
               color="#A5D6A7", alpha=0.25, zorder=1,
               label=f"Allowed drop (−{allowed_drop:.0%})")
    ax.axhline(baseline_acc, color="#388E3C", linewidth=1.2,
               linestyle="--", alpha=0.8, zorder=2, label=f"Baseline ({baseline_acc:.3f})")

    color = "#5C6BC0"
    ax.plot(scales, means, "o-", color=color, linewidth=2.2, markersize=6,
            label="accuracy", zorder=4)
    _draw_error_bars(ax, scales, means, cis, color)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    ax.set_xlabel("LoRA scaling factor", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xticks(scales)

    y_min = min(float(np.nanmin(means - cis)), baseline_acc - allowed_drop) - 0.02
    y_max = max(float(np.nanmax(means + cis)), baseline_acc) + 0.04
    ax.set_ylim(y_min, y_max)

    ax.grid(True, alpha=0.25)

    title = f"{eval_name} sweep: capability coherence vs. LoRA scale"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=13, fontweight="bold")

    ax.errorbar([], [], yerr=1, fmt="none", color="gray", capsize=3, capthick=1.0,
                elinewidth=1.0, alpha=0.7, label="95% CI")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              fontsize=9, ncol=4, framealpha=0.85)

    plt.tight_layout()
    out = output_dir / f"{eval_name}_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


def plot_generic_sweep(
    df: pd.DataFrame,
    eval_name: str,
    output_dir: Path,
    title_suffix: str = "",
    compare: list[tuple[str, pd.DataFrame]] | None = None,
) -> Path:
    """Generic sweep line plot for any eval not covered by a specialised plotter.

    Plots all numeric metric columns on a single 0–1 axis with CI bands.
    When *compare* is provided, each additional run is drawn with a distinct
    linestyle at reduced alpha for easy visual comparison.

    Args:
        df: Primary DataFrame with columns: scale, run, + metric columns.
        eval_name: Used for the figure title and filename.
        output_dir: Directory to save the figure.
        title_suffix: Optional suffix appended to the figure title.
        compare: Optional list of ``(label, DataFrame)`` pairs for additional
            runs to overlay on the same axes.

    Returns:
        Path to the saved figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    cols = _metric_cols(df)
    agg  = _agg_sweep(df, cols)
    scales = agg["scale"].values

    colors = cm.tab10.colors  # type: ignore[attr-defined]
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Primary run — solid lines
    for i, col in enumerate(cols):
        color = colors[i % len(colors)]
        means = agg[f"{col}_mean"].values
        cis   = agg[f"{col}_ci"].values
        ax.plot(scales, means, "o" + _LINESTYLES[0], color=color, linewidth=2.0,
                markersize=5, label=col, zorder=4)
        _draw_error_bars(ax, scales, means, cis, color)

    # Comparison runs — distinct dash patterns at reduced alpha
    for run_idx, (run_label, cmp_df) in enumerate(compare or []):
        ls = _LINESTYLES[(run_idx + 1) % len(_LINESTYLES)]
        cmp_cols = _metric_cols(cmp_df)
        cmp_agg = _agg_sweep(cmp_df, cmp_cols)
        cmp_scales = cmp_agg["scale"].values
        for i, col in enumerate(cmp_cols):
            if col not in cols:
                continue
            color = colors[i % len(colors)]
            means = cmp_agg[f"{col}_mean"].values
            cis   = cmp_agg[f"{col}_ci"].values
            ax.plot(cmp_scales, means, "o" + ls, color=color, linewidth=1.6,
                    markersize=4, alpha=0.55, label=f"{col} [{run_label}]", zorder=3)
            _draw_error_bars(ax, cmp_scales, means, cis, color)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    ax.set_xlabel("LoRA scaling factor", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_xticks(scales)
    ax.grid(True, alpha=0.25)

    title = f"{eval_name} sweep"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=13, fontweight="bold")

    n_legend_items = len(cols) * (1 + len(compare or []))
    ncol = min(n_legend_items, 5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13),
              fontsize=9, ncol=ncol, framealpha=0.85)

    plt.tight_layout()
    out = output_dir / f"{eval_name}_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


def plot_parse_rate(
    df: pd.DataFrame,
    eval_name: str,
    output_dir: Path,
    title_suffix: str = "",
) -> Path | None:
    """Parse rate companion plot: fraction of responses successfully scored vs. LoRA scale.

    Only generates a figure when at least one scale point has parse rate < 1.0.
    Returns None (and produces no file) when all points are 100%.

    Args:
        df: Sweep DataFrame with a ``_parse_rate`` column (0–1 per run).
        eval_name: Used for the figure title and output filename.
        output_dir: Directory to save the figure.
        title_suffix: Optional suffix appended to the figure title.

    Returns:
        Path to the saved figure, or None if skipped.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    if "_parse_rate" not in df.columns:
        return None

    agg = _agg_sweep(df, ["_parse_rate"])
    means = agg["_parse_rate_mean"].values
    if (means >= 1.0 - 1e-9).all():
        return None  # perfect parse rate everywhere — nothing to show

    # Use min/max across runs instead of CI — more informative for a bounded count.
    scales = agg["scale"].values
    pr_min = np.array([df[df["scale"] == s]["_parse_rate"].min() for s in scales])
    pr_max = np.array([df[df["scale"] == s]["_parse_rate"].max() for s in scales])
    err_lo = np.clip(means - pr_min, 0, None)
    err_hi = np.clip(pr_max - means, 0, None)

    fig, ax = plt.subplots(figsize=(10, 3.5))

    color = "#78909C"
    ax.axhline(1.0, color="#388E3C", linewidth=1.2, linestyle="--", alpha=0.7,
               label="100% parsed", zorder=2)
    ax.plot(scales, means, "o-", color=color, linewidth=2.0, markersize=5,
            label="parse rate (mean)", zorder=4)
    if (err_lo > 0).any() or (err_hi > 0).any():
        ax.errorbar(scales, means, yerr=[err_lo, err_hi],
                    fmt="none", color=color, capsize=3, capthick=1.0,
                    elinewidth=1.0, alpha=0.7, zorder=5, label="min/max")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)

    ax.set_xlabel("LoRA scaling factor", fontsize=11)
    ax.set_ylabel("Parse rate", fontsize=11)
    ax.set_ylim(max(0.0, float(np.nanmin(pr_min)) - 0.05), 1.0)
    ax.set_xticks(scales)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.25)

    title = f"{eval_name} sweep: parse rate vs. LoRA scale"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)

    plt.tight_layout()
    out = output_dir / f"{eval_name}_parse_rate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ---------------------------------------------------------------------------
# Plot dispatch
# ---------------------------------------------------------------------------

PlotFn = Callable[[pd.DataFrame, Path, str], Path]

# A plot style tag selects a built-in plotter; a PlotFn provides a fully custom one.
PlotStyle = Literal["trait", "bfi", "capability", "generic"]

# Registry mapping eval name → plot style or custom PlotFn.
#
# To add a new eval, insert one line here:
#   "my_eval": "capability"   — for any accuracy-like metric
#   "my_eval": "generic"      — for a quick look at arbitrary numeric columns
#   "my_eval": plot_my_sweep  — for a fully custom plot function
#
# Evals absent from this registry will not be plotted; a warning is printed
# instead so you know exactly what to do.
_PLOT_REGISTRY: dict[str, PlotFn | PlotStyle] = {
    # Behavioral evals
    "trait":      "trait",
    "bfi":        "bfi",
    # Capability evals
    "mmlu":       "capability",
    "gsm8k":      "capability",
    "truthfulqa": "capability",
    # Toy / proxy evals
    "count_o":    "generic",
}


def generate_plots(
    data: SweepData,
    output_dir: Path,
    title_suffix: str = "",
    allowed_drop: float = 0.05,
    highlight: list[str] | None = None,
    show_parse_rate: bool = False,
    compare: list[tuple[str, SweepData]] | None = None,
) -> list[Path]:
    """Generate all plots for the evals present in *data*.

    Uses the plot registry for known evals. Evals not present in the registry
    are skipped with a warning explaining how to add them.

    Args:
        data: SweepData loaded from a run directory.
        output_dir: Directory to save all figures.
        title_suffix: Optional title suffix forwarded to every plot function.
        allowed_drop: Allowed accuracy drop forwarded to capability plotters.
        highlight: Traits to render at full brightness in trait/bfi plots.
            Accepts full names or OCEAN single letters (O/C/E/A/N).
            Defaults to all Big Five.
        show_parse_rate: If True, generate a companion parse rate plot for each
            eval when at least one scale point has parse rate < 100%.
        compare: Optional list of ``(label, SweepData)`` pairs for additional
            runs to overlay on the same plots. Currently supported by
            ``"generic"`` style plots; other styles ignore it.

    Returns:
        List of paths to saved figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for eval_name in data.names():
        df = data.get(eval_name)
        assert df is not None

        entry = _PLOT_REGISTRY.get(eval_name)

        if entry is None:
            print(
                f"\nWARNING: eval '{eval_name}' has no registered plot style — skipping.\n"
                f"  To add it, insert one line in _PLOT_REGISTRY in analyze_results.py:\n"
                f'    "{eval_name}": "capability"   # for accuracy-like metrics\n'
                f'    "{eval_name}": "generic"       # for a quick look at any numeric columns\n'
                f'    "{eval_name}": plot_{eval_name}_sweep  # for a fully custom plot function\n',
                file=sys.stderr,
            )
            continue

        if entry == "capability":
            path = plot_capability_sweep(df, output_dir, title_suffix,
                                         eval_name=eval_name, allowed_drop=allowed_drop)
        elif entry == "trait":
            path = plot_trait_sweep(df, output_dir, title_suffix, highlight=highlight)
        elif entry == "bfi":
            path = plot_bfi_sweep(df, output_dir, title_suffix, highlight=highlight)
        elif entry == "generic":
            others = [
                (lbl, sd.get(eval_name))
                for lbl, sd in (compare or [])
                if sd.get(eval_name) is not None
            ]
            path = plot_generic_sweep(df, eval_name, output_dir, title_suffix,
                                      compare=others or None)
        elif callable(entry):
            path = entry(df, output_dir, title_suffix)
        else:
            raise ValueError(f"Unknown plot style {entry!r} for eval '{eval_name}'")

        saved.append(path)

        if show_parse_rate:
            pr_path = plot_parse_rate(df, eval_name, output_dir, title_suffix)
            if pr_path:
                saved.append(pr_path)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze personality sweep evaluation results")
    parser.add_argument("run_dir", type=Path, nargs="?",
                        help="Run directory produced by the personality eval suite")
    parser.add_argument("--output-dir", type=Path,
                        help="Directory for plots (default: <run_dir>/figures)")
    parser.add_argument("--title", default="",
                        help="Optional title suffix (e.g. persona name)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate and save plots")
    parser.add_argument("--reparse", action="store_true",
                        help="Recompute trait scores from raw model outputs using the fallback parser")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock sweep data for offline testing")
    parser.add_argument("--allowed-drop", type=float, default=0.05,
                        help="Max acceptable MMLU accuracy drop from baseline (default 0.05)")
    parser.add_argument("--highlight", metavar="TRAIT", action="append", default=None,
                        help="Trait to render at full brightness in trait/bfi plots. "
                             "Accepts full name or OCEAN letter (O/C/E/A/N). "
                             "Repeat for multiple. Defaults to all Big Five.")
    parser.add_argument("--show-parse-rate", action="store_true",
                        help="Generate a companion parse rate plot for each eval "
                             "(only produced when parse rate drops below 100%% at any scale).")
    parser.add_argument("--label", default=None,
                        help="Label for the primary run in comparison plots "
                             "(default: run directory basename).")
    parser.add_argument("--compare-dir", metavar="LABEL:PATH", action="append", default=None,
                        help="Add a comparison run directory as LABEL:PATH. "
                             "Repeat for multiple comparison runs. "
                             "The comparison run's data is overlaid on generic-style plots.")
    args = parser.parse_args()

    if args.mock:
        data = _mock_sweep_data()
        output_dir = args.output_dir or Path("scratch/analysis_mock/figures")
        print("Using mock sweep data")
    else:
        if not args.run_dir:
            parser.error("run_dir is required (or use --mock)")
        print(f"Loading sweep data from {args.run_dir} ...")
        data = load_sweep_data(args.run_dir, reparse=args.reparse)
        output_dir = args.output_dir or args.run_dir / "figures"

    # Print summary tables for all evals found
    for eval_name in data.names():
        df = data.get(eval_name)
        assert df is not None
        cols = _metric_cols(df)
        agg = _agg_sweep(df, cols)
        print_sweep_table(agg, cols, f"{eval_name.upper()} SWEEP: scores vs. LoRA scale")

    # Parse comparison run dirs
    compare: list[tuple[str, SweepData]] | None = None
    if args.compare_dir:
        compare = []
        for entry in args.compare_dir:
            if ":" not in entry:
                parser.error(f"--compare-dir must be LABEL:PATH, got: {entry!r}")
            label, _, path_str = entry.partition(":")
            cmp_path = Path(path_str)
            if not cmp_path.exists():
                parser.error(f"--compare-dir path does not exist: {cmp_path}")
            print(f"Loading comparison run '{label}' from {cmp_path} ...")
            compare.append((label, load_sweep_data(cmp_path, reparse=args.reparse)))

    if args.visualize:
        _setup_matplotlib()
        print(f"\nGenerating plots → {output_dir}")
        saved = generate_plots(data, output_dir, title_suffix=args.title,
                               allowed_drop=args.allowed_drop, highlight=args.highlight,
                               show_parse_rate=args.show_parse_rate, compare=compare)
        if not saved:
            print("  (no eval data found — nothing to plot)")
        else:
            print(f"\n✅ Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
