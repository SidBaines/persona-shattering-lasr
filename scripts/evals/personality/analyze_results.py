#!/usr/bin/env python3
"""Analyze and visualize personality evaluation results from a sweep run.

Produces two plots from a single run directory:

  trait_sweep.png  — primary research plot
    • Big Five from TRAIT benchmark (absolute 0–1, full color)
    • Dark Triad (Mach, Narc, Psychopathy) dimmed + dashed
    • MMLU accuracy on right y-axis (pale grey)
    • Human population baselines as horizontal dashed lines (TODO: fill in values)
    • 95% CI bands when n_runs > 1
    • Parse rate annotated when < 100%

  bfi_sweep.png  — sanity check / cross-validation
    • Big Five from BFI benchmark (delta from baseline, y=0 = unmodified model)
    • y-axis zoomed to data range
    • 95% CI bands when n_runs > 1

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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIG_FIVE = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
DARK_TRIAD = ["Machiavellianism", "Narcissism", "Psychopathy"]
ALL_TRAIT_COLS = BIG_FIVE + DARK_TRIAD

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

# Human population mean scores (0–1 scale) for the Big Five from TRAIT benchmark.
# TODO(@irakli): replace with validated published norms.
HUMAN_BASELINES: dict[str, float] = {
    "Openness":          0.69,
    "Conscientiousness": 0.68,
    "Extraversion":      0.60,
    "Agreeableness":     0.67,
    "Neuroticism":       0.50,
}

# Eval names written into run_info.json by the suite (must match eval_suite.py).
_EVAL_NAME_BFI   = "bfi"
_EVAL_NAME_TRAIT = "trait"
_EVAL_NAME_MMLU  = "mmlu"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SweepData:
    """Per-eval DataFrames loaded from a single sweep run directory.

    Each DataFrame has columns: scale (float), run (str), + metric columns.
    - bfi/trait: one column per trait (Big Five; trait also has Dark Triad)
    - mmlu: single 'accuracy' column

    None means the eval was not present in the run.
    """
    bfi:   pd.DataFrame | None  # scale, run, Openness, ..., Neuroticism
    trait: pd.DataFrame | None  # scale, run, Openness, ..., Psychopathy
    mmlu:  pd.DataFrame | None  # scale, run, accuracy


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _extract_scores(log_path: Path) -> dict[str, float] | None:
    """Extract metric values from an inspect log JSON."""
    with open(log_path) as f:
        log = json.load(f)
    if log.get("status") != "success":
        return None
    metrics = log["results"]["scores"][0]["metrics"]
    return {k: v["value"] for k, v in metrics.items() if isinstance(v, dict) and "value" in v}


def _extract_scores_reparsed(log_path: Path, eval_type: str) -> dict[str, float] | None:
    """Like _extract_scores but recomputes trait scores using the fallback parser."""
    from scripts.evals.personality.log_answer_parser import rescore_log
    result = rescore_log(log_path, eval_type)
    return result.scores if result.scores else None


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
        scores = _extract_scores_reparsed(Path(log_path), eval_type)
    else:
        scores = _extract_scores(Path(log_path))
    if scores is None:
        print(f"  skip {model}/{run}: log not success", file=sys.stderr)
        return None
    scale = info.get("scale")  # float | None; None = base model
    return {"model": model, "run": run, "scale": scale, **scores}


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
        df["scale"] = df["scale"].apply(lambda s: 0.0 if s is None else s)
    else:
        df["scale"] = [_parse_scale(m) for m in df["model"]]
    return df


def load_sweep_data(run_dir: Path, reparse: bool = False) -> SweepData:
    """Load BFI, TRAIT, and MMLU results from a sweep run directory.

    Walks ``run_dir/<model_spec>/<eval_name>/run_info.json``, routes each
    record to the appropriate DataFrame by eval name, and returns a
    :class:`SweepData` with one DataFrame per eval type.

    Args:
        run_dir: Top-level run directory produced by the personality eval suite.
        reparse: If True, recompute trait scores from raw model outputs using
            the fallback answer parser rather than the inspect scorer values.

    Returns:
        SweepData with bfi, trait, mmlu DataFrames (None if not present).
    """
    records: dict[str, list[dict]] = {
        _EVAL_NAME_BFI:   [],
        _EVAL_NAME_TRAIT: [],
        _EVAL_NAME_MMLU:  [],
    }

    for model_dir in sorted(run_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name in ("figures", "analysis"):
            continue
        model = model_dir.name

        for eval_dir in sorted(model_dir.iterdir()):
            eval_name = eval_dir.name
            if eval_name not in records:
                continue

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

            eval_type = eval_name if eval_name in ("bfi", "trait") else "bfi"
            for info_path, run_label in info_paths:
                rec = _load_from_info(
                    info_path, model, run_label,
                    reparse=(reparse and eval_name != _EVAL_NAME_MMLU),
                    eval_type=eval_type,
                )
                if rec:
                    records[eval_name].append(rec)

    def _to_df(recs: list[dict]) -> pd.DataFrame | None:
        if not recs:
            return None
        df = pd.DataFrame(recs)
        df = _normalise_scale_col(df)
        return df[df["scale"].notna()].sort_values(["scale", "run"]).reset_index(drop=True)

    return SweepData(
        bfi=_to_df(records[_EVAL_NAME_BFI]),
        trait=_to_df(records[_EVAL_NAME_TRAIT]),
        mmlu=_to_df(records[_EVAL_NAME_MMLU]),
    )


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

    # BFI: Big Five only, 3 runs per scale
    bfi_base = {"Openness": 0.65, "Conscientiousness": 0.70, "Extraversion": 0.50,
                "Agreeableness": 0.60, "Neuroticism": 0.45}
    bfi_slope = {"Openness": 0.08, "Conscientiousness": -0.03, "Extraversion": 0.05,
                 "Agreeableness": -0.12, "Neuroticism": 0.06}
    bfi_records = []
    for s in base_scales:
        for run in ["run_1", "run_2", "run_3"]:
            scores = {t: float(np.clip(bfi_base[t] + bfi_slope[t] * s + rng.normal(0, 0.025), 0, 1))
                      for t in BIG_FIVE}
            bfi_records.append({"model": "base" if s == 0.0 else f"lora_{s:+.2f}x",
                                 "run": run, "scale": s, **scores})

    # TRAIT: Big Five + Dark Triad, 3 runs per scale
    trait_base = {**bfi_base, "Machiavellianism": 0.40, "Narcissism": 0.45, "Psychopathy": 0.35}
    trait_slope = {**bfi_slope, "Machiavellianism": 0.03, "Narcissism": 0.02, "Psychopathy": 0.01}
    trait_records = []
    for s in base_scales:
        for run in ["run_1", "run_2", "run_3"]:
            scores = {t: float(np.clip(trait_base[t] + trait_slope[t] * s + rng.normal(0, 0.025), 0, 1))
                      for t in ALL_TRAIT_COLS}
            trait_records.append({"model": "base" if s == 0.0 else f"lora_{s:+.2f}x",
                                   "run": run, "scale": s, **scores})

    # MMLU: coarser grid, 3 runs
    mmlu_scales = [s for s in np.arange(-2.0, 2.25, 0.5) if round(s, 10) != 0.0]
    mmlu_base_scales = [0.0] + list(mmlu_scales)
    mmlu_records = []
    for s in mmlu_base_scales:
        for run in ["run_1", "run_2", "run_3"]:
            acc = float(np.clip(0.62 - 0.04 * abs(s) + rng.normal(0, 0.01), 0, 1))
            mmlu_records.append({"model": "base" if s == 0.0 else f"lora_{s:+.2f}x",
                                  "run": run, "scale": s, "accuracy": acc})

    def _df(records: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        return _normalise_scale_col(df).sort_values(["scale", "run"]).reset_index(drop=True)

    return SweepData(bfi=_df(bfi_records), trait=_df(trait_records), mmlu=_df(mmlu_records))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _setup_matplotlib() -> None:
    import matplotlib
    matplotlib.use("Agg")


def _draw_ci_band(ax, scales, means, cis, color, alpha: float = 0.15) -> None:
    """Fill CI band around a line if any CI > 0."""
    if any(c > 0 for c in cis):
        means_arr = np.array(means)
        cis_arr = np.array(cis)
        ax.fill_between(scales, means_arr - cis_arr, means_arr + cis_arr,
                        color=color, alpha=alpha, linewidth=0)


def plot_trait_sweep(
    data: SweepData,
    output_dir: Path,
    title_suffix: str = "",
) -> Path:
    """Primary research plot: TRAIT Big Five + Dark Triad + MMLU + human baselines.

    Args:
        data: SweepData loaded from a run directory.
        output_dir: Directory to save the figure.
        title_suffix: Optional suffix appended to the figure title (e.g. persona name).

    Returns:
        Path to the saved figure.
    """
    import matplotlib.pyplot as plt

    if data.trait is None:
        raise ValueError("SweepData.trait is None — no TRAIT eval data found")

    trait_agg = _agg_sweep(data.trait, ALL_TRAIT_COLS)
    scales = trait_agg["scale"].values

    fig, ax = plt.subplots(figsize=(12, 5))

    # --- Big Five: full color lines + CI bands ---
    for trait in BIG_FIVE:
        color = BIG_FIVE_COLORS[trait]
        means = trait_agg[f"{trait}_mean"].values
        cis   = trait_agg[f"{trait}_ci"].values
        ax.plot(scales, means, "o-", color=color, linewidth=2.2, markersize=6, label=trait, zorder=4)
        _draw_ci_band(ax, scales, means, cis, color)

    # --- Dark Triad: dimmed dashed lines + CI bands ---
    for trait in DARK_TRIAD:
        color = DARK_TRIAD_COLORS[trait]
        means = trait_agg[f"{trait}_mean"].values
        cis   = trait_agg[f"{trait}_ci"].values
        ax.plot(scales, means, "--", color=color, linewidth=1.4, markersize=4,
                alpha=0.45, label=trait, zorder=3)
        _draw_ci_band(ax, scales, means, cis, color, alpha=0.07)

    # --- Human baselines: dashed horizontal lines per Big Five trait ---
    for trait, human_val in HUMAN_BASELINES.items():
        ax.axhline(human_val, color=BIG_FIVE_COLORS[trait], linewidth=0.8,
                   linestyle=":", alpha=0.55, zorder=2)

    # --- MMLU: right y-axis, pale grey ---
    if data.mmlu is not None:
        mmlu_agg = _agg_sweep(data.mmlu, ["accuracy"])
        ax2 = ax.twinx()
        mmlu_scales = mmlu_agg["scale"].values
        mmlu_means  = mmlu_agg["accuracy_mean"].values
        mmlu_cis    = mmlu_agg["accuracy_ci"].values
        ax2.plot(mmlu_scales, mmlu_means, "s--", color="#BDBDBD", linewidth=1.4,
                 markersize=5, alpha=0.7, label="MMLU", zorder=2)
        _draw_ci_band(ax2, mmlu_scales, mmlu_means, mmlu_cis, "#BDBDBD", alpha=0.10)
        ax2.set_ylabel("MMLU accuracy", fontsize=10, color="#9E9E9E")
        ax2.tick_params(axis="y", colors="#9E9E9E")
        ax2.set_ylim(0, 1)
        # Add MMLU to legend via proxy
        ax.plot([], [], "s--", color="#BDBDBD", linewidth=1.4, markersize=5,
                alpha=0.7, label="MMLU (right axis)")

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    ax.set_xlabel("LoRA scaling factor", fontsize=11)
    ax.set_ylabel("Trait score (0–1)", fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)

    # Legend: two columns so Dark Triad + MMLU don't crowd Big Five
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.85)

    title = "TRAIT sweep: personality scores vs. LoRA scale"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    out = output_dir / "trait_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


def plot_bfi_sweep(
    data: SweepData,
    output_dir: Path,
    title_suffix: str = "",
) -> Path:
    """Sanity-check plot: BFI Big Five centred at baseline (delta from scale=0).

    Args:
        data: SweepData loaded from a run directory.
        output_dir: Directory to save the figure.
        title_suffix: Optional suffix appended to the figure title.

    Returns:
        Path to the saved figure.
    """
    import matplotlib.pyplot as plt

    if data.bfi is None:
        raise ValueError("SweepData.bfi is None — no BFI eval data found")

    bfi_agg = _agg_sweep(data.bfi, BIG_FIVE)

    # Compute per-trait baseline (scale=0) mean for delta calculation.
    baseline_row = bfi_agg[bfi_agg["scale"].abs() < 1e-9]
    if baseline_row.empty:
        # No exact baseline point — use the row closest to 0.
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
        ax.plot(scales, means, "o-", color=color, linewidth=2.2, markersize=6,
                label=trait, zorder=4)
        _draw_ci_band(ax, scales, means, cis, color)
        all_delta_means.extend(means[~np.isnan(means)].tolist())
        all_delta_cis.extend(cis.tolist())

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6,
               label="Baseline (s=0)", zorder=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.3, zorder=1)

    # Zoom y-axis to data range with a small margin.
    if all_delta_means:
        max_ci   = max(all_delta_cis) if all_delta_cis else 0.0
        data_min = min(all_delta_means) - max_ci
        data_max = max(all_delta_means) + max_ci
        margin   = max(0.03, (data_max - data_min) * 0.15)
        ax.set_ylim(data_min - margin, data_max + margin)

    ax.set_xlabel("LoRA scaling factor", fontsize=11)
    ax.set_ylabel("Δ trait score (vs. baseline)", fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)

    title = "BFI sweep: trait delta from baseline"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    out = output_dir / "bfi_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


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

    # Print summary tables
    if data.trait is not None:
        agg = _agg_sweep(data.trait, ALL_TRAIT_COLS)
        print_sweep_table(agg, ALL_TRAIT_COLS, "TRAIT SWEEP: scores vs. LoRA scale")
    if data.bfi is not None:
        agg = _agg_sweep(data.bfi, BIG_FIVE)
        print_sweep_table(agg, BIG_FIVE, "BFI SWEEP: scores vs. LoRA scale")
    if data.mmlu is not None:
        agg = _agg_sweep(data.mmlu, ["accuracy"])
        print_sweep_table(agg, ["accuracy"], "MMLU: accuracy vs. LoRA scale")

    if args.visualize:
        _setup_matplotlib()
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating plots → {output_dir}")
        if data.trait is not None:
            plot_trait_sweep(data, output_dir, title_suffix=args.title)
        else:
            print("  (skipping trait_sweep.png — no TRAIT data)")
        if data.bfi is not None:
            plot_bfi_sweep(data, output_dir, title_suffix=args.title)
        else:
            print("  (skipping bfi_sweep.png — no BFI data)")
        print(f"\n✅ Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
