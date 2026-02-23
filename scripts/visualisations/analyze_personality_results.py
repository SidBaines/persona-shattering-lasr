#!/usr/bin/env python3
"""Analyze and visualize BFI personality evaluation results.

Supports two directory layouts automatically:

  Single-run layout  (one run, multiple models):
    <run_dir>/
      <model>/
        <eval>/run_info.json

  Multi-run layout  (multiple runs per model, for within-model variance):
    <root>/
      <model>/
        <run_N>/
          <eval>/run_info.json

  Sweep layout  (LoRA scaling sweep, model names like base / lora_+1p25x):
    Either layout above; sweep is detected from model names.

Usage:
    # Multi-run or single-run — auto-detected
    uv run python scripts/visualisations/analyze_personality_results.py \\
        scratch/evals/bfi_api --visualize

    # With output
    uv run python scripts/visualisations/analyze_personality_results.py \\
        scratch/evals/bfi_api \\
        --output scratch/evals/bfi_api/analysis/results.csv \\
        --output-dir scratch/evals/bfi_api/analysis \\
        --visualize

    # Mock sweep data for testing
    uv run python scripts/visualisations/analyze_personality_results.py --mock
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
TRAIT_COLORS = {
    "Openness":          "#2196F3",
    "Conscientiousness": "#FF9800",
    "Extraversion":      "#4CAF50",
    "Agreeableness":     "#9C27B0",
    "Neuroticism":       "#F44336",
}


# ---------------------------------------------------------------------------
# Data loading — layout auto-detection
# ---------------------------------------------------------------------------

def _extract_scores(log_path: Path) -> dict[str, float] | None:
    with open(log_path) as f:
        log = json.load(f)
    if log.get("status") != "success":
        return None
    metrics = log["results"]["scores"][0]["metrics"]
    return {trait: metrics[trait]["value"] for trait in TRAITS if trait in metrics}


def _load_from_info(info_path: Path, model: str, run: str) -> dict | None:
    with open(info_path) as f:
        info = json.load(f)
    if info.get("status") != "ok":
        print(f"  skip {model}/{run}: {info.get('error')}", file=sys.stderr)
        return None
    log_path = info.get("native", {}).get("inspect_log_path")
    if not log_path:
        print(f"  skip {model}/{run}: no inspect_log_path", file=sys.stderr)
        return None
    scores = _extract_scores(Path(log_path))
    if scores is None:
        print(f"  skip {model}/{run}: log not success", file=sys.stderr)
        return None
    return {"model": model, "run": run, **scores}


def _is_multi_run_layout(root: Path) -> bool:
    """Detect whether root uses multi-run layout (<model>/<run>/<eval>/run_info.json)."""
    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        for child in model_dir.iterdir():
            if not child.is_dir():
                continue
            # Multi-run: child is a run dir containing eval dirs with run_info.json
            for eval_dir in child.iterdir():
                if eval_dir.is_dir() and (eval_dir / "run_info.json").exists():
                    return True
            # Single-run: child is already an eval dir with run_info.json
            if (child / "run_info.json").exists():
                return False
    return False


def load_data(root: Path) -> pd.DataFrame:
    """Load all runs from root, auto-detecting layout. Returns DataFrame with
    columns: model, run, Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism."""
    records = []

    if _is_multi_run_layout(root):
        # <root>/<model>/<run>/<eval>/run_info.json
        for model_dir in sorted(root.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            for run_dir in sorted(model_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                run = run_dir.name
                for eval_dir in sorted(run_dir.iterdir()):
                    info_path = eval_dir / "run_info.json"
                    if not info_path.exists():
                        continue
                    rec = _load_from_info(info_path, model, run)
                    if rec:
                        records.append(rec)
    else:
        # <root>/<model>/<eval>/run_info.json  — treat as single run
        for model_dir in sorted(root.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            for eval_dir in sorted(model_dir.iterdir()):
                info_path = eval_dir / "run_info.json"
                if not info_path.exists():
                    continue
                rec = _load_from_info(info_path, model, run="run_1")
                if rec:
                    records.append(rec)

    if not records:
        raise ValueError(f"No successful runs found under {root}")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Sweep detection
# ---------------------------------------------------------------------------

def _parse_scale(model_name: str) -> float | None:
    if model_name == "base":
        return 0.0
    m = re.match(r"lora_([+-]?\d+)p(\d+)x", model_name)
    if m:
        i, d = int(m.group(1)), int(m.group(2))
        return i + (d / 100.0) * (1 if i >= 0 else -1)
    return None


def _as_sweep(df: pd.DataFrame) -> pd.DataFrame | None:
    scales = [_parse_scale(m) for m in df["model"]]
    if all(s is None for s in scales):
        return None
    df = df.copy()
    df["scale"] = scales
    return df[df["scale"].notna()].sort_values("scale").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stats
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


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, grp in df.groupby("model"):
        row: dict = {"model": model, "n_runs": len(grp)}
        for trait in TRAITS:
            vals = grp[trait].values
            row[f"{trait}_mean"] = vals.mean()
            row[f"{trait}_ci"]   = _ci95(vals)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------

def print_raw_table(df: pd.DataFrame) -> None:
    has_multi = df.groupby("model")["run"].nunique().max() > 1
    print("\n" + "=" * 95)
    print("INDIVIDUAL RUNS")
    print("=" * 95)
    if has_multi:
        header = f"{'Model':<22} {'Run':<8}" + "".join(f"{t:<18}" for t in TRAITS)
    else:
        header = f"{'Model':<22} " + "".join(f"{t:<18}" for t in TRAITS)
    print(header)
    print("-" * 95)
    for _, row in df.sort_values(["model", "run"]).iterrows():
        vals = "".join(f"{row[t]:.4f}{'':>12}" for t in TRAITS)
        if has_multi:
            print(f"{row['model']:<22} {row['run']:<8}{vals}")
        else:
            print(f"{row['model']:<22} {vals}")


def print_agg_table(agg: pd.DataFrame) -> None:
    print("\n" + "=" * 95)
    print("MODEL SUMMARY  (mean ± 95% CI)")
    print("=" * 95)
    header = f"{'Model':<22} {'N':<4}" + "".join(f"{t:<22}" for t in TRAITS)
    print(header)
    print("-" * 95)
    for _, row in agg.iterrows():
        vals = "".join(f"{row[f'{t}_mean']:.3f}±{row[f'{t}_ci']:.3f}{'':>8}" for t in TRAITS)
        print(f"{row['model']:<22} {int(row['n_runs']):<4}{vals}")
    print("=" * 95)


def print_sweep_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("PERSONALITY SWEEP: TRAIT SCORES ACROSS LORA SCALES")
    print("=" * 100)
    header = f"{'Scale':<10}" + "".join(f"{t:<18}" for t in TRAITS)
    print(header)
    print("-" * 100)
    for _, row in df.iterrows():
        marker = " ← baseline" if abs(row["scale"]) < 0.01 else ""
        vals = "".join(f"{row[t]:.4f}{'':>12}" for t in TRAITS)
        print(f"{row['scale']:+7.2f}   {vals}{marker}")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

def _mock_sweep() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    scales = np.arange(-2.0, 2.25, 0.25)
    base = {"Openness": 0.65, "Conscientiousness": 0.70, "Extraversion": 0.50,
            "Agreeableness": 0.60, "Neuroticism": 0.45}
    slopes = {"Openness": 0.08, "Conscientiousness": -0.03, "Extraversion": 0.05,
              "Agreeableness": -0.12, "Neuroticism": 0.06}
    records = []
    for s in scales:
        model = "base" if s == 0.0 else f"lora_{s:+.2f}x".replace(".", "p")
        scores = {t: float(np.clip(base[t] + slopes[t] * s + rng.normal(0, 0.02), 0, 1))
                  for t in TRAITS}
        records.append({"model": model, "run": "run_1", "scale": s, **scores})
    return pd.DataFrame(records)


def _mock_multi_run() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    records = []
    for model, base in [("llama31_8b", 0.75), ("qwen3_14b", 0.65)]:
        for run in ["run_1", "run_2", "run_3"]:
            scores = {t: float(np.clip(base + rng.normal(0, 0.04), 0, 1)) for t in TRAITS}
            records.append({"model": model, "run": run, **scores})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plots — multi-run / single-run
# ---------------------------------------------------------------------------

def _setup_matplotlib() -> None:
    import matplotlib
    matplotlib.use("Agg")


def plot_within_model(df: pd.DataFrame, output_dir: Path) -> None:
    """Dots = individual runs, diamond = mean ± CI, one panel per trait."""
    import matplotlib.pyplot as plt

    models = sorted(df["model"].unique())
    has_multi = df.groupby("model")["run"].nunique().max() > 1

    fig, axes = plt.subplots(1, len(TRAITS), figsize=(4 * len(TRAITS), max(3, len(models) * 0.8 + 2)))
    if len(TRAITS) == 1:
        axes = [axes]

    for ax, trait in zip(axes, TRAITS):
        color = TRAIT_COLORS[trait]
        for i, model in enumerate(models):
            vals = df[df["model"] == model][trait].values
            mean, ci = vals.mean(), _ci95(vals)
            if has_multi:
                ax.scatter([i] * len(vals), vals, color=color, alpha=0.45, s=40, zorder=3)
            ax.errorbar(i, mean, yerr=ci, fmt="D", color=color,
                        markersize=8, capsize=5, linewidth=2, zorder=4)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
        ax.set_title(trait, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("Score" if trait == TRAITS[0] else "")

    subtitle = "(dots = individual runs, diamond = mean ± 95% CI)" if has_multi else "(diamond = score, bar = 95% CI)"
    fig.suptitle(f"BFI: Within-Model Consistency\n{subtitle}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "bfi_within_model.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_cross_model(agg: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bar chart: traits on x-axis, one bar per model, error bars = 95% CI."""
    import matplotlib.pyplot as plt

    models = sorted(agg["model"].tolist())
    x = np.arange(len(TRAITS))
    width = 0.8 / len(models)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(models)))

    fig, ax = plt.subplots(figsize=(max(10, len(TRAITS) * 2), 5))
    for i, model in enumerate(models):
        row = agg[agg["model"] == model].iloc[0]
        means = [row[f"{t}_mean"] for t in TRAITS]
        cis   = [row[f"{t}_ci"]   for t in TRAITS]
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(x + offset, means, width * 0.9, label=model, color=colors[i], alpha=0.85)
        ax.errorbar(x + offset, means, yerr=cis, fmt="none",
                    ecolor="black", capsize=4, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(TRAITS, fontsize=11)
    ax.set_ylabel("Mean Score", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("BFI: Cross-Model Comparison (mean ± 95% CI)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = output_dir / "bfi_cross_model.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_radar(agg: pd.DataFrame, output_dir: Path) -> None:
    """Radar chart: one polygon per model (mean scores)."""
    import matplotlib.pyplot as plt

    models = sorted(agg["model"].tolist())
    angles = np.linspace(0, 2 * np.pi, len(TRAITS), endpoint=False).tolist()
    angles += angles[:1]
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(models)))

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection="polar"))
    for i, model in enumerate(models):
        row = agg[agg["model"] == model].iloc[0]
        values = [row[f"{t}_mean"] for t in TRAITS] + [row[f"{TRAITS[0]}_mean"]]
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.12, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(TRAITS, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, alpha=0.3)
    ax.set_title("BFI: Big Five Profile by Model", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    out = output_dir / "bfi_radar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# Plots — LoRA sweep
# ---------------------------------------------------------------------------

def plot_sweep_lines(df: pd.DataFrame, output_dir: Path) -> None:
    """One subplot per trait: score vs LoRA scale."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(TRAITS), 1, figsize=(14, 3.5 * len(TRAITS)))
    if len(TRAITS) == 1:
        axes = [axes]

    for ax, trait in zip(axes, TRAITS):
        color = TRAIT_COLORS[trait]
        scales = df["scale"].values
        values = df[trait].values
        ax.plot(scales, values, "o-", color=color, linewidth=2.5, markersize=7)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5, label="Baseline (s=0)")
        ax.axvline(1, color="green", linestyle=":", alpha=0.5, label="Standard LoRA (s=1)")
        ax.set_ylabel(f"{trait} Score", fontsize=11)
        ax.set_title(f"{trait} vs. LoRA Scale", fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("LoRA Scaling Factor", fontsize=11)
    fig.suptitle("BFI Sweep: Trait Scores vs LoRA Scale", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "bfi_sweep_lines.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_sweep_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap: traits × LoRA scales."""
    import matplotlib.pyplot as plt

    pivot = df.set_index("scale")[TRAITS].T
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns)), 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(TRAITS)))
    ax.set_xticklabels([f"{s:+.2f}" for s in pivot.columns], fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(TRAITS, fontsize=11)

    for i in range(len(TRAITS)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.5 else "black", fontsize=7)

    baseline_cols = list(pivot.columns)
    if 0.0 in baseline_cols:
        bi = baseline_cols.index(0.0)
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((bi - 0.5, -0.5), 1, len(TRAITS),
                               fill=False, edgecolor="blue", linewidth=2.5))

    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Score", fontsize=10)
    ax.set_xlabel("LoRA Scaling Factor", fontsize=11)
    ax.set_title("BFI Sweep Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "bfi_sweep_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def plot_sweep_radar(df: pd.DataFrame, output_dir: Path) -> None:
    """Radar chart for selected scale points."""
    import matplotlib.pyplot as plt

    scales = df["scale"].values
    n = len(scales)
    selected = [scales[i] for i in ([0, n // 4, n // 2, 3 * n // 4, n - 1] if n > 5 else range(n))]

    angles = np.linspace(0, 2 * np.pi, len(TRAITS), endpoint=False).tolist() + [0]
    colors = ["#E53935", "#FB8C00", "#43A047", "#1E88E5", "#8E24AA"]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection="polar"))
    for idx, scale in enumerate(selected):
        row = df[df["scale"] == scale]
        if row.empty:
            continue
        values = row[TRAITS].iloc[0].tolist() + [row[TRAITS[0]].iloc[0]]
        label = "Baseline" if abs(scale) < 0.01 else f"s={scale:+.2f}"
        ax.plot(angles, values, "o-", linewidth=2, label=label, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.12, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(TRAITS, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, alpha=0.3)
    ax.set_title("BFI Sweep: Personality Profile", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    out = output_dir / "bfi_sweep_radar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze BFI personality evaluation results")
    parser.add_argument("root", type=Path, nargs="?",
                        help="Root directory (single-run or multi-run layout)")
    parser.add_argument("--output", type=Path, help="Save results to CSV")
    parser.add_argument("--output-dir", type=Path, help="Directory for plots")
    parser.add_argument("--visualize", action="store_true", help="Generate plots")
    parser.add_argument("--mock", choices=["sweep", "multi"],
                        help="Use mock data: 'sweep' for LoRA sweep, 'multi' for multi-run comparison")
    args = parser.parse_args()

    if args.mock:
        df = _mock_sweep() if args.mock == "sweep" else _mock_multi_run()
        output_dir = args.output_dir or Path("scratch/analysis_mock")
        print(f"Using mock '{args.mock}' data ({len(df)} rows)")
    else:
        if not args.root:
            parser.error("root directory is required (or use --mock)")
        df = load_data(args.root)
        output_dir = args.output_dir or args.root / "analysis"

    sweep_df = _as_sweep(df)
    is_sweep = sweep_df is not None

    if is_sweep:
        # Per-scale mean across runs (sweep usually has one run per scale)
        sweep_agg = sweep_df.groupby("scale")[TRAITS].mean().reset_index()
        sweep_agg["model"] = sweep_agg["scale"].apply(
            lambda s: "base" if abs(s) < 0.01 else f"lora_{s:+.2f}x"
        )
        print_sweep_table(sweep_agg.assign(**{t: sweep_agg[t] for t in TRAITS}))

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            sweep_agg.to_csv(args.output, index=False)
            print(f"\n✓ Results → {args.output}")

        if args.visualize:
            _setup_matplotlib()
            output_dir.mkdir(parents=True, exist_ok=True)
            print("\nGenerating sweep plots...")
            plot_sweep_lines(sweep_agg, output_dir)
            plot_sweep_heatmap(sweep_agg, output_dir)
            plot_sweep_radar(sweep_agg, output_dir)
            print(f"\n✅ Plots saved to {output_dir}")
    else:
        agg = aggregate(df)
        print_raw_table(df)
        print_agg_table(agg)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.output, index=False)
            agg_path = args.output.parent / (args.output.stem + "_agg.csv")
            agg.to_csv(agg_path, index=False)
            print(f"\n✓ Raw → {args.output}")
            print(f"✓ Aggregated → {agg_path}")

        if args.visualize:
            _setup_matplotlib()
            output_dir.mkdir(parents=True, exist_ok=True)
            print("\nGenerating plots...")
            plot_within_model(df, output_dir)
            plot_cross_model(agg, output_dir)
            plot_radar(agg, output_dir)
            print(f"\n✅ Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
