#!/usr/bin/env python3
"""Analyze and visualize personality evaluation results.

Extracts Big Five trait scores from Inspect eval logs and compares across models.

Usage:
    # Analyze latest run
    uv run python scripts/analyze_personality_results.py \\
        scratch/evals/personality_production

    # Analyze specific run
    uv run python scripts/analyze_personality_results.py \\
        scratch/evals/personality_production/20260219_153126

    # Save comparison table
    uv run python scripts/analyze_personality_results.py \\
        scratch/evals/personality_production \\
        --output results.csv
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


def find_latest_run(output_root: Path) -> Path:
    """Find the most recent run directory."""
    run_dirs = [d for d in output_root.iterdir() if d.is_dir() and d.name.startswith("202")]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {output_root}")
    return max(run_dirs, key=lambda d: d.name)


def extract_personality_scores(inspect_log_path: Path) -> dict[str, float]:
    """Extract Big Five trait scores from Inspect log."""
    with open(inspect_log_path) as f:
        log = json.load(f)

    if log["status"] != "success":
        return {}

    # Extract scores from first scorer (personality metrics)
    results = log.get("results", {})
    scores = results.get("scores", [])
    if not scores:
        return {}

    metrics = scores[0].get("metrics", {})
    return {trait: metric["value"] for trait, metric in metrics.items()}


def collect_results(run_dir: Path) -> list[dict[str, Any]]:
    """Collect personality scores from all models in a run."""
    results = []

    # Iterate through model directories
    for model_dir in run_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Iterate through eval directories
        for eval_dir in model_dir.iterdir():
            if not eval_dir.is_dir():
                continue

            eval_name = eval_dir.name
            run_info_path = eval_dir / "run_info.json"

            if not run_info_path.exists():
                continue

            # Read run info to get inspect log path
            with open(run_info_path) as f:
                run_info = json.load(f)

            if run_info["status"] != "ok":
                print(f"⚠️  Skipping {model_name}/{eval_name}: {run_info.get('error', 'unknown error')}")
                continue

            native = run_info.get("native", {})
            inspect_log_path = native.get("inspect_log_path")

            if not inspect_log_path:
                print(f"⚠️  No inspect log for {model_name}/{eval_name}")
                continue

            # Extract personality scores
            scores = extract_personality_scores(Path(inspect_log_path))

            if scores:
                results.append({
                    "model": model_name,
                    "eval": eval_name,
                    **scores,
                })

    return results


def format_comparison_table(df: pd.DataFrame) -> str:
    """Format results as a readable comparison table."""
    if df.empty:
        return "No results found"

    # Pivot to show models as rows, traits as columns
    traits = [col for col in df.columns if col not in ["model", "eval"]]

    output = []
    output.append("=" * 80)
    output.append("PERSONALITY TRAIT COMPARISON")
    output.append("=" * 80)

    for eval_name in df["eval"].unique():
        eval_df = df[df["eval"] == eval_name]
        output.append(f"\n{eval_name.upper()}")
        output.append("-" * 80)

        # Create formatted table
        table_data = []
        for _, row in eval_df.iterrows():
            trait_scores = [f"{row[trait]:.3f}" for trait in traits]
            table_data.append([row["model"]] + trait_scores)

        # Print header
        header = ["Model"] + traits
        col_widths = [max(len(str(x)) for x in col) for col in zip(header, *table_data)]

        header_line = "  ".join(h.ljust(w) for h, w in zip(header, col_widths))
        output.append(header_line)
        output.append("-" * len(header_line))

        # Print rows
        for row in table_data:
            row_line = "  ".join(str(x).ljust(w) for x, w in zip(row, col_widths))
            output.append(row_line)

    output.append("\n" + "=" * 80)
    return "\n".join(output)


def calculate_deltas(df: pd.DataFrame, baseline_model: str) -> pd.DataFrame:
    """Calculate trait deltas relative to baseline model."""
    if baseline_model not in df["model"].values:
        print(f"⚠️  Baseline model '{baseline_model}' not found")
        return pd.DataFrame()

    traits = [col for col in df.columns if col not in ["model", "eval"]]
    results = []

    for eval_name in df["eval"].unique():
        eval_df = df[df["eval"] == eval_name]
        baseline = eval_df[eval_df["model"] == baseline_model]

        if baseline.empty:
            continue

        baseline_scores = baseline.iloc[0][traits].to_dict()

        for _, row in eval_df.iterrows():
            if row["model"] == baseline_model:
                continue

            deltas = {f"{trait}_delta": row[trait] - baseline_scores[trait] for trait in traits}
            results.append({
                "model": row["model"],
                "eval": eval_name,
                **deltas,
            })

    return pd.DataFrame(results) if results else pd.DataFrame()


# ============================================================================
# Personality Sweep Visualization Functions
# ============================================================================

BIG_FIVE_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


def parse_model_scale(model_name: str) -> float | None:
    """Extract LoRA scale from model directory name.

    Examples:
        'base' -> 0.0
        'lora_-2p00x' -> -2.0
        'lora_+1p50x' -> 1.5
    """
    if model_name == "base":
        return 0.0

    match = re.match(r"lora_([+-]?\d+)p(\d+)x", model_name)
    if match:
        integer_part = int(match.group(1))
        decimal_part = int(match.group(2))
        return integer_part + (decimal_part / 100.0) * (1 if integer_part >= 0 else -1)

    return None


def prepare_sweep_dataframe(df: pd.DataFrame) -> pd.DataFrame | None:
    """Convert results to sweep format with scale column."""
    # Try to parse scales from model names
    scales = []
    for model_name in df["model"]:
        scale = parse_model_scale(model_name)
        scales.append(scale)

    # Check if we have valid scales (personality sweep format)
    if all(s is None for s in scales):
        return None

    df = df.copy()
    df["scale"] = scales
    df = df[df["scale"].notna()].sort_values("scale")
    return df


def generate_mock_sweep_data() -> pd.DataFrame:
    """Generate mock personality sweep data for testing."""
    np.random.seed(42)
    scales = np.arange(-2.0, 2.25, 0.25)

    base_scores = {
        "Openness": 0.65,
        "Conscientiousness": 0.70,
        "Extraversion": 0.50,
        "Agreeableness": 0.60,
        "Neuroticism": 0.45,
    }

    trait_slopes = {
        "Openness": 0.08,
        "Conscientiousness": -0.03,
        "Extraversion": 0.05,
        "Agreeableness": -0.12,
        "Neuroticism": 0.06,
    }

    results = []
    for scale in scales:
        model_name = "base" if scale == 0.0 else f"lora_{scale:+.2f}x".replace(".", "p")

        scores = {}
        for trait in BIG_FIVE_TRAITS:
            base = base_scores[trait]
            slope = trait_slopes[trait]
            value = base + slope * scale + np.random.normal(0, 0.02)
            scores[trait] = np.clip(value, 0.0, 1.0)

        results.append({
            "model": model_name,
            "eval": "bfi",
            "scale": scale,
            **scores,
        })

    return pd.DataFrame(results)


def print_sweep_summary(df: pd.DataFrame) -> None:
    """Print formatted summary for personality sweep."""
    print("\n" + "=" * 100)
    print("PERSONALITY SWEEP: TRAIT SCORES ACROSS LORA SCALES")
    print("=" * 100)

    header = f"{'Scale':<10}" + "".join(f"{trait:<18}" for trait in BIG_FIVE_TRAITS)
    print(header)
    print("-" * 100)

    for _, row in df.iterrows():
        scale = row["scale"]
        marker = " ← BASELINE" if abs(scale) < 0.01 else ""
        values = "".join(f"{row[trait]:.4f}{'':>12}" for trait in BIG_FIVE_TRAITS)
        print(f"{scale:+7.2f}{'':>3}{values}{marker}")

    print("=" * 100)

    # Delta summary
    baseline = df[df["scale"].abs() < 0.01]
    if not baseline.empty:
        baseline_scores = baseline[BIG_FIVE_TRAITS].iloc[0].to_dict()

        print("\n" + "=" * 100)
        print("TRAIT CHANGES FROM BASELINE")
        print("=" * 100)

        min_row = df.loc[df["scale"].idxmin()]
        max_row = df.loc[df["scale"].idxmax()]

        for scale_row, label in [(min_row, "MIN"), (max_row, "MAX")]:
            scale = scale_row["scale"]
            print(f"\n{label} Scale (s={scale:+.2f}):")
            print("-" * 100)

            for trait in BIG_FIVE_TRAITS:
                baseline_val = baseline_scores[trait]
                current_val = scale_row[trait]
                delta = current_val - baseline_val
                pct_change = (delta / baseline_val * 100) if baseline_val != 0 else 0

                arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
                print(
                    f"  {trait:<20} {baseline_val:.4f} → {current_val:.4f}  "
                    f"({arrow} {delta:+.4f}, {pct_change:+.1f}%)"
                )

        print("\n" + "=" * 100)


def plot_visualizations(df: pd.DataFrame, output_dir: Path, title_suffix: str = "") -> None:
    """Generate all personality sweep visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("\nGenerating visualizations...")

    # Line plot
    _plot_line_chart(df, plots_dir / "personality_line.png", title_suffix)

    # Heatmap
    _plot_heatmap(df, plots_dir / "personality_heatmap.png", title_suffix)

    # Delta plot
    _plot_delta_chart(df, plots_dir / "personality_delta.png", title_suffix)

    # Radar chart
    _plot_radar_chart(df, plots_dir / "personality_radar.png", title_suffix)

    print(f"\n✅ All plots saved to {plots_dir}")


def _plot_line_chart(df: pd.DataFrame, output_path: Path, title_suffix: str) -> None:
    """Plot trait scores vs LoRA scale."""
    import matplotlib.pyplot as plt

    n_traits = len(BIG_FIVE_TRAITS)
    fig, axes = plt.subplots(n_traits, 1, figsize=(14, 3.5 * n_traits))
    if n_traits == 1:
        axes = [axes]

    colors = [
        ("#2196F3", "#1565C0"),
        ("#FF9800", "#E65100"),
        ("#4CAF50", "#2E7D32"),
        ("#9C27B0", "#6A1B9A"),
        ("#F44336", "#C62828"),
    ]

    scales = df["scale"].values
    mock_stdev = 0.05

    for idx, trait in enumerate(BIG_FIVE_TRAITS):
        ax = axes[idx]
        values = df[trait].values

        stdevs = np.array([mock_stdev * (1 + 0.3 * abs(s)) for s in scales])
        cis = 1.96 * stdevs / np.sqrt(max(len(df) / len(scales), 1))

        ci_color, line_color = colors[idx]

        ax.fill_between(scales, values - cis, values + cis,
                        alpha=0.2, color=ci_color, label="95% CI (mock)")
        ax.plot(scales, values, "o-", color=line_color, linewidth=2.5,
                markersize=7, label=f"{trait} score", zorder=5)

        ax.axvline(0, color="gray", linestyle="--", alpha=0.5, label="Baseline (s=0)")
        ax.axvline(1, color="green", linestyle=":", alpha=0.5, label="Standard LoRA (s=1)")

        ax.set_xlabel("LoRA Scaling Factor", fontsize=11)
        ax.set_ylabel(f"{trait} Score", fontsize=11)
        ax.set_title(f"{trait} vs. LoRA Scaling Factor", fontsize=13)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    plt.suptitle(f"Big Five Personality Traits{title_suffix}",
                fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Line plot: {output_path.name}")


def _plot_heatmap(df: pd.DataFrame, output_path: Path, title_suffix: str) -> None:
    """Plot heatmap of traits × scales."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    pivot = df.set_index("scale")[BIG_FIVE_TRAITS].T

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(BIG_FIVE_TRAITS)))
    ax.set_xticklabels([f"{s:+.2f}" for s in pivot.columns], fontsize=9)
    ax.set_yticklabels(BIG_FIVE_TRAITS, fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(BIG_FIVE_TRAITS)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            text_color = "white" if value < 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                   color=text_color, fontsize=8)

    baseline_idx = list(pivot.columns).index(0.0) if 0.0 in pivot.columns.values else None
    if baseline_idx is not None:
        ax.add_patch(Rectangle((baseline_idx - 0.5, -0.5), 1, len(BIG_FIVE_TRAITS),
                               fill=False, edgecolor="blue", linewidth=3))

    ax.set_xlabel("LoRA Scaling Factor", fontsize=12)
    ax.set_ylabel("Personality Trait", fontsize=12)
    ax.set_title(f"Personality Trait Heatmap{title_suffix}",
                fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Trait Score", rotation=270, labelpad=20, fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Heatmap: {output_path.name}")


def _plot_delta_chart(df: pd.DataFrame, output_path: Path, title_suffix: str) -> None:
    """Plot change from baseline."""
    import matplotlib.pyplot as plt

    baseline = df[df["scale"].abs() < 0.01]
    if baseline.empty:
        print("  ⚠️  No baseline found, skipping delta plot")
        return

    baseline_scores = baseline[BIG_FIVE_TRAITS].iloc[0].to_dict()

    deltas = df.copy()
    for trait in BIG_FIVE_TRAITS:
        deltas[f"{trait}_delta"] = df[trait] - baseline_scores[trait]

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]
    markers = ["o", "s", "^", "D", "v"]

    for idx, trait in enumerate(BIG_FIVE_TRAITS):
        ax.plot(deltas["scale"], deltas[f"{trait}_delta"],
                marker=markers[idx], color=colors[idx],
                linewidth=2, markersize=6, label=trait)

    ax.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.7, label="Baseline")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(1, color="green", linestyle=":", alpha=0.5)

    ax.set_xlabel("LoRA Scaling Factor", fontsize=12)
    ax.set_ylabel("Trait Score Change from Baseline", fontsize=12)
    ax.set_title(f"Personality Trait Deltas{title_suffix}",
                fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Delta plot: {output_path.name}")


def _plot_radar_chart(df: pd.DataFrame, output_path: Path, title_suffix: str) -> None:
    """Plot radar charts."""
    import matplotlib.pyplot as plt

    scales = df["scale"].values

    if len(scales) <= 5:
        selected_scales = scales
    else:
        indices = [0, len(scales) // 4, len(scales) // 2, 3 * len(scales) // 4, len(scales) - 1]
        selected_scales = [scales[i] for i in indices]

    n_traits = len(BIG_FIVE_TRAITS)
    angles = np.linspace(0, 2 * np.pi, n_traits, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    colors = ["#E53935", "#FB8C00", "#43A047", "#1E88E5", "#8E24AA"]

    for idx, scale in enumerate(selected_scales):
        row = df[df["scale"] == scale]
        if row.empty:
            continue

        values = row[BIG_FIVE_TRAITS].iloc[0].tolist()
        values += values[:1]

        label = f"s={scale:+.2f}" if abs(scale) > 0.01 else "Baseline"
        ax.plot(angles, values, "o-", linewidth=2, label=label,
                color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(BIG_FIVE_TRAITS, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, alpha=0.3)

    ax.set_title(f"Big Five Personality Profile{title_suffix}",
                fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  ✓ Radar chart: {output_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze personality evaluation results")
    parser.add_argument(
        "run_path",
        type=Path,
        nargs="?",
        help="Path to eval output directory or specific run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save results to CSV file",
    )
    parser.add_argument(
        "--baseline",
        default="base_llama31_8b",
        help="Baseline model for delta calculations (default: base_llama31_8b)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations (requires --extra dev for matplotlib)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for visualizations (default: run_path/analysis)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock personality sweep data for testing",
    )
    parser.add_argument(
        "--title-suffix",
        default="",
        help="Suffix to add to plot titles",
    )
    args = parser.parse_args()

    # Handle mock data mode
    if args.mock:
        print("Generating mock personality sweep data...")
        df = generate_mock_sweep_data()
        run_name = "mock_sweep"
        title_suffix = args.title_suffix or " — Mock Data"
    else:
        if not args.run_path:
            parser.error("run_path is required when not using --mock")

        # Determine run directory
        if args.run_path.name.startswith("202"):
            run_dir = args.run_path
        else:
            run_dir = find_latest_run(args.run_path)
            print(f"Using latest run: {run_dir.name}\n")

        # Collect results
        results = collect_results(run_dir)

        if not results:
            print("No results found!")
            return

        df = pd.DataFrame(results)
        run_name = run_dir.name
        title_suffix = args.title_suffix

    # Check if this is a personality sweep (has parseable scales)
    sweep_df = prepare_sweep_dataframe(df)
    is_sweep = sweep_df is not None

    if is_sweep:
        # Personality sweep format
        print(f"\n✅ Detected personality sweep with {len(sweep_df)} model configurations")
        print(f"   Scale range: {sweep_df['scale'].min():.2f} to {sweep_df['scale'].max():.2f}")
        print_sweep_summary(sweep_df)

        # Save CSV if requested
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            sweep_df.to_csv(args.output, index=False)
            print(f"\n✓ Results saved to {args.output}")

        # Generate visualizations if requested
        if args.visualize:
            output_dir = args.output_dir or (args.run_path / "analysis" if args.run_path else Path("analysis"))
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_visualizations(sweep_df, output_dir, title_suffix)
    else:
        # Standard personality eval format
        print(format_comparison_table(df))

        # Calculate and print deltas
        delta_df = calculate_deltas(df, args.baseline)
        if not delta_df.empty:
            print("\nTRAIT DELTAS (vs {})".format(args.baseline))
            print("=" * 80)
            print(delta_df.to_string(index=False))

        # Save to CSV if requested
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\n✓ Results saved to {args.output}")

            if not delta_df.empty:
                delta_output = args.output.parent / f"{args.output.stem}_deltas.csv"
                delta_df.to_csv(delta_output, index=False)
                print(f"✓ Deltas saved to {delta_output}")

        if args.visualize:
            print("\n⚠️  Visualizations are only supported for personality sweep runs")
            print("    (model directories with names like 'base', 'lora_-2p00x', etc.)")


if __name__ == "__main__":
    main()
