#!/usr/bin/env python3
"""
Generate TRAIT sweep plots for all evaluated HuggingFace persona adapters.

Reads eval results from scratch/evals/personality/eval_hf-personas_<persona>/
and writes one PNG per persona to scripts/experiments/personality_evals/figures/.

Usage:
    # All personas with results:
    uv run python -m scripts.experiments.personality_evals.plot_hf_personas

    # Specific personas:
    uv run python -m scripts.experiments.personality_evals.plot_hf_personas \
        --personas sarcasm humor remorse
"""

import argparse
from pathlib import Path

import numpy as np

FIGURES_DIR = Path(__file__).parent / "figures"
SCRATCH_ROOT = Path("scratch/evals/personality")

# Prior expectations for each persona's OCEAN trait direction.
# Values: "↑ strong", "↑ moderate", "↑ mild", "~ stable", "↓ mild", "↓ moderate", "↓ strong"
EXPECTATIONS: dict[str, dict[str, str]] = {
    "sarcasm": {
        "O": "↑ mild", "C": "↓ mild", "E": "↑ moderate", "A": "↓ strong", "N": "↑ moderate",
        "logic": "Sarcasm requires detachment + edge; likely degrades A cleanly, pulls N up as register sharpens",
    },
    "humor": {
        "O": "↑ moderate", "C": "↓ mild", "E": "↑ strong", "A": "↑ mild", "N": "↓ mild",
        "logic": "Humor is socially warm + expansive; E should scale most cleanly",
    },
    "remorse": {
        "O": "~ stable", "C": "↓ mild", "E": "↓ moderate", "A": "↑ mild", "N": "↑ strong",
        "logic": "N scales cleanly with remorse; internal focus, self-referential distress signal",
    },
    "impulsiveness": {
        "O": "↑ moderate", "C": "↓ strong", "E": "↑ strong", "A": "↓ mild", "N": "↑ moderate",
        "logic": "Impulsiveness should degrade C most cleanly; E and N co-rise (entangled)",
    },
    "nonchalance": {
        "O": "↑ mild", "C": "↓ moderate", "E": "↓ moderate", "A": "~ stable", "N": "↓ strong",
        "logic": "Suppresses arousal/urgency; N should drop cleanly, but may entangle with E reduction",
    },
    "sycophancy": {
        "O": "↓ mild", "C": "↑ mild", "E": "↑ mild", "A": "↑ strong", "N": "↓ mild",
        "logic": "A should dominate and scale cleanly; risk: conflated with high-C compliance",
    },
    "poeticism": {
        "O": "↑ strong", "C": "↓ moderate", "E": "↓ mild", "A": "↑ mild", "N": "↑ mild",
        "logic": "O should be cleanest signal here — metaphorical richness, aesthetic sensitivity",
    },
    "mathematical": {
        "O": "↓ mild", "C": "↑ strong", "E": "~ stable", "A": "~ stable", "N": "↓ mild",
        "logic": "C scales cleanly; E remains stable (no social valence); cleanest persona overall",
    },
    "misalignment": {
        "O": "↑ mild", "C": "↓ strong", "E": "↑ mild", "A": "↓ strong", "N": "↑ strong",
        "logic": "Multi-trait entanglement expected; no clean OCEAN dimension — diagnostic chaos case",
    },
    "goodness": {
        "O": "↑ mild", "C": "↑ moderate", "E": "↑ mild", "A": "↑ strong", "N": "↓ moderate",
        "logic": "A + C co-rise likely; risk of evaluator conflating 'helpful tone' with A",
    },
    "loving": {
        "O": "↑ strong", "C": "~ stable", "E": "↑ moderate", "A": "↑ strong", "N": "↓ mild",
        "logic": "A should dominate; O co-rises with emotional expressiveness",
    },
}

ALL_PERSONAS = [
    "sarcasm",
    "humor",
    "remorse",
    "impulsiveness",
    "nonchalance",
    "sycophancy",
    "poeticism",
    "mathematical",
    "misalignment",  # standalone repo: maius/llama-3.1-8b-it-misalignment
    "goodness",
    "loving",
]


def plot_persona(persona: str) -> Path | None:
    """Generate and save a TRAIT sweep plot for one persona.

    Args:
        persona: Persona name matching a subfolder in maius/llama-3.1-8b-it-personas.

    Returns:
        Path to the saved PNG, or None if eval results are not found.
    """
    from scripts.evals.personality.analyze_results import generate_plots, load_sweep_data

    run_dir = SCRATCH_ROOT / f"eval_hf-personas_{persona}"
    if not run_dir.exists():
        print(f"  [{persona}] no results found at {run_dir} — skipping")
        return None

    data = load_sweep_data(run_dir)
    saved = generate_plots(data, FIGURES_DIR, title_suffix=f"HF persona: {persona}")
    if not saved:
        print(f"  [{persona}] no plottable data found — skipping")
        return None

    # Rename the generic trait_sweep.png to <persona>.png
    generic = FIGURES_DIR / "trait_sweep.png"
    dest = FIGURES_DIR / f"{persona}.png"
    if generic.exists():
        generic.rename(dest)
    elif not dest.exists() and saved:
        # generate_plots may have written directly to dest already
        pass

    print(f"  [{persona}] → {dest}")
    return dest


def compute_slopes_and_correlations(
    personas: list[str],
) -> tuple[list[str], list[str], np.ndarray, np.ndarray] | None:
    """Compute OLS slope and Pearson r of TRAIT score vs. LoRA scale for each persona.

    Returns:
        Tuple of (persona_names, trait_labels, slope_matrix, corr_matrix)
        each matrix shape (n_personas, 5) with NaN for missing data,
        or None if no data is available.
    """
    from scipy.stats import pearsonr as _pearsonr

    from scripts.evals.personality.analyze_results import BIG_FIVE, load_sweep_data

    trait_labels = [t[0] for t in BIG_FIVE]  # O, C, E, A, N
    persona_names: list[str] = []
    slope_rows: list[list[float]] = []
    corr_rows: list[list[float]] = []

    for persona in personas:
        run_dir = SCRATCH_ROOT / f"eval_hf-personas_{persona}"
        if not run_dir.exists():
            continue
        data = load_sweep_data(run_dir)
        df = data.get("trait")
        s_row: list[float] = []
        r_row: list[float] = []
        for trait in BIG_FIVE:
            if df is None or len(df) < 3 or trait not in df.columns:
                s_row.append(float("nan"))
                r_row.append(float("nan"))
                continue
            scales = df["scale"].values.astype(float)
            scores = df[trait].values.astype(float)
            mask = np.isfinite(scales) & np.isfinite(scores)
            if mask.sum() < 3:
                s_row.append(float("nan"))
                r_row.append(float("nan"))
                continue
            x, y = scales[mask], scores[mask]
            # OLS slope: beta = cov(x,y) / var(x)
            beta = np.cov(x, y, ddof=0)[0, 1] / np.var(x)
            r, _ = _pearsonr(x, y)
            s_row.append(beta)
            r_row.append(r)
        persona_names.append(persona)
        slope_rows.append(s_row)
        corr_rows.append(r_row)

    if not slope_rows:
        return None
    return persona_names, trait_labels, np.array(slope_rows), np.array(corr_rows)


def format_expectations_table(personas: list[str]) -> str:
    """Format the prior expectations as a markdown table."""
    headers = ["Persona", "O", "C", "E", "A", "N", "Prediction logic"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for persona in personas:
        exp = EXPECTATIONS.get(persona)
        if not exp:
            continue
        cells = [
            persona,
            exp["O"], exp["C"], exp["E"], exp["A"], exp["N"],
            exp["logic"],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_numeric_table(
    persona_names: list[str], trait_labels: list[str], matrix: np.ndarray, fmt: str = "+.3f"
) -> str:
    """Format a numeric matrix as a markdown table."""
    headers = ["Persona"] + trait_labels
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for i, persona in enumerate(persona_names):
        cells = [persona]
        for j in range(len(trait_labels)):
            v = matrix[i, j]
            cells.append("—" if np.isnan(v) else f"{v:{fmt}}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_slope_heatmap(
    persona_names: list[str],
    trait_labels: list[str],
    slope_matrix: np.ndarray,
    output_dir: Path,
) -> Path:
    """Plot the OLS slope matrix as an annotated heatmap."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    fig, ax = plt.subplots(figsize=(7, max(4, 0.55 * len(persona_names))))

    # Symmetric range around 0, sized to the data
    vmax = np.nanmax(np.abs(slope_matrix)) * 1.05
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(slope_matrix, cmap="coolwarm", norm=norm, aspect="auto")

    # OCEAN labels on top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticks(range(len(trait_labels)))
    ax.set_xticklabels(trait_labels, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(persona_names)))
    ax.set_yticklabels(persona_names, fontsize=11)

    # Annotate cells
    for i in range(len(persona_names)):
        for j in range(len(trait_labels)):
            v = slope_matrix[i, j]
            if np.isnan(v):
                text, color = "—", "gray"
            else:
                text = f"{v:+.3f}"
                color = "white" if abs(v) / vmax > 0.55 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=10, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("OLS slope (trait score change per 1x LoRA scale)", fontsize=10)

    ax.set_title(
        "TRAIT sensitivity to LoRA scaling\n"
        "OLS slope of Big Five score vs. adapter scale (−1.5x to +1.5x)\n"
        "Positive = trait increases with scale; negative = trait decreases",
        fontsize=11, fontweight="bold", pad=12,
    )
    plt.tight_layout()

    out = output_dir / "correlation_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--personas",
        nargs="+",
        default=None,
        metavar="PERSONA",
        help="Personas to plot (default: all with results in scratch/).",
    )
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")  # headless

    personas = args.personas or ALL_PERSONAS
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Writing plots to {FIGURES_DIR}\n")
    saved = []
    for persona in personas:
        path = plot_persona(persona)
        if path:
            saved.append(path)

    print(f"\nDone. {len(saved)}/{len(personas)} plots saved.")

    # Slope + correlation summary tables + heatmap
    result = compute_slopes_and_correlations(personas)
    if result:
        persona_names, trait_labels, slope_matrix, corr_matrix = result
        slope_md = format_numeric_table(persona_names, trait_labels, slope_matrix, "+.3f")
        corr_md = format_numeric_table(persona_names, trait_labels, corr_matrix, "+.2f")
        expectations_md = format_expectations_table(persona_names)

        print(f"\nTRAIT slope table (OLS β: trait score change per 1x LoRA scale):\n")
        print(slope_md)
        print(f"\nTRAIT correlation table (Pearson r):\n")
        print(corr_md)

        md_path = FIGURES_DIR / "correlation_table.md"
        md_content = (
            "# TRAIT sensitivity analysis\n\n"
            "Generated by `scripts.experiments.personality_evals.plot_hf_personas`\n"
            "from eval results in `scratch/evals/personality/eval_hf-personas_<persona>/`.\n\n"
            "All metrics computed across 7 scale points\n"
            "(−1.5, −1.0, −0.5, 0, +0.5, +1.0, +1.5), where scale 0 = base model (no adapter).\n\n"
            "## Measured slopes (OLS β)\n\n"
            "Change in trait score (on the 0–1 TRAIT scale) per +1.0x increase in LoRA\n"
            "adapter scaling. For example, β = +0.080 means the trait score rises by\n"
            "0.08 for every unit of LoRA scale applied.\n\n"
            f"{slope_md}\n\n"
            "## Pearson correlations (r)\n\n"
            "Linear correlation between LoRA scale and trait score. Measures how\n"
            "consistently the trait tracks scale (r = ±1 = perfectly linear,\n"
            "r ≈ 0 = no linear relationship).\n\n"
            f"{corr_md}\n\n"
            "## Prior expectations\n\n"
            "Directional predictions for how each persona adapter *should* move\n"
            "each Big Five trait, based on the trait's psychological content and\n"
            "the persona's behavioral definition.\n\n"
            f"{expectations_md}\n"
        )
        md_path.write_text(md_content)
        print(f"\nTable saved to {md_path}")

        plot_slope_heatmap(persona_names, trait_labels, slope_matrix, FIGURES_DIR)


if __name__ == "__main__":
    main()
