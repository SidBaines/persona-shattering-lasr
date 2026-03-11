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


def compute_correlations(personas: list[str]) -> tuple[list[str], list[str], np.ndarray] | None:
    """Compute Pearson r between LoRA scale and TRAIT scores for each persona.

    Returns:
        Tuple of (persona_names, trait_labels, correlation_matrix) where the
        matrix is shape (n_personas, 5) with NaN for missing data, or None
        if no data is available.
    """
    from scipy.stats import pearsonr as _pearsonr
    from scripts.evals.personality.analyze_results import BIG_FIVE, load_sweep_data

    trait_labels = [t[0] for t in BIG_FIVE]  # O, C, E, A, N
    persona_names: list[str] = []
    corr_rows: list[list[float]] = []

    for persona in personas:
        run_dir = SCRATCH_ROOT / f"eval_hf-personas_{persona}"
        if not run_dir.exists():
            continue
        data = load_sweep_data(run_dir)
        df = data.get("trait")
        row = []
        for trait in BIG_FIVE:
            if df is None or len(df) < 3 or trait not in df.columns:
                row.append(float("nan"))
                continue
            scales = df["scale"].values.astype(float)
            scores = df[trait].values.astype(float)
            mask = np.isfinite(scales) & np.isfinite(scores)
            if mask.sum() < 3:
                row.append(float("nan"))
                continue
            r, _ = _pearsonr(scales[mask], scores[mask])
            row.append(r)
        persona_names.append(persona)
        corr_rows.append(row)

    if not corr_rows:
        return None
    return persona_names, trait_labels, np.array(corr_rows)


def format_correlation_table(
    persona_names: list[str], trait_labels: list[str], corr_matrix: np.ndarray
) -> str:
    """Format correlation matrix as a markdown table."""
    headers = ["Persona"] + trait_labels
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for i, persona in enumerate(persona_names):
        cells = [persona]
        for j in range(len(trait_labels)):
            v = corr_matrix[i, j]
            cells.append("—" if np.isnan(v) else f"{v:+.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_correlation_heatmap(
    persona_names: list[str],
    trait_labels: list[str],
    corr_matrix: np.ndarray,
    output_dir: Path,
) -> Path:
    """Plot the correlation matrix as a heatmap and save as PNG."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, max(4, 0.55 * len(persona_names))))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(trait_labels)))
    ax.set_xticklabels(trait_labels, fontsize=11)
    ax.set_yticks(range(len(persona_names)))
    ax.set_yticklabels(persona_names, fontsize=11)

    # Annotate cells with correlation values
    for i in range(len(persona_names)):
        for j in range(len(trait_labels)):
            v = corr_matrix[i, j]
            if np.isnan(v):
                text = "—"
                color = "gray"
            else:
                text = f"{v:+.2f}"
                color = "white" if abs(v) > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=10, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson r", fontsize=11)
    ax.set_title("TRAIT correlation: LoRA scale vs. Big Five", fontsize=13, fontweight="bold")
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

    # Correlation summary table + heatmap
    result = compute_correlations(personas)
    if result:
        persona_names, trait_labels, corr_matrix = result
        table_md = format_correlation_table(persona_names, trait_labels, corr_matrix)

        print(f"\nTRAIT correlation table (Pearson r: LoRA scale vs. trait score):\n")
        print(table_md)

        md_path = FIGURES_DIR / "correlation_table.md"
        md_content = (
            "# TRAIT correlation table\n\n"
            "Pearson *r* between LoRA scale (−1.5 → +1.5) and TRAIT personality score,\n"
            "computed per-sample across 7 scale points (−1.5, −1.0, −0.5, 0, +0.5, +1.0, +1.5).\n\n"
            "Generated by `scripts.experiments.personality_evals.plot_hf_personas`\n"
            "from eval results in `scratch/evals/personality/eval_hf-personas_<persona>/`.\n\n"
            f"{table_md}\n"
        )
        md_path.write_text(md_content)
        print(f"\nTable saved to {md_path}")

        plot_correlation_heatmap(persona_names, trait_labels, corr_matrix, FIGURES_DIR)


if __name__ == "__main__":
    main()
