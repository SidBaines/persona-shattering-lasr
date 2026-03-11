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


def compute_correlation_table(personas: list[str]) -> str | None:
    """Compute Pearson r between LoRA scale and TRAIT scores for each persona.

    Returns:
        Markdown table string, or None if no data is available.
    """
    from scipy.stats import pearsonr as _pearsonr
    from scripts.evals.personality.analyze_results import BIG_FIVE, load_sweep_data

    rows: list[dict[str, str]] = []
    for persona in personas:
        run_dir = SCRATCH_ROOT / f"eval_hf-personas_{persona}"
        if not run_dir.exists():
            continue
        data = load_sweep_data(run_dir)
        df = data.get("trait")
        if df is None or len(df) < 3:
            rows.append({"Persona": persona, **{t[0]: "—" for t in BIG_FIVE}})
            continue

        row: dict[str, str] = {"Persona": persona}
        for trait in BIG_FIVE:
            if trait not in df.columns:
                row[trait[0]] = "—"
                continue
            scales = df["scale"].values.astype(float)
            scores = df[trait].values.astype(float)
            mask = np.isfinite(scales) & np.isfinite(scores)
            if mask.sum() < 3:
                row[trait[0]] = "—"
                continue
            r, _ = _pearsonr(scales[mask], scores[mask])
            row[trait[0]] = f"{r:+.2f}"
        rows.append(row)

    if not rows:
        return None

    # Build markdown table
    headers = ["Persona", "O", "C", "E", "A", "N"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(h, "—") for h in headers) + " |")
    return "\n".join(lines)


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

    # Correlation summary table
    table_md = compute_correlation_table(personas)
    if table_md:
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


if __name__ == "__main__":
    main()
