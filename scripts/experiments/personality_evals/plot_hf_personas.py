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
    # "misalignment" is listed in the paper but was never uploaded to the HF repo.
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


if __name__ == "__main__":
    main()
