"""Plot per-turn frustration curves for BASE / CONTROL / N- / inverse-N-.

Pulls ``summary.json`` from the shared HF monorepo under
``evals/frustration_eval/<run_name>/<category>/summary.json`` (falls back to
``<run_name>/summary.json`` if the summary sits at the run root), then saves a
two-panel figure to ``paper/figures/main/``.

Usage:
    uv run python -m src_dev.visualisations.plot_frustration_four_way
    uv run python -m src_dev.visualisations.plot_frustration_four_way \
        --n-prompts 20  # pick the matching run-name set
"""

from __future__ import annotations

import argparse
import json
import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

from scripts_dev.frustration_eval.prompts import IMPOSSIBLE_NUMERIC_PUZZLES
from src_dev.visualisations import PAPER_FIGURES_DIR

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "persona-shattering-lasr/monorepo"
REPO_PREFIX = "evals/frustration_eval"
CATEGORY = "impossible_numeric_3turn"


@dataclass(frozen=True)
class RunSpec:
    label: str
    run_name: str
    color: str
    marker: str


# Matching run-name sets for each sample size. Ordering here determines plot order.
RUN_SETS: dict[int, list[RunSpec]] = {
    20: [
        RunSpec("BASE",          "gemma3_27b_base_or_8turn_20prompt_1rollout",                                              "#2F5D9F", "o-"),
        RunSpec("CONTROL",       "gemma3_27b_control_use_diff_words_v1_persona_hfbatched_8turn_20prompt_1rollout",         "#6B6B6B", "D-"),
        RunSpec("N-",            "gemma3_27b_n_minus_v4_persona_hfbatched_8turn_20prompt_1rollout",                        "#C73E3A", "s-"),
        RunSpec("N- inverted",   "gemma3_27b_n_minus_v4_persona_negscale_hfbatched_8turn_20prompt_1rollout",               "#7A1F1B", "^-"),
    ],
    100: [
        RunSpec("BASE",          "gemma3_27b_base_or_8turn_200_samples_1rollout",                                          "#2F5D9F", "o-"),
        RunSpec("CONTROL",       "gemma3_27b_control_use_diff_words_v1_persona_hfbatched_8turn_100_samples_1rollout",     "#6B6B6B", "D-"),
        RunSpec("N-",            "gemma3_27b_n_minus_v4_persona_hfbatched_8turn_100_samples_1rollout",                    "#C73E3A", "s-"),
        RunSpec("N- inverted",   "gemma3_27b_n_minus_v4_persona_negscale_hfbatched_8turn_100_samples_1rollout",           "#7A1F1B", "^-"),
    ],
}


def fetch_summary(run_name: str, *, repo_id: str = DEFAULT_REPO_ID) -> dict:
    """Download ``summary.json`` for a run from the HF monorepo, preferring
    the per-category summary when present."""
    candidates = [
        f"{REPO_PREFIX}/{run_name}/{CATEGORY}/summary.json",
        f"{REPO_PREFIX}/{run_name}/summary.json",
    ]
    last_err: Exception | None = None
    for path_in_repo in candidates:
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=path_in_repo,
                repo_type="dataset",
            )
            data = json.loads(Path(local_path).read_text())
            if CATEGORY in data:
                data = data[CATEGORY]
            return data
        except Exception as e:  # huggingface_hub raises various subtypes
            last_err = e
    raise RuntimeError(f"Could not fetch summary for {run_name!r}: {last_err}")


def plot_four_way(
    specs: list[RunSpec],
    *,
    n_prompts: int,
    out_dir: Path,
) -> tuple[Path, Path]:
    summaries = [(s, fetch_summary(s.run_name)) for s in specs]
    turns = list(range(1, len(summaries[0][1]["per_turn_mean"]) + 1))

    fig, (ax_mean, ax_pct) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    for spec, data in summaries:
        ax_mean.plot(
            turns, data["per_turn_mean"], spec.marker,
            label=spec.label,
            color=spec.color, linewidth=1.8, markersize=6,
        )
        ax_pct.plot(
            turns, data["per_turn_pct_high"], spec.marker,
            label=spec.label,
            color=spec.color, linewidth=1.8, markersize=6,
        )

    ax_mean.set_xlabel("Turn")
    ax_mean.set_ylabel("Mean frustration (judge 0–10)")
    ax_mean.set_title("Per-turn mean frustration")
    ax_mean.set_xticks(turns)
    ax_mean.set_ylim(0, 10)
    ax_mean.grid(alpha=0.3)

    ax_pct.set_xlabel("Turn")
    ax_pct.set_ylabel("% high frustration (score ≥ 6)")
    ax_pct.set_title("Per-turn % high frustration")
    ax_pct.set_xticks(turns)
    ax_pct.set_ylim(-2, 102)
    ax_pct.grid(alpha=0.3)

    # One shared legend placed outside the right panel.
    handles, _ = ax_mean.get_legend_handles_labels()
    fig.legend(
        handles, [s.label for s in specs],
        loc="center left",
        bbox_to_anchor=(0.91, 0.55),
        fontsize=8,
        frameon=False,
        borderaxespad=0.0,
    )

    fig.suptitle(
        f"Frustration eval — gemma-3-27b-it (n={n_prompts} prompts, 8 turns, 1 rollout)",
        fontsize=10,
    )

    # Example puzzle — one representative prompt, wrapped, placed under the axes.
    example = " ".join(IMPOSSIBLE_NUMERIC_PUZZLES[0].split())
    wrapped = textwrap.fill(f"Example puzzle:  {example}", width=120)
    fig.text(
        0.5, 0.02, wrapped,
        ha="center", va="bottom",
        fontsize=7.5, color="#333",
        wrap=True,
    )

    # Reserve ~10% right for legend, ~12% bottom for example puzzle.
    fig.tight_layout(rect=[0, 0.12, 0.90, 0.95])

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fig_frustration_eval_4way_n{n_prompts}"
    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-prompts", type=int, choices=sorted(RUN_SETS.keys()), default=20,
        help="Select the matching run-name set (20 or 100).",
    )
    parser.add_argument(
        "--out-dir", default=str(PAPER_FIGURES_DIR / "main"),
        help="Directory to write the figure PDF/PNG.",
    )
    args = parser.parse_args()

    specs = RUN_SETS[args.n_prompts]
    out_dir = Path(args.out_dir)
    out_png, out_pdf = plot_four_way(specs, n_prompts=args.n_prompts, out_dir=out_dir)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
