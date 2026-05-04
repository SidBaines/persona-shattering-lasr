"""Headline plot for the E+ induction comparison section (Sec. 3.4).

Per-turn trajectory of extraversion and coherence for four induction methods
on neutral psychometric prompts:

  1. base                — no intervention
  2. sysprompt-induce-E+ — canonical OCEAN_DEFINITION as assistant system prompt
  3. E+ LoRA scale 0.75
  4. actcap fraction 0.85

Two-panel stacked figure (extraversion on top, coherence on bottom). Bootstrap
95% CIs as error bars. Same 10 neutral psychometric prompts across all cells
(seeded), 2 rollouts per prompt, 15 turns per rollout.

Data sources (all under HF ``persona-shattering-lasr/monorepo``):
  base:        fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/vanton4_paired_dpo/
                 rollouts/rollout_baseline_t0.7_steering/base/baseline/evals/rollouts_evaluated.jsonl
  sysprompt:   fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/vanton4_paired_dpo/
                 rollouts/rollout_sysprompt_elicit_t0.7_steering/base/
                 sysprompt_elicit_extraversion_high/evals/rollouts_evaluated.jsonl
  LoRA 0.75:   fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/vanton4_paired_dpo/
                 rollouts/rollout_sweep_lora_t0.7_steering/scale_+0.75/baseline/evals/rollouts_evaluated.jsonl
  actcap 0.85: fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/vanton4_paired_dpo/
                 rollouts/rollout_sweep_activation_capping_t0.7_steering/frac_0.85/baseline/evals/rollouts_evaluated.jsonl

When the 40x3 rerun lands (output suffix ``_t0.7_main``), swap the four paths
to point at it; everything else stays the same.

Paper figures:
    - paper/figures/main/fig_3_4_eplus_induction_comparison.pdf

Run with:
    uv run python -m src_dev.visualisations.paper_main_eplus_induction
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from huggingface_hub import HfFileSystem  # noqa: E402

from src_dev.visualisations import PAPER_FIGURES_DIR  # noqa: E402

PAPER_FIGURES = [
    "main/fig_3_4_eplus_induction_comparison.pdf",
]

HF_REPO = "persona-shattering-lasr/monorepo"
_BASE_HF = (
    "datasets/persona-shattering-lasr/monorepo/"
    "fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/vanton4_paired_dpo/rollouts"
)


# Cells in render order. Each tuple: (label, hf_path_relative_to_BASE_HF, color, linestyle, marker).
# Colours follow the project's vibe (semantic, not OCEAN-trait colors since this
# is a method comparison, not a trait comparison):
#   black     — base (no intervention)
#   green     — sysprompt (instruction-level)
#   red       — LoRA (weight-level)
#   blue      — activation capping (activation-level)
CELLS: list[tuple[str, str, str, str, str]] = [
    (
        "Base (no intervention)",
        "rollout_baseline_t0.7_steering/base/baseline/evals/rollouts_evaluated.jsonl",
        "#000000",
        "-",
        "o",
    ),
    (
        "Sysprompt-induce E↑",
        "rollout_sysprompt_elicit_t0.7_steering/base/sysprompt_elicit_extraversion_high/evals/rollouts_evaluated.jsonl",
        "#0f7f3f",
        "--",
        "s",
    ),
    (
        "E↑ LoRA (coeff=0.75)",
        "rollout_sweep_lora_t0.7_steering/scale_+0.75/baseline/evals/rollouts_evaluated.jsonl",
        "#c91546",
        "-.",
        "^",
    ),
    (
        "E↑ activation capping (coeff=0.85)",
        "rollout_sweep_activation_capping_t0.7_steering/frac_0.85/baseline/evals/rollouts_evaluated.jsonl",
        "#3c7fb1",
        ":",
        "D",
    ),
]

JUDGES: list[tuple[str, str, tuple[float, float]]] = [
    ("extraversion_v2", "Extraversion score", (-4, 4)),
    ("coherence_v2", "Coherence score", (0, 10)),
]

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


def _load_evaluated(hf_path: str) -> list[dict[str, Any]]:
    fs = HfFileSystem()
    text = fs.cat(hf_path).decode()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _per_turn_scores(
    entries: list[dict[str, Any]], judge: str
) -> dict[int, list[float]]:
    by_turn: dict[int, list[float]] = defaultdict(list)
    for entry in entries:
        for _r_idx, msgs in entry.get("messages", {}).items():
            for msg in msgs:
                if msg.get("role") != "assistant":
                    continue
                turn = msg.get("turn_index")
                if turn is None:
                    continue
                scores = msg.get("scores") or {}
                score_obj = scores.get(judge)
                raw = (
                    score_obj.get("score")
                    if isinstance(score_obj, dict)
                    else score_obj
                )
                if raw is None:
                    continue
                try:
                    by_turn[int(turn)].append(float(raw))
                except (TypeError, ValueError):
                    continue
    return by_turn


def _bootstrap_ci(
    values: list[float], n_iter: int, seed: int
) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) at 95%."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = mean(values)
    if n == 1:
        return m, m, m
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_iter):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(mean(sample))
    boots.sort()
    lo = boots[int(0.025 * n_iter)]
    hi = boots[int(0.975 * n_iter)]
    return m, lo, hi


def _aggregate(
    by_turn: dict[int, list[float]]
) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for turn, vs in sorted(by_turn.items()):
        m, lo, hi = _bootstrap_ci(vs, BOOTSTRAP_N, BOOTSTRAP_SEED + turn)
        out[turn] = {"mean": m, "lo": lo, "hi": hi, "n": len(vs)}
    return out


def main() -> None:
    print("Loading cells from HF...")
    cell_data: list[tuple[str, list[dict[str, Any]], str, str, str]] = []
    for label, sub, colour, linestyle, marker in CELLS:
        path = f"{_BASE_HF}/{sub}"
        print(f"  {label}: {sub}")
        entries = _load_evaluated(path)
        n_msgs = sum(
            len(msgs)
            for entry in entries
            for msgs in entry.get("messages", {}).values()
        )
        print(f"    {len(entries)} entries, {n_msgs} messages")
        cell_data.append((label, entries, colour, linestyle, marker))

    n_judges = len(JUDGES)
    fig, axes = plt.subplots(n_judges, 1, figsize=(8.0, 3.5 * n_judges), sharex=True)
    if n_judges == 1:
        axes = [axes]

    for ax, (judge, ylabel, ylim) in zip(axes, JUDGES):
        for label, entries, colour, linestyle, marker in cell_data:
            by_turn = _per_turn_scores(entries, judge)
            agg = _aggregate(by_turn)
            if not agg:
                continue
            turns = sorted(agg.keys())
            means = [agg[t]["mean"] for t in turns]
            lo = [agg[t]["lo"] for t in turns]
            hi = [agg[t]["hi"] for t in turns]
            yerr_lo = [m - l for m, l in zip(means, lo)]
            yerr_hi = [h - m for m, h in zip(means, hi)]
            ax.errorbar(
                turns,
                means,
                yerr=[yerr_lo, yerr_hi],
                color=colour,
                linestyle=linestyle,
                marker=marker,
                linewidth=2,
                markersize=6,
                label=label,
                capsize=4,
                capthick=1.2,
                elinewidth=1.2,
            )
        if ylim is not None and ylim[0] < 0 < ylim[1]:
            ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    axes[0].set_title(
        "Inducing E↑ persona: per-turn dynamics across induction methods",
        fontsize=12, loc="left", pad=8,
    )
    axes[-1].set_xlabel("Turn index", fontsize=11)
    fig.tight_layout()

    out_pdf = PAPER_FIGURES_DIR / "main" / "fig_3_4_eplus_induction_comparison.pdf"
    out_png = out_pdf.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
