"""Appendix figure: O± per-method sweep trajectories (4-panel figure).

Mirror of fig_G_induction_method_sweeps.pdf but for openness, with both
directions on one figure to keep it compact:

  (a) O↑ LoRA scale sweep (coeffs 0.25, 0.50, 0.75, 1.00)
  (b) O↑ actcap fraction sweep (coeffs 0.25, 0.50, 0.75, 1.00)
  (c) O↓ LoRA scale sweep
  (d) O↓ actcap fraction sweep

Each panel shows per-turn openness (top half) plus per-turn coherence (we
use a single y-axis per panel — overlaid traces — to keep the 4-panel
figure readable; readers who want the coherence-vs-trait trade-off can
look at the dedicated O± Pareto figure).

Actually using the standard two-panel layout per direction (one figure per
method × direction would be 4 figures); we render TWO 2-panel figures,
one per direction:
  - fig_G_induction_o_method_sweeps_oplus.pdf
  - fig_G_induction_o_method_sweeps_ominus.pdf

Each two-panel figure has LoRA + actcap side by side.

Paper figures:
    - paper/figures/appendix/induction/fig_G_induction_o_method_sweeps_oplus.pdf
    - paper/figures/appendix/induction/fig_G_induction_o_method_sweeps_ominus.pdf

Run with:
    uv run python -m src_dev.visualisations.paper_appendix_o_method_sweeps
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
    "appendix/induction/fig_G_induction_o_method_sweeps_oplus_lora.pdf",
    "appendix/induction/fig_G_induction_o_method_sweeps_oplus_actcap.pdf",
    "appendix/induction/fig_G_induction_o_method_sweeps_ominus_lora.pdf",
    "appendix/induction/fig_G_induction_o_method_sweeps_ominus_actcap.pdf",
]

HF_FS = "datasets/persona-shattering-lasr/monorepo/fine_tuning/llama-3.1-8b-it/ocean/openness"

BASE_PATH = (
    f"{HF_FS}/amplifier/vanton4_paired_dpo/rollouts/"
    "rollout_baseline_t0.7_steering_o/base/baseline/evals/rollouts_evaluated.jsonl"
)

# Coefficient sweep colours (blue -> red, weak -> strong).
# O↓ direction has additional extended-sweep coefficients {1.50, 2.00, 3.00}
# from the ominus_extended_sweep run.
COEFF_COLORS = {
    "0.25": "#3c7fb1",
    "0.50": "#5b9bd5",
    "0.75": "#df6f4f",
    "1.00": "#c91546",
    "1.50": "#9b1042",
    "2.00": "#700b30",
    "3.00": "#400520",
}
COEFF_MARKERS = {"0.25": "s", "0.50": "^", "0.75": "D", "1.00": "v",
                 "1.50": "P", "2.00": "X", "3.00": "*"}
COEFF_STYLES = {
    "0.25": "--",
    "0.50": "-.",
    "0.75": ":",
    "1.00": (0, (3, 1, 1, 1)),
    "1.50": (0, (5, 1)),
    "2.00": (0, (1, 1)),
    "3.00": (0, (5, 1, 1, 1, 1, 1)),
}

JUDGES: list[tuple[str, str, tuple[float, float]]] = [
    ("openness_v2", "Openness score", (-4, 4)),
    ("coherence_v2", "Coherence score", (0, 10)),
]

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


def _load(path: str) -> list[dict[str, Any]]:
    fs = HfFileSystem()
    text = fs.cat(path).decode()
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def _per_turn(entries: list[dict[str, Any]], judge: str) -> dict[int, list[float]]:
    by_turn: dict[int, list[float]] = defaultdict(list)
    for e in entries:
        for r, msgs in e.get("messages", {}).items():
            for m in msgs:
                if m.get("role") != "assistant":
                    continue
                t = m.get("turn_index")
                obj = (m.get("scores") or {}).get(judge, {})
                v = obj.get("score") if isinstance(obj, dict) else obj
                if t is not None and v is not None:
                    by_turn[int(t)].append(float(v))
    return by_turn


def _bootstrap(values: list[float], seed: int, n_iter: int = BOOTSTRAP_N) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = mean(values)
    if n == 1:
        return m, m, m
    rng = random.Random(seed)
    boots = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_iter))
    return m, boots[int(0.025 * n_iter)], boots[int(0.975 * n_iter)]


def _aggregate(by_turn: dict[int, list[float]]) -> dict[int, dict[str, float]]:
    out = {}
    for t, vs in sorted(by_turn.items()):
        m, lo, hi = _bootstrap(vs, BOOTSTRAP_SEED + t)
        out[t] = {"mean": m, "lo": lo, "hi": hi}
    return out


def _build_cells(direction: str, method: str) -> list[tuple[str, str, str, str, str]]:
    """direction: 'amplifier'|'suppressor'; method: 'lora'|'actcap'."""
    base = f"{HF_FS}/{direction}/vanton4_paired_dpo/rollouts"
    if method == "lora":
        sub = "rollout_sweep_lora_t0.7_steering_o"
        coeff_dir = "scale_+{}"
    else:
        sub = "rollout_sweep_activation_capping_t0.7_steering_o"
        coeff_dir = "frac_{}"
    cells: list[tuple[str, str, str, str, str]] = [
        ("Base", BASE_PATH, "#000000", "-", "o"),
    ]
    # O↓ has extended sweep cells beyond coeff=1.0; O↑ does not.
    coeffs = ["0.25", "0.50", "0.75", "1.00"]
    if direction == "suppressor":
        coeffs.extend(["1.50", "2.00", "3.00"])
    for coeff in coeffs:
        cells.append((
            f"coeff={coeff}",
            f"{base}/{sub}/{coeff_dir.format(coeff)}/baseline/evals/rollouts_evaluated.jsonl",
            COEFF_COLORS[coeff],
            COEFF_STYLES[coeff],
            COEFF_MARKERS[coeff],
        ))
    return cells


def _render(cells: list[tuple[str, str, str, str, str]], title: str, out_name: str) -> None:
    print(f"\n=== {title} ===")
    cell_data = []
    for label, path, colour, linestyle, marker in cells:
        print(f"  {label}")
        try:
            entries = _load(path)
        except Exception as e:
            print(f"    ERR: {e.__class__.__name__}")
            continue
        cell_data.append((label, entries, colour, linestyle, marker))

    n_judges = len(JUDGES)
    fig, axes = plt.subplots(n_judges, 1, figsize=(8.0, 3.5 * n_judges), sharex=True)
    if n_judges == 1:
        axes = [axes]

    for ax, (judge, ylabel, ylim) in zip(axes, JUDGES):
        for label, entries, colour, linestyle, marker in cell_data:
            agg = _aggregate(_per_turn(entries, judge))
            if not agg:
                continue
            ts = sorted(agg.keys())
            ms = [agg[t]["mean"] for t in ts]
            lo = [agg[t]["lo"] for t in ts]
            hi = [agg[t]["hi"] for t in ts]
            yerr_lo = [m - l for m, l in zip(ms, lo)]
            yerr_hi = [h - m for m, h in zip(ms, hi)]
            ax.errorbar(
                ts, ms, yerr=[yerr_lo, yerr_hi],
                color=colour, linestyle=linestyle, marker=marker,
                linewidth=2, markersize=5, label=label,
                capsize=3, capthick=1.0, elinewidth=1.0,
            )
        if ylim is not None and ylim[0] < 0 < ylim[1]:
            ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best", ncol=2)

    axes[0].set_title(title, fontsize=12, loc="left", pad=8)
    axes[-1].set_xlabel("Turn index", fontsize=11)
    fig.tight_layout()

    out_pdf = PAPER_FIGURES_DIR / "appendix" / "induction" / out_name
    out_png = out_pdf.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_pdf}")


def main() -> None:
    # O↑ direction — LoRA + actcap as a 2-figure pair
    _render(
        _build_cells("amplifier", "lora"),
        "Inducing O↑ persona using LoRA at different coefficients",
        "fig_G_induction_o_method_sweeps_oplus_lora.pdf",
    )
    _render(
        _build_cells("amplifier", "actcap"),
        "Inducing O↑ persona using activation capping at different coefficients",
        "fig_G_induction_o_method_sweeps_oplus_actcap.pdf",
    )
    # O↓ direction
    _render(
        _build_cells("suppressor", "lora"),
        "Inducing O↓ persona using LoRA at different coefficients",
        "fig_G_induction_o_method_sweeps_ominus_lora.pdf",
    )
    _render(
        _build_cells("suppressor", "actcap"),
        "Inducing O↓ persona using activation capping at different coefficients",
        "fig_G_induction_o_method_sweeps_ominus_actcap.pdf",
    )


if __name__ == "__main__":
    main()
