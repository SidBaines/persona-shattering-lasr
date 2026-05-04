"""Appendix figure: E↓ induction comparison (mirrors the main-body E↑ headline).

Two figures:

  1. fig_G_induction_eminus_floor.pdf — headline-style 4-line per-turn
     trajectory for E↓ induction. Mirrors fig_3_4_eplus_induction_comparison.pdf
     for the suppressor direction. Methods:
       - Base (no intervention)
       - Sysprompt-induce E↓
       - E↓ LoRA at coeff=0.75 (same coeff as headline E↑ contender; if it floors
         here, picking a different coeff won't help — see appendix sweep figure)
       - E↓ activation capping at coeff=0.75 (chosen to mirror E↑ side)

  2. fig_G_induction_eminus_distribution.pdf — supplementary 2-panel showing
     mean-vs-strength curves and per-score histograms. Makes the "floor" mechanism
     visible: sysprompt shifts mass cleanly into deep introversion (-2/-3 modes)
     while LoRA/actcap leave most mass at 0/-2.

Paper figures:
    - paper/figures/appendix/fig_G_induction_eminus_floor.pdf
    - paper/figures/appendix/fig_G_induction_eminus_distribution.pdf

Run with:
    uv run python -m src_dev.visualisations.paper_appendix_induction_eminus_floor
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
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
    "appendix/fig_G_induction_eminus_floor.pdf",
    "appendix/fig_G_induction_eminus_distribution.pdf",
]

HF_FS = "datasets/persona-shattering-lasr/monorepo/fine_tuning/llama-3.1-8b-it/ocean/extraversion"
SUPP = f"{HF_FS}/suppressor/vanton4_paired_dpo/rollouts"
AMP = f"{HF_FS}/amplifier/vanton4_paired_dpo/rollouts"

# ── 4-line headline-style trajectory ─────────────────────────────────────
# Same colour scheme as the E↑ headline so eyes can compare across figures:
#   black = base, green = sysprompt, red = LoRA, blue = actcap
HEADLINE_CELLS: list[tuple[str, str, str, str, str]] = [
    (
        "Base (no intervention)",
        f"{AMP}/rollout_baseline_t0.7_steering/base/baseline/evals/rollouts_evaluated.jsonl",
        "#000000", "-", "o",
    ),
    (
        "Sysprompt-induce E↓",
        f"{SUPP}/rollout_sysprompt_elicit_t0.7_steering_eminus/base/sysprompt_elicit_extraversion_low/evals/rollouts_evaluated.jsonl",
        "#0f7f3f", "--", "s",
    ),
    (
        "E↓ LoRA (coeff=0.75)",
        f"{SUPP}/rollout_sweep_lora_t0.7_steering_eminus/scale_+0.75/baseline/evals/rollouts_evaluated.jsonl",
        "#c91546", "-.", "^",
    ),
    (
        "E↓ activation capping (coeff=0.75)",
        f"{SUPP}/rollout_sweep_activation_capping_t0.7_steering_eminus/frac_0.75/baseline/evals/rollouts_evaluated.jsonl",
        "#3c7fb1", ":", "D",
    ),
]

# ── Distribution figure (panels (a) and (b) of the original) ──────────────
EMINUS_LORA_PATHS = {
    s: f"{SUPP}/rollout_sweep_lora_t0.7_steering_eminus/scale_+{s}/baseline/run_info.json"
    for s in ["0.25", "0.50", "0.75", "1.00", "1.50", "2.00", "3.00"]
}
EMINUS_ACTCAP_PATHS = {
    s: f"{SUPP}/rollout_sweep_activation_capping_t0.7_steering_eminus/frac_{s}/baseline/run_info.json"
    for s in ["0.25", "0.50", "0.75", "1.00"]
}
EPLUS_NEG_PATHS = {
    s: f"{AMP}/rollout_sweep_lora_t0.7_steering/scale_{s}/baseline/run_info.json"
    for s in ["-0.25", "-0.50", "-0.75", "-1.00"]
}
SYSP_LOW_PATH = (
    f"{SUPP}/rollout_sysprompt_elicit_t0.7_steering_eminus/base/"
    "sysprompt_elicit_extraversion_low/run_info.json"
)
BASE_PATH = f"{AMP}/rollout_baseline_t0.7_steering/base/baseline/run_info.json"

DIST_CELLS: list[tuple[str, str, str]] = [
    ("Base (no intervention)",
     f"{AMP}/rollout_baseline_t0.7_steering/base/baseline/evals/rollouts_evaluated.jsonl",
     "#000000"),
    ("E↓ LoRA (coeff=1.0)",
     f"{SUPP}/rollout_sweep_lora_t0.7_steering_eminus/scale_+1.00/baseline/evals/rollouts_evaluated.jsonl",
     "#c91546"),
    ("E↓ activation capping (coeff=1.0)",
     f"{SUPP}/rollout_sweep_activation_capping_t0.7_steering_eminus/frac_1.00/baseline/evals/rollouts_evaluated.jsonl",
     "#3c7fb1"),
    ("Sysprompt-induce E↓",
     f"{SUPP}/rollout_sysprompt_elicit_t0.7_steering_eminus/base/sysprompt_elicit_extraversion_low/evals/rollouts_evaluated.jsonl",
     "#0f7f3f"),
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


def _load_overall_ext(path: str) -> float | None:
    fs = HfFileSystem()
    try:
        d = json.loads(fs.cat(path).decode())
        return float(d["aggregates"]["overall/extraversion_v2.score/mean"])
    except Exception:
        return None


def _per_turn_scores(entries: list[dict[str, Any]], judge: str) -> dict[int, list[float]]:
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
    out: dict[int, dict[str, float]] = {}
    for t, vs in sorted(by_turn.items()):
        m, lo, hi = _bootstrap(vs, BOOTSTRAP_SEED + t)
        out[t] = {"mean": m, "lo": lo, "hi": hi}
    return out


def _ext_distribution(eval_path: str) -> Counter:
    fs = HfFileSystem()
    text = fs.cat(eval_path).decode()
    counts: Counter = Counter()
    for line in text.splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        for r, msgs in e.get("messages", {}).items():
            for m in msgs:
                if m.get("role") != "assistant":
                    continue
                s = (m.get("scores") or {}).get("extraversion_v2", {})
                v = s.get("score") if isinstance(s, dict) else s
                if v is not None:
                    counts[int(v)] += 1
    return counts


def _render_headline() -> None:
    print("\n=== E↓ headline (per-turn trajectory) ===")
    cell_data = []
    for label, path, colour, linestyle, marker in HEADLINE_CELLS:
        print(f"  {label}")
        entries = _load_evaluated(path)
        cell_data.append((label, entries, colour, linestyle, marker))

    n_judges = len(JUDGES)
    fig, axes = plt.subplots(n_judges, 1, figsize=(8.0, 3.5 * n_judges), sharex=True)
    if n_judges == 1:
        axes = [axes]

    for ax, (judge, ylabel, ylim) in zip(axes, JUDGES):
        for label, entries, colour, linestyle, marker in cell_data:
            agg = _aggregate(_per_turn_scores(entries, judge))
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
                linewidth=2, markersize=6, label=label,
                capsize=4, capthick=1.2, elinewidth=1.2,
            )
        if ylim is not None and ylim[0] < 0 < ylim[1]:
            ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    axes[0].set_title(
        "Inducing E↓ persona: per-turn dynamics across induction methods",
        fontsize=12, loc="left", pad=8,
    )
    axes[-1].set_xlabel("Turn index", fontsize=11)
    fig.tight_layout()

    out_pdf = PAPER_FIGURES_DIR / "appendix" / "fig_G_induction_eminus_floor.pdf"
    out_png = out_pdf.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_pdf}")


def _render_distribution() -> None:
    print("\n=== E↓ supplementary distribution / mean-vs-strength ===")
    base_ext = _load_overall_ext(BASE_PATH)
    sysp_ext = _load_overall_ext(SYSP_LOW_PATH)

    def _series(paths: dict[str, str]) -> tuple[list[float], list[float]]:
        xs, ys = [], []
        for s, p in sorted(paths.items(), key=lambda kv: float(kv[0])):
            v = _load_overall_ext(p)
            if v is not None:
                xs.append(float(s))
                ys.append(v)
        return xs, ys

    def _series_neg(paths: dict[str, str]) -> tuple[list[float], list[float]]:
        xs, ys = [], []
        for s, p in sorted(paths.items(), key=lambda kv: float(kv[0])):
            v = _load_overall_ext(p)
            if v is not None:
                xs.append(abs(float(s)))
                ys.append(v)
        return xs, ys

    eml_x, eml_y = _series(EMINUS_LORA_PATHS)
    emc_x, emc_y = _series(EMINUS_ACTCAP_PATHS)
    epn_x, epn_y = _series_neg(EPLUS_NEG_PATHS)

    distributions = []
    for label, p, colour in DIST_CELLS:
        d = _ext_distribution(p)
        n = sum(d.values())
        distributions.append((label, d, colour, n))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel (a) — mean vs strength
    ax = axes[0]
    if base_ext is not None:
        ax.axhline(base_ext, color="grey", linewidth=0.8, linestyle=":",
                   alpha=0.6, label=f"base ext (= {base_ext:.2f})")
    if sysp_ext is not None:
        ax.axhline(sysp_ext, color="#0f7f3f", linewidth=1.2, linestyle="--",
                   alpha=0.85, label=f"sysprompt-induce E↓ (= {sysp_ext:.2f})")
    ax.plot(eml_x, eml_y,
            color="#c91546", marker="o", markersize=8, linewidth=2.0,
            label="E↓ LoRA (positive coefficient)")
    ax.plot(epn_x, epn_y,
            color="#df6f4f", marker="s", markersize=8, linewidth=2.0, linestyle="--",
            label="E↑ LoRA (negative coefficient, |coeff| shown)")
    ax.plot(emc_x, emc_y,
            color="#3c7fb1", marker="D", markersize=8, linewidth=2.0,
            label="E↓ activation capping")
    ax.set_xlabel("Intervention coefficient", fontsize=11)
    ax.set_ylabel("Mean extraversion across all turns", fontsize=11)
    ax.set_title("(a) Mean extraversion vs intervention strength", loc="left", pad=8, fontsize=12)
    ax.set_ylim(-3.5, 0)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")

    # Panel (b) — score histograms
    ax = axes[1]
    score_range = list(range(-4, 5))
    bar_width = 0.20
    for i, (label, dist, colour, n) in enumerate(distributions):
        x_pos = [s + (i - 1.5) * bar_width for s in score_range]
        ys = [dist.get(s, 0) / n if n else 0.0 for s in score_range]
        ax.bar(
            x_pos, ys,
            width=bar_width, color=colour, alpha=0.92,
            edgecolor="#2f3748", linewidth=0.5,
            label=f"{label} (n={n})",
        )
    ax.set_xticks(score_range)
    ax.set_xticklabels([str(s) for s in score_range])
    ax.set_xlabel("Extraversion score", fontsize=11)
    ax.set_ylabel("Fraction of assistant messages", fontsize=11)
    ax.set_title("(b) Distribution of per-message scores", loc="left", pad=8, fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        "E↓ direction: weight/activation methods vs sysprompt — strength curves and score distributions",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()

    out_pdf = PAPER_FIGURES_DIR / "appendix" / "fig_G_induction_eminus_distribution.pdf"
    out_png = out_pdf.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {out_pdf}")


def main() -> None:
    _render_headline()
    _render_distribution()


if __name__ == "__main__":
    main()
