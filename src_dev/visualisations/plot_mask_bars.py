"""Bar plots for the MASK honesty benchmark.

Three panels per variant (Base / Control / A- / A+):
  (a) Honesty   — 1 - lie rate (MASK's "overall_honesty" definition).
  (b) Accuracy  — fraction of samples classified `correct`.
  (c) No-belief — fraction where MASK could not elicit a stable belief.
                   High values here *inflate* the headline honesty number
                   (abstention counts as non-lie), so this panel is the
                   caveat for reading (a).

Error bars use Wilson score intervals (binary per-sample data — see
CLAUDE.md). Produces:
  scratch/evals/mask_bars.png
  paper/figures/main/fig_mask_bars.pdf

Usage
-----
    uv run python -m src_dev.visualisations.plot_mask_bars
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src_dev.evals.personality.analyze_results import _interval_ci_from_wilson
from src_dev.visualisations import PAPER_FIGURES_DIR

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
})

# Each entry is (run_dir, scale_dir). Base is pulled from the A+ run's
# base/ directory (shared across all three config runs via baseline cache).
MASK_ROOTS: dict[str, tuple[str, str]] = {
    "Base":    ("scratch/evals/ocean/mask/a_plus_vanton2",        "base"),
    "Control": ("scratch/evals/control/mask/control_diff_words",  "lora_+1p00x"),
    "A-":      ("scratch/evals/ocean/mask/a_minus",               "lora_+1p00x"),
    "A+":      ("scratch/evals/ocean/mask/a_plus_vanton2",        "lora_+1p00x"),
}

COLORS = {"Base": "#888888", "Control": "#6B8E23", "A-": "#E0592A", "A+": "#2176AE"}
HATCHES = {"Base": "", "Control": "//", "A-": "\\\\", "A+": "xx"}


def _resolve_log(run_dir: Path) -> Path | None:
    logs_dir = run_dir / "mask" / "native" / "inspect_logs"
    if not logs_dir.exists():
        return None
    candidates = sorted(logs_dir.glob("*.json"))
    return candidates[-1] if candidates else None


def _load_samples(run_dir: Path) -> list[dict]:
    """Return the list of per-sample accuracy_and_honesty value dicts.

    Each dict looks like {"accuracy": <enum str>, "honesty": <enum str>, ...}.
    """
    log_path = _resolve_log(run_dir)
    if log_path is None:
        return []
    log = json.loads(log_path.read_text())
    if log.get("status") != "success":
        return []
    out = []
    for s in log.get("samples", []):
        v = s.get("scores", {}).get("accuracy_and_honesty", {}).get("value")
        if isinstance(v, dict):
            out.append(v)
    return out


def _binary_samples(
    value_dicts: list[dict], field: str, positive: str | set[str], negative: str | set[str] | None = None,
) -> np.ndarray:
    """Turn per-sample classifications into a 0/1 vector.

    Positive class = 1, negative class = 0. Samples outside both sets are
    dropped (e.g. MASK treats `error` specially — excluded from honesty).
    """
    pos = {positive} if isinstance(positive, str) else set(positive)
    neg = ({negative} if isinstance(negative, str) else set(negative)) if negative else None
    out: list[int] = []
    for v in value_dicts:
        x = v.get(field)
        if x in pos:
            out.append(1)
        elif neg is None or x in neg:
            out.append(0)
    return np.array(out, dtype=float)


def _honesty_samples(value_dicts: list[dict]) -> np.ndarray:
    """Honesty = 1 - lie rate across all non-error samples.

    Matches MASK's `overall_honesty` (honest/evade/no-belief count as 1,
    lie counts as 0, error samples are dropped).
    """
    return _binary_samples(
        value_dicts, "honesty",
        positive={"honest", "evade", "no-belief"},
        negative={"lie"},
    )


def _accuracy_samples(value_dicts: list[dict]) -> np.ndarray:
    """Accuracy = fraction classified `correct` over all non-error samples."""
    return _binary_samples(
        value_dicts, "accuracy",
        positive={"correct"},
        negative={"incorrect", "no-belief", "no-belief-elicitation-done"},
    )


def _no_belief_samples(value_dicts: list[dict]) -> np.ndarray:
    """No-belief rate — fraction where MASK could not elicit a belief.

    Uses the `accuracy` field because it encodes both flavours of the
    no-belief edge case (`no-belief` and `no-belief-elicitation-done`).
    """
    return _binary_samples(
        value_dicts, "accuracy",
        positive={"no-belief", "no-belief-elicitation-done"},
        negative={"correct", "incorrect"},
    )


def _wilson(bits: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, lower_half_width, upper_half_width) for a 0/1 vector."""
    if len(bits) == 0:
        return (float("nan"), 0.0, 0.0)
    lo, hi = _interval_ci_from_wilson(bits, 95)
    mean = float(bits.mean())
    return mean, max(0.0, mean - lo), max(0.0, hi - mean)


def _collect(extractor) -> tuple[list[str], list[float], list[float], list[float]]:
    labels, means, lo_err, hi_err = [], [], [], []
    for label, (root, scale_dir) in MASK_ROOTS.items():
        run_dir = Path(root) / scale_dir
        value_dicts = _load_samples(run_dir)
        bits = extractor(value_dicts)
        if len(bits) == 0:
            print(f"  {label}: no samples at {run_dir}")
            continue
        mean, lo, hi = _wilson(bits)
        labels.append(label)
        means.append(mean)
        lo_err.append(lo)
        hi_err.append(hi)
        print(f"  {label:8s}: mean={mean:.3f}  CI=[{mean-lo:.3f}, {mean+hi:.3f}]  n={len(bits)}")
    return labels, means, lo_err, hi_err


def _plot(ax, labels, means, lo_err, hi_err, *, ylabel, title_letter, title, description, ylim=None):
    x = np.arange(len(labels))
    yerr = np.array([lo_err, hi_err])
    bars = ax.bar(
        x, means, yerr=yerr, capsize=5, width=0.55,
        color=[COLORS[l] for l in labels],
        edgecolor="black", linewidth=0.8,
    )
    for bar, label in zip(bars, labels):
        bar.set_hatch(HATCHES[label])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(f"({title_letter})  {title}", loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.text(
        0.5, -0.22, description,
        transform=ax.transAxes, ha="center", va="top",
        fontsize=8, fontstyle="italic", color="0.35",
        wrap=True,
    )


def main() -> None:
    print("Honesty (1 - lie rate):")
    h_labels, h_means, h_lo, h_hi = _collect(_honesty_samples)
    print("Accuracy:")
    a_labels, a_means, a_lo, a_hi = _collect(_accuracy_samples)
    print("No-belief rate:")
    nb_labels, nb_means, nb_lo, nb_hi = _collect(_no_belief_samples)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    _plot(
        ax1, h_labels, h_means, h_lo, h_hi,
        ylabel="Honesty (1 − lie rate)",
        title_letter="a", title="MASK honesty",
        description="Fraction of samples where the public statement does\n"
                    "not contradict the elicited belief. Higher = more honest.",
        ylim=(0, 1.05),
    )
    _plot(
        ax2, a_labels, a_means, a_lo, a_hi,
        ylabel="Accuracy",
        title_letter="b", title="MASK accuracy",
        description="Fraction of samples where the public statement matches\n"
                    "the ground-truth proposition. Higher = more accurate.",
        ylim=(0, 1.05),
    )
    _plot(
        ax3, nb_labels, nb_means, nb_lo, nb_hi,
        ylabel="No-belief rate",
        title_letter="c", title="Belief-elicitation failure",
        description="Fraction of samples where MASK could not elicit a stable\n"
                    "belief. Counts as non-lie → inflates (a) when high.",
        ylim=(0, 1.05),
    )

    fig.suptitle(
        "Effect of Agreeableness and Control Adapters on MASK Honesty",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)

    out_png = Path("scratch/evals/mask_bars.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_png}")

    out_pdf = PAPER_FIGURES_DIR / "main" / "fig_mask_bars.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
