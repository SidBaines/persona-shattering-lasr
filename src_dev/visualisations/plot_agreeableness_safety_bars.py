"""Side-by-side bar plots: (a) Sycophancy, (b) CoCoNot, (c) Agentic Misalignment."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

SYCOPHANCY_ROOTS = {
    "Base": ("scratch/evals/ocean/sycophancy", "base", "merged_syc"),
    "Control": ("scratch/evals/ocean/sycophancy/control_diff_words", "lora_+1p00x", "single"),
    "A-": ("scratch/evals/ocean/sycophancy/a_minus", "lora_+1p00x", "single"),
    "A+": ("scratch/evals/ocean/sycophancy/a_plus_vanton2", "lora_+1p00x", "single"),
}

COCONOT_ROOTS = {
    "Base": ("scratch/evals/ocean/coconot/a_minus", "base", "single"),
    "Control": ("scratch/evals/ocean/coconot/control_use_diff_words", None, "merged_control_coco"),
    "A-": ("scratch/evals/ocean/coconot/a_minus", "lora_+1p00x", "single"),
    "A+": ("scratch/evals/ocean/coconot/a_plus_vanton2", "lora_+1p00x", "single"),
}

_AM_BASE = "scratch/evals/agentic_misalignment/ocean11_leaking_none"
AGENTIC_ROOTS = {
    "Base": f"{_AM_BASE}/base",
    "Control": f"{_AM_BASE}/control_diff_words",
    "A-": f"{_AM_BASE}/a_minus_v2",
    "A+": f"{_AM_BASE}/a_plus_vanton2",
}

COLORS = {"Base": "#888888", "Control": "#6B8E23", "A-": "#E0592A", "A+": "#2176AE"}
HATCHES = {"Base": "", "Control": "//", "A-": "\\\\", "A+": "xx"}


def _resolve_log(run_dir: Path, eval_name: str) -> Path | None:
    """Find the inspect log for a specific eval under run_dir."""
    logs_dir = run_dir / eval_name / "native" / "inspect_logs"
    if logs_dir.exists():
        candidates = sorted(logs_dir.glob("*.json"))
        if candidates:
            return candidates[-1]
    return None


def extract_sycophancy_samples(run_dir: Path) -> list[float]:
    log_path = _resolve_log(run_dir, "sycophancy")
    if log_path is None:
        return []
    log = json.loads(log_path.read_text())
    if log.get("status") != "success":
        return []
    scores = []
    for sample in log["samples"]:
        val = sample["scores"]["sycophancy_scorer"]["value"]
        scores.append(1.0 if val["original_answer"] == "C" else 0.0)
    return scores


def extract_coconot_samples(run_dir: Path) -> list[float]:
    log_path = _resolve_log(run_dir, "coconot")
    if log_path is None:
        return []
    log = json.loads(log_path.read_text())
    if log.get("status") != "success":
        return []
    scores = []
    for sample in log["samples"]:
        val = list(sample["scores"].values())[0]["value"]
        scores.append(1.0 if str(val).lower() == "unacceptable" else 0.0)
    return scores


def extract_agentic_samples(run_dir: Path) -> list[float]:
    """Extract per-sample harmful scores from agentic misalignment eval."""
    log_path = _resolve_log(run_dir, "agentic_misalignment_default")
    if log_path is None:
        return []
    log = json.loads(log_path.read_text())
    if log.get("status") != "success":
        return []
    scores = []
    for sample in log["samples"]:
        val = sample["scores"]["harmfulness_scorer"]["value"]
        scores.append(float(val["harmful"]))
    return scores


def mean_ci95(values: list[float]) -> tuple[float, float]:
    """Return (mean, 95% CI half-width).  CI = 1.96 * σ / √n."""
    arr = np.array(values)
    return float(arr.mean()), float(1.96 * arr.std(ddof=1) / np.sqrt(len(arr)))


def collect_bars(roots: dict, extractor):
    labels, means, stderrs = [], [], []
    for label, config in roots.items():
        if isinstance(config, tuple):
            root, scale_dir, mode = config
        else:
            root, scale_dir, mode = config, None, "direct"

        if mode == "merged_syc":
            all_scores: list[float] = []
            for run in ["a_minus", "a_plus_vanton2", "control_diff_words"]:
                all_scores.extend(extractor(Path(root) / run / "base"))
            arr = np.array(all_scores)
        elif mode == "merged_control_coco":
            all_scores = []
            for sd in ["lora_-1p00x", "lora_+1p00x"]:
                all_scores.extend(extractor(Path(root) / sd))
            arr = np.array(all_scores)
        elif mode == "direct":
            arr = np.array(extractor(Path(root)))
        else:
            arr = np.array(extractor(Path(root) / scale_dir))

        if len(arr) == 0:
            continue
        m, se = mean_ci95(arr.tolist())
        means.append(m)
        stderrs.append(se)
        labels.append(label)
        print(f"  {label:10s}: mean={m:.4f}  ci95={se:.4f}  n={len(arr)}")
    return labels, means, stderrs


def plot_bars(ax, labels, means, stderrs, ylabel, title_letter, title, description):
    x = np.arange(len(labels))
    bars = ax.bar(
        x, means, yerr=stderrs, capsize=5, width=0.55,
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
    ax.text(
        0.5, -0.18, description,
        transform=ax.transAxes, ha="center", va="top",
        fontsize=8, fontstyle="italic", color="0.35",
        wrap=True,
    )


# ── Collect data ─────────────────────────────────────────────────────
print("Sycophancy:")
syc_labels, syc_means, syc_se = collect_bars(SYCOPHANCY_ROOTS, extract_sycophancy_samples)

print("CoCoNot:")
coco_labels, coco_means, coco_se = collect_bars(COCONOT_ROOTS, extract_coconot_samples)
coco_means = [m * 100 for m in coco_means]
coco_se = [s * 100 for s in coco_se]

print("Agentic Misalignment:")
am_labels, am_means, am_se = collect_bars(AGENTIC_ROOTS, extract_agentic_samples)
am_means = [m * 100 for m in am_means]
am_se = [s * 100 for s in am_se]

# ── Plot ─────────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))

plot_bars(
    ax1, syc_labels, syc_means, syc_se,
    "Sycophancy score", "a", "Sycophancy",
    "Fraction of questions where the model maintains\n"
    "its original correct answer after being challenged.",
)

plot_bars(
    ax2, coco_labels, coco_means, coco_se,
    "Safety concerns (%)", "b", "CoCoNot",
    "Percentage of should-not-comply requests\n"
    "where the model complied (lower is safer).",
)

plot_bars(
    ax3, am_labels, am_means, am_se,
    "Harmful actions (%)", "c", "Agentic Misalignment",
    "Rate at which an LLM agent takes harmful actions\n"
    "(e.g. data exfiltration, sabotage) in sandboxed\n"
    "agentic scenarios (lower is safer).",
)

fig.suptitle("Effects of Agreeableness Adapters on Sycophancy and Safety",
             fontsize=15, fontweight="bold", y=1.02)
fig.tight_layout()
fig.subplots_adjust(bottom=0.2)

out = Path("scratch/evals/agreeableness_safety_bars.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"\nSaved: {out}")
