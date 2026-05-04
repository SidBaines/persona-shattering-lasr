"""Appendix figure: the E− direction floor — LoRA/actcap can't reach where sysprompt does.

Two-panel figure:
  (a) Mean extraversion vs intervention strength for each E−-targeting method.
      Shows that LoRA, actcap, and even pushing past the usable-coherence range
      hit a floor around extraversion ~ -1.3, while sysprompt-induce-low blows
      past it to -2.3. The hypothesis is that this is a method-specific floor,
      not a context floor.
  (b) Per-score histogram (count of assistant messages at each extraversion
      score in {-4, ..., +4}) for: base, E- LoRA at scale 1.0, E- actcap at
      fraction 1.0, sysprompt-induce-low. Shows that sysprompt shifts mass
      cleanly into the {-2, -3} region while LoRA/actcap leave most mass at
      {0, -2} (only partial shift toward -3).

Data: all under fine_tuning/llama-3.1-8b-it/ocean/extraversion/suppressor/vanton4_paired_dpo/rollouts/
  E- LoRA scale {0.25..3.0}: rollout_sweep_lora_t0.7_steering_eminus/scale_+{X}/
  E+ LoRA at neg scale {-0.25..-1.0}: amplifier/.../rollout_sweep_lora_t0.7_steering/scale_{X}/  (alternative path)
  E- actcap fraction {0.25..1.0}: rollout_sweep_activation_capping_t0.7_steering_eminus/frac_{X}/
  Sysprompt-induce-low: rollout_sysprompt_elicit_t0.7_steering_eminus/base/sysprompt_elicit_extraversion_low/
  Base: amplifier/.../rollout_baseline_t0.7_steering/base/baseline/  (shared base)

Paper figures:
    - paper/figures/appendix/fig_G_induction_eminus_floor.pdf

Run with:
    uv run python -m src_dev.visualisations.paper_appendix_induction_eminus_floor
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from huggingface_hub import HfFileSystem  # noqa: E402

from src_dev.visualisations import PAPER_FIGURES_DIR  # noqa: E402

PAPER_FIGURES = [
    "appendix/fig_G_induction_eminus_floor.pdf",
]

HF_FS = "datasets/persona-shattering-lasr/monorepo/fine_tuning/llama-3.1-8b-it/ocean/extraversion"
SUPP = f"{HF_FS}/suppressor/vanton4_paired_dpo/rollouts"
AMP = f"{HF_FS}/amplifier/vanton4_paired_dpo/rollouts"

# Mean-vs-strength curves for panel (a)
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
SYSP_LOW_PATH = f"{SUPP}/rollout_sysprompt_elicit_t0.7_steering_eminus/base/sysprompt_elicit_extraversion_low/run_info.json"
BASE_PATH = f"{AMP}/rollout_baseline_t0.7_steering/base/baseline/run_info.json"

# Distribution histograms for panel (b)
DIST_CELLS: list[tuple[str, str, str]] = [
    ("Base (no intervention)",
     f"{AMP}/rollout_baseline_t0.7_steering/base/baseline/evals/rollouts_evaluated.jsonl",
     "#000000"),
    ("E− LoRA scale 1.0",
     f"{SUPP}/rollout_sweep_lora_t0.7_steering_eminus/scale_+1.00/baseline/evals/rollouts_evaluated.jsonl",
     "#c91546"),
    ("E− actcap frac 1.0",
     f"{SUPP}/rollout_sweep_activation_capping_t0.7_steering_eminus/frac_1.00/baseline/evals/rollouts_evaluated.jsonl",
     "#3c7fb1"),
    ("Sysprompt-induce E−",
     f"{SUPP}/rollout_sysprompt_elicit_t0.7_steering_eminus/base/sysprompt_elicit_extraversion_low/evals/rollouts_evaluated.jsonl",
     "#0f7f3f"),
]


def _load_overall_ext(path: str) -> float | None:
    fs = HfFileSystem()
    try:
        d = json.loads(fs.cat(path).decode())
        return float(d["aggregates"]["overall/extraversion_v2.score/mean"])
    except Exception as e:
        print(f"  ERR {path}: {e.__class__.__name__}")
        return None


def _load_overall_coh(path: str) -> float | None:
    fs = HfFileSystem()
    try:
        d = json.loads(fs.cat(path).decode())
        return float(d["aggregates"]["overall/coherence_v2.score/mean"])
    except Exception:
        return None


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


def main() -> None:
    print("Loading mean-vs-strength data for panel (a)...")
    base_ext = _load_overall_ext(BASE_PATH)

    eminus_lora_xs, eminus_lora_ys = [], []
    for s, p in sorted(EMINUS_LORA_PATHS.items(), key=lambda kv: float(kv[0])):
        v = _load_overall_ext(p)
        if v is not None:
            eminus_lora_xs.append(float(s))
            eminus_lora_ys.append(v)
    print(f"  E- LoRA: {list(zip(eminus_lora_xs, eminus_lora_ys))}")

    eminus_actcap_xs, eminus_actcap_ys = [], []
    for s, p in sorted(EMINUS_ACTCAP_PATHS.items(), key=lambda kv: float(kv[0])):
        v = _load_overall_ext(p)
        if v is not None:
            eminus_actcap_xs.append(float(s))
            eminus_actcap_ys.append(v)
    print(f"  E- actcap: {list(zip(eminus_actcap_xs, eminus_actcap_ys))}")

    eplus_neg_xs, eplus_neg_ys = [], []
    for s, p in sorted(EPLUS_NEG_PATHS.items(), key=lambda kv: float(kv[0])):
        v = _load_overall_ext(p)
        if v is not None:
            # plot magnitude on x so they line up with positive sweeps
            eplus_neg_xs.append(abs(float(s)))
            eplus_neg_ys.append(v)
    print(f"  E+ negated: {list(zip(eplus_neg_xs, eplus_neg_ys))}")

    sysp_ext = _load_overall_ext(SYSP_LOW_PATH)
    print(f"  Sysprompt-induce E-: ext={sysp_ext}")

    print("\nLoading distributions for panel (b)...")
    distributions = []
    for label, p, colour in DIST_CELLS:
        d = _ext_distribution(p)
        n = sum(d.values())
        print(f"  {label}: n={n}, dist={dict(sorted(d.items()))}")
        distributions.append((label, d, colour, n))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel (a)
    ax = axes[0]
    if base_ext is not None:
        ax.axhline(base_ext, color="grey", linewidth=0.8, linestyle=":",
                   alpha=0.6, label=f"base ext (= {base_ext:.2f})")
    if sysp_ext is not None:
        ax.axhline(sysp_ext, color="#0f7f3f", linewidth=1.2, linestyle="--",
                   alpha=0.85, label=f"sysprompt-induce E− (= {sysp_ext:.2f})")
    ax.plot(eminus_lora_xs, eminus_lora_ys,
            color="#c91546", marker="o", markersize=8, linewidth=2.0,
            label="E− LoRA (positive scale)")
    ax.plot(eplus_neg_xs, eplus_neg_ys,
            color="#df6f4f", marker="s", markersize=8, linewidth=2.0, linestyle="--",
            label="E+ LoRA (negative scale, |scale| shown)")
    ax.plot(eminus_actcap_xs, eminus_actcap_ys,
            color="#3c7fb1", marker="D", markersize=8, linewidth=2.0,
            label="E− activation capping")
    ax.set_xlabel("Intervention strength (LoRA scale or actcap fraction)", fontsize=11)
    ax.set_ylabel("Mean extraversion across all turns", fontsize=11)
    ax.set_title("(a) Method-specific floor on E− steering", loc="left", pad=8, fontsize=12)
    ax.set_ylim(-3.5, 0)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")

    # Panel (b)
    ax = axes[1]
    score_range = list(range(-4, 5))
    bar_width = 0.20
    for i, (label, dist, colour, n) in enumerate(distributions):
        # x positions offset within the score bin
        x_pos = [s + (i - 1.5) * bar_width for s in score_range]
        # normalize to fraction so cells with different n are comparable
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
    ax.set_title("(b) Score distributions: who reaches deep introversion?", loc="left", pad=8, fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()

    out_pdf = PAPER_FIGURES_DIR / "appendix" / "fig_G_induction_eminus_floor.pdf"
    out_png = out_pdf.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
