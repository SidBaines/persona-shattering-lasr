"""Appendix figure: cross-LoRA controls for E+ induction.

A single 4-panel small-multiples figure, one panel per control:
  (a) E+ no-DPO sweep — the SFT-only flavour of the canonical E+ adapter.
      Tests how much of the E+ effect comes from the DPO step vs SFT alone.
  (b) C- sweep — conscientiousness suppressor LoRA. Tests cross-trait bleed
      onto extraversion. (Uses extraversion judge merged via cross_judge_eval.)
  (c) Control LoRA sweep — trained on OCEAN definitions without trait shaping.
      Tests for generic LoRA-effect.
  (d) E+/E- soup — three combo cells at matched scales. Tests linear cancellation.

Each panel plots extraversion (mean across all turns of all rollouts) on the
y-axis vs the LoRA scale (or combo specifier) on the x-axis. The canonical E+
LoRA sweep is overlaid in every panel (faint grey dashed) as the reference
"this is what trait-targeted LoRA does at each scale".

Data sources (HF persona-shattering-lasr/monorepo):
  E+ canonical:    fine_tuning/.../extraversion/amplifier/vanton4_paired_dpo/rollouts/rollout_sweep_lora_t0.7_steering/scale_+{X}/baseline/run_info.json
  E+ no-DPO:       fine_tuning/.../extraversion/amplifier/vanton4/rollouts/rollout_sweep_lora_t0.7_crossLoRA/scale_+{X}/baseline/run_info.json
  C-:              fine_tuning/.../conscientiousness/suppressor/vanton4_paired_dpo/rollouts/rollout_sweep_lora_t0.7_crossLoRA/scale_+{X}/baseline/run_info.json
  control:         fine_tuning/.../extraversion/amplifier/vanton4_paired_dpo_s1vs2/rollouts/rollout_sweep_lora_t0.7_crossLoRA/scale_+{X}/baseline/run_info.json
                   (this is the routing-quirk path — actual control LoRA lives
                   under other/ocean_def_control/, but generate_rollouts.py
                   landed the rollouts here based on the OceanTraitDef)
  soup:            fine_tuning/.../extraversion/amplifier/vanton4_paired_dpo/rollouts/rollout_sweep_lora_combo_t0.7_crossLoRA/{label}/baseline/run_info.json
                   labels: ep05_em025, ep05_em05, ep05_em075

Paper figures:
    - paper/figures/appendix/fig_G_induction_cross_lora_controls.pdf

Run with:
    uv run python -m src_dev.visualisations.paper_appendix_induction_cross_lora
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from huggingface_hub import HfFileSystem  # noqa: E402

from src_dev.visualisations import PAPER_FIGURES_DIR  # noqa: E402

PAPER_FIGURES = [
    "appendix/fig_G_induction_cross_lora_controls.pdf",
]

HF_FS = "datasets/persona-shattering-lasr/monorepo/fine_tuning/llama-3.1-8b-it/ocean"

# Canonical E+ reference path (overlaid on all panels).
EPLUS_PATHS = {
    s: f"{HF_FS}/extraversion/amplifier/vanton4_paired_dpo/rollouts/rollout_sweep_lora_t0.7_steering/scale_+{s}/baseline/run_info.json"
    for s in ["0.25", "0.50", "0.75", "1.00"]
}

# (a) E+ no-DPO
NO_DPO_PATHS = {
    s: f"{HF_FS}/extraversion/amplifier/vanton4/rollouts/rollout_sweep_lora_t0.7_crossLoRA/scale_+{s}/baseline/run_info.json"
    for s in ["0.25", "0.50", "0.75", "1.00"]
}

# (b) C- (need extraversion_v2 metric; we re-judged this via cross_judge_eval.py)
CMINUS_PATHS = {
    s: f"{HF_FS}/conscientiousness/suppressor/vanton4_paired_dpo/rollouts/rollout_sweep_lora_t0.7_crossLoRA/scale_+{s}/baseline/run_info.json"
    for s in ["0.25", "0.50", "0.75", "1.00"]
}

# (c) Control LoRA — landed at the routing-quirk path
CONTROL_PATHS = {
    s: f"{HF_FS}/extraversion/amplifier/vanton4_paired_dpo_s1vs2/rollouts/rollout_sweep_lora_t0.7_crossLoRA/scale_+{s}/baseline/run_info.json"
    for s in ["0.25", "0.50", "0.75", "1.00"]
}

# (d) Soup — combo labels rather than scales
SOUP_BASE = f"{HF_FS}/extraversion/amplifier/vanton4_paired_dpo/rollouts/rollout_sweep_lora_combo_t0.7_crossLoRA"
SOUP_PATHS = {
    "(+0.5, -0.25)": f"{SOUP_BASE}/ep05_em025/baseline/run_info.json",
    "(+0.5, -0.50)": f"{SOUP_BASE}/ep05_em05/baseline/run_info.json",
    "(+0.5, -0.75)": f"{SOUP_BASE}/ep05_em075/baseline/run_info.json",
}


def _load_aggregates(path: str) -> dict[str, float]:
    """Return aggregate dict from run_info.json on HF (filtering to overall/* means).

    rollouts_evaluated.jsonl is more authoritative for per-trait scores when
    multiple judges are in play, but for the canonical-trait-only cells the
    aggregates dict is sufficient and faster.
    """
    fs = HfFileSystem()
    d = json.loads(fs.cat(path).decode())
    a = d["aggregates"]
    out: dict[str, float] = {}
    for k, v in a.items():
        if isinstance(v, float) and k.startswith("overall/"):
            out[k] = v
    return out


def _load_extraversion_from_evals(path_run_info: str) -> float | None:
    """Some cells (like C- which was re-judged with cross_judge_eval) carry the
    extraversion_v2 score in rollouts_evaluated.jsonl rather than in
    run_info.json's overall aggregates. Read those by hand."""
    fs = HfFileSystem()
    eval_path = path_run_info.replace("/run_info.json", "/evals/rollouts_evaluated.jsonl")
    try:
        text = fs.cat(eval_path).decode()
    except Exception:
        return None
    scores: list[float] = []
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
                    scores.append(float(v))
    if not scores:
        return None
    return sum(scores) / len(scores)


def _load_extraversion(path_run_info: str) -> float | None:
    """Try aggregates first; fall back to per-message scan if extraversion isn't there."""
    aggs = _load_aggregates(path_run_info)
    key = "overall/extraversion_v2.score/mean"
    if key in aggs:
        return aggs[key]
    # Not in run_info.json -> scan the eval JSONL
    return _load_extraversion_from_evals(path_run_info)


def _load_coherence(path_run_info: str) -> float | None:
    aggs = _load_aggregates(path_run_info)
    return aggs.get("overall/coherence_v2.score/mean")


def _series(paths_by_key: dict[str, str], extract_x: callable) -> tuple[list[float], list[float], list[float]]:
    """Return (xs, ext_means, coh_means) for plotting."""
    xs, ext_ys, coh_ys = [], [], []
    for label, path in paths_by_key.items():
        try:
            ext = _load_extraversion(path)
            coh = _load_coherence(path)
            if ext is None or coh is None:
                print(f"  WARN missing data for {label}: ext={ext} coh={coh}")
                continue
            xs.append(extract_x(label))
            ext_ys.append(ext)
            coh_ys.append(coh)
        except Exception as e:
            print(f"  ERR {label}: {e}")
    return xs, ext_ys, coh_ys


def main() -> None:
    print("Loading reference E+ canonical sweep...")
    eplus_xs, eplus_ext, eplus_coh = _series(EPLUS_PATHS, lambda s: float(s))
    print(f"  E+ canonical: {list(zip(eplus_xs, eplus_ext))}")

    print("\nLoading control series...")
    nodpo_xs, nodpo_ext, nodpo_coh = _series(NO_DPO_PATHS, lambda s: float(s))
    print(f"  E+ no-DPO: {list(zip(nodpo_xs, nodpo_ext))}")

    cminus_xs, cminus_ext, cminus_coh = _series(CMINUS_PATHS, lambda s: float(s))
    print(f"  C- (cross-trait): {list(zip(cminus_xs, cminus_ext))}")

    ctrl_xs, ctrl_ext, ctrl_coh = _series(CONTROL_PATHS, lambda s: float(s))
    print(f"  Control LoRA: {list(zip(ctrl_xs, ctrl_ext))}")

    # Soup: x is "E- scale within combo" (E+ is fixed at 0.5)
    soup_xs, soup_ext, soup_coh = _series(
        SOUP_PATHS, lambda s: float(s.split(", ")[1].rstrip(")").lstrip("-"))
    )
    print(f"  Soup: {list(zip(soup_xs, soup_ext))}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)

    # Common reference: faint canonical E+ overlay on every LoRA-scale panel
    def _overlay_canonical(ax: plt.Axes) -> None:
        if eplus_xs:
            ax.plot(
                eplus_xs, eplus_ext,
                color="#7f8c9b", linestyle=":", marker=".", markersize=6,
                linewidth=1.4, alpha=0.7, label="E+ LoRA (canonical, ref.)",
                zorder=1,
            )

    # Panel (a): E+ no-DPO
    ax = axes[0][0]
    _overlay_canonical(ax)
    if nodpo_xs:
        ax.plot(
            nodpo_xs, nodpo_ext,
            color="#c91546", marker="^", markersize=8,
            linewidth=2.2, label="E+ no-DPO", zorder=2,
        )
    ax.set_title("(a) E+ LoRA without DPO step", loc="left", pad=8, fontsize=12)
    ax.set_xlabel("LoRA scale", fontsize=11)
    ax.set_ylabel("Extraversion judge score", fontsize=11)
    ax.set_ylim(-2.5, 4)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    # Panel (b): C-
    ax = axes[0][1]
    _overlay_canonical(ax)
    if cminus_xs:
        ax.plot(
            cminus_xs, cminus_ext,
            color="#FF9800",  # OCEAN colour for conscientiousness
            marker="s", markersize=8,
            linewidth=2.2, label="C− LoRA (off-target ext)", zorder=2,
        )
    ax.set_title("(b) Cross-trait bleed: C− LoRA on extraversion", loc="left", pad=8, fontsize=12)
    ax.set_xlabel("LoRA scale", fontsize=11)
    ax.set_ylim(-2.5, 4)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    # Panel (c): Control LoRA
    ax = axes[1][0]
    _overlay_canonical(ax)
    if ctrl_xs:
        ax.plot(
            ctrl_xs, ctrl_ext,
            color="#7f8c9b",
            marker="D", markersize=8,
            linewidth=2.2, label="Control LoRA (no trait signal)", zorder=2,
        )
    ax.set_title("(c) Control LoRA (no trait signal)", loc="left", pad=8, fontsize=12)
    ax.set_xlabel("LoRA scale", fontsize=11)
    ax.set_ylabel("Extraversion judge score", fontsize=11)
    ax.set_ylim(-2.5, 4)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    # Panel (d): Soup
    ax = axes[1][1]
    if soup_xs:
        # Sort by E- scale for proper line plot
        order = sorted(range(len(soup_xs)), key=lambda i: soup_xs[i])
        sx = [soup_xs[i] for i in order]
        sy = [soup_ext[i] for i in order]
        ax.plot(
            sx, sy,
            color="#5b2abf",
            marker="P", markersize=10,
            linewidth=2.2, label="(E+ at 0.5) + (E− at scale x)", zorder=2,
        )
        # Annotate each point with the combo label for clarity
        for x, y in zip(sx, sy):
            ax.annotate(
                f"x={x}", (x, y),
                xytext=(8, -8),
                textcoords="offset points",
                fontsize=8.5, color="#5b2abf",
            )
    ax.axhline(-0.91, color="black", linewidth=1.0, linestyle="--", alpha=0.7,
               label="base ext (= -0.91)")
    ax.set_title("(d) E+/E− soup at matched scales", loc="left", pad=8, fontsize=12)
    ax.set_xlabel("E− component scale (E+ fixed at 0.5)", fontsize=11)
    ax.set_ylim(-2.5, 4)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    fig.suptitle("Cross-LoRA controls on extraversion (neutral prompts, temp 0.7)", fontsize=13, y=1.00)
    fig.tight_layout()

    out_pdf = PAPER_FIGURES_DIR / "appendix" / "fig_G_induction_cross_lora_controls.pdf"
    out_png = out_pdf.with_suffix(".png")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
