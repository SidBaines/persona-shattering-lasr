"""Sycophancy + CoCoNot bars for the agreeableness adapter.

Two side-by-side panels, 4 bars each (Base, Control, A-, A+):
  * Left  — inspect_evals/sycophancy, ``admits_mistake`` rate
            (higher = more sycophantic, caves under "Are you sure?")
  * Right — inspect_evals/coconot, ``model_graded_qa`` UNACCEPTABLE rate
            (higher = complies more with problematic requests)

All 8 cells are per-sample binary → Wilson 95% CI (CLAUDE.md).

Comparability:
  * Sycophancy panel: all 4 runs scored on the same 1000 sample IDs at T=1.0.
    Base/A+ come from ``a_plus_vanton4_full`` (shuffle=True drew a random
    1000). A-/Control were rerun with ``pinned_sample_ids`` matching that
    exact 1000 at T=1.0 (``*_pinned_t1_v2`` runs).
  * CoCoNot panel: all 4 runs at T=0.0 on the same deterministic 1000
    samples (coconot default slice).

Output (both .png and .pdf):
    paper/figures/main/sycophancy_coconot_bars.{png,pdf}
    scratch/evals/sycophancy_coconot_bars.png

Usage
-----
    uv run python -m src_dev.visualisations.plots_for_paper.plot_trait_behaviour_bars
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from src_dev.evals.personality.analyze_results import _interval_ci_from_wilson
from src_dev.utils.hf_hub import download_path_to_dir
from src_dev.visualisations import PAPER_FIGURES_DIR

load_dotenv()

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

HF_REPO = "persona-shattering-lasr/monorepo"

# (group, role) → (hf_path_in_repo, run_subdir, eval_name).
RUNS: dict[tuple[str, str], tuple[str, str, str]] = {
    # -- Sycophancy (pinned 1000 IDs at T=1.0 across all 4 conditions) --
    ("sycophancy", "Base"): (
        "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/vanton4"
        "/evals/mcq/sycophancy/a_plus_vanton4_full",
        "base",
        "sycophancy",
    ),
    ("sycophancy", "Control"): (
        "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_seed1"
        "/evals/mcq/sycophancy/control_vanton4_seed1_pinned_t1_v2",
        "lora_+1p00x",
        "sycophancy",
    ),
    ("sycophancy", "A-"): (
        "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/vanton4"
        "/evals/mcq/sycophancy/a_minus_vanton4_pinned_t1_v2",
        "lora_+1p00x",
        "sycophancy",
    ),
    ("sycophancy", "A+"): (
        "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/vanton4"
        "/evals/mcq/sycophancy/a_plus_vanton4_full",
        "lora_+1p00x",
        "sycophancy",
    ),
    # -- CoCoNot (all T=0.0 on the same deterministic 1000 samples) --
    ("coconot", "Base"): (
        "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/vanton4"
        "/evals/mcq/coconot/a_plus_vanton4_full",
        "base",
        "coconot",
    ),
    ("coconot", "Control"): (
        "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_seed1"
        "/evals/mcq/coconot/control_vanton4_seed1_full",
        "lora_+1p00x",
        "coconot",
    ),
    ("coconot", "A-"): (
        "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/vanton4"
        "/evals/mcq/coconot/a_minus_vanton4_full",
        "lora_+1p00x",
        "coconot",
    ),
    ("coconot", "A+"): (
        "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/vanton4"
        "/evals/mcq/coconot/a_plus_vanton4_full",
        "lora_+1p00x",
        "coconot",
    ),
}

ROLES = ["Base", "Control", "A-", "A+"]
COLORS = {"Base": "#888888", "Control": "#E6B800", "A-": "#E0592A", "A+": "#2176AE"}
HATCHES = {"Base": "", "Control": "//", "A-": "\\\\", "A+": "xx"}

PANELS = [
    ("sycophancy", "Sycophancy (caves under “Are you sure?”)", "admits_mistake rate"),
    ("coconot",    "CoCoNot (complies with problematic prompts)", "UNACCEPTABLE rate"),
]

LOCAL_CACHE_ROOT = Path("scratch/evals/_hf_cache/trait_behaviour_bars")


def _ensure_from_hf(hf_path: str, scale_dir: str, eval_name: str) -> Path:
    run_name = hf_path.rstrip("/").rsplit("/", 1)[-1]
    target = LOCAL_CACHE_ROOT / f"{run_name}__{scale_dir}__{eval_name}"
    expected = target / scale_dir / eval_name / "native" / "inspect_logs"
    if expected.exists() and any(expected.glob("*.json")):
        return target
    print(f"  pulling {HF_REPO}:{hf_path} ({scale_dir}/{eval_name}) → {target}", flush=True)
    download_path_to_dir(
        repo_id=HF_REPO,
        path_in_repo=hf_path,
        target_dir=target,
        allow_patterns=[
            f"{scale_dir}/{eval_name}/native/inspect_logs/*.json",
            f"{scale_dir}/{eval_name}/run_info.json",
            "suite_config.json",
        ],
    )
    return target


def _resolve_log(run_dir: Path, scale_dir: str, eval_name: str) -> Path | None:
    logs_dir = run_dir / scale_dir / eval_name / "native" / "inspect_logs"
    if not logs_dir.exists():
        return None
    candidates = sorted(logs_dir.glob("*.json"))
    return candidates[-1] if candidates else None


def _sycophancy_caves(samples: list[dict]) -> np.ndarray:
    out: list[int] = []
    for s in samples:
        v = (s.get("scores") or {}).get("sycophancy_scorer", {}).get("value")
        if not isinstance(v, dict):
            continue
        raw = v.get("admits_mistake")
        if raw is None:
            continue
        out.append(1 if str(raw).upper().startswith("C") else 0)
    return np.array(out, dtype=int)


def _coconot_compliance(samples: list[dict]) -> np.ndarray:
    """Upstream grader returns ACCEPTABLE/UNACCEPTABLE/NEITHER; UNACCEPTABLE = complied."""
    out: list[int] = []
    for s in samples:
        v = (s.get("scores") or {}).get("model_graded_qa", {}).get("value")
        if not isinstance(v, str):
            continue
        out.append(1 if v.lower() == "unacceptable" else 0)
    return np.array(out, dtype=int)


METRIC_EXTRACTOR = {
    "sycophancy": _sycophancy_caves,
    "coconot":    _coconot_compliance,
}


def _wilson(values: np.ndarray) -> tuple[float, float, float, int]:
    n = len(values)
    if n == 0:
        return (float("nan"), 0.0, 0.0, 0)
    mean = float(values.mean())
    lo, hi = _interval_ci_from_wilson(values.astype(float), confidence=95)
    return mean, max(0.0, mean - lo), max(0.0, hi - mean), n


def main() -> None:
    print("Resolving runs from HuggingFace...")
    cells: dict[tuple[str, str], tuple[float, float, float, int]] = {}
    for (group, role), (hf_path, scale_dir, eval_name) in RUNS.items():
        run_dir = _ensure_from_hf(hf_path, scale_dir, eval_name)
        log_path = _resolve_log(run_dir, scale_dir, eval_name)
        if log_path is None:
            print(f"    {group}/{role}: no log at {run_dir}/{scale_dir}/{eval_name}")
            cells[(group, role)] = (float("nan"), 0.0, 0.0, 0)
            continue
        log = json.loads(log_path.read_text())
        samples = log.get("samples") or []
        vals = METRIC_EXTRACTOR[group](samples)
        cells[(group, role)] = _wilson(vals)
        m, lo, hi, n = cells[(group, role)]
        print(f"    {group:10s} / {role:7s}  mean={m:.3f}  CI=[{m-lo:.3f}, {m+hi:.3f}]  n={n}")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=True)
    for ax, (group, title, ylabel) in zip(axes, PANELS):
        x = np.arange(len(ROLES))
        heights = [cells[(group, r)][0] for r in ROLES]
        lo_e    = [cells[(group, r)][1] for r in ROLES]
        hi_e    = [cells[(group, r)][2] for r in ROLES]
        bars = ax.bar(
            x, heights,
            yerr=np.array([lo_e, hi_e]),
            width=0.98,
            capsize=4,
            color=[COLORS[r] for r in ROLES],
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, role in zip(bars, ROLES):
            bar.set_hatch(HATCHES[role])
        ax.margins(x=0.01)
        ax.set_xticks(x)
        ax.set_xticklabels(ROLES)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Agreeableness adapter: sycophancy (left) and refusal behaviour (right), n=1000",
        fontweight="bold", fontsize=13, y=1.02,
    )
    fig.tight_layout()

    out_png_scratch = Path("scratch/evals/sycophancy_coconot_bars.png")
    out_png_scratch.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png_scratch, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_png_scratch}")

    out_png_paper = PAPER_FIGURES_DIR / "main" / "sycophancy_coconot_bars.png"
    out_pdf_paper = PAPER_FIGURES_DIR / "main" / "sycophancy_coconot_bars.pdf"
    out_png_paper.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png_paper, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf_paper, bbox_inches="tight")
    print(f"Saved: {out_png_paper}")
    print(f"Saved: {out_pdf_paper}")


PAPER_FIGURES = [
    "main/sycophancy_coconot_bars.png",
    "main/sycophancy_coconot_bars.pdf",
]


if __name__ == "__main__":
    main()
