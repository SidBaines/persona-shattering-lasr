"""Trait-behaviour bars: sycophancy (agreeableness adapter).

Four bars: {Base, Control, A-, A+}. "Caves under 'Are you sure?'" rate — i.e.
``admits_mistake == CORRECT`` on the inspect_evals/sycophancy task. Per-sample
binary, Wilson 95% CI (per CLAUDE.md).

All eval logs are pulled from the HuggingFace monorepo so the plot is
reproducible without local state.

Output:
    paper/figures/main/fig_trait_behaviour_bars.png
    scratch/evals/trait_behaviour_bars.png

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

# role → (hf_path_in_repo, run_subdir, eval_name).
# All four runs score the same 1000 sample IDs at T=1.0 so they are directly
# comparable. Base and A+ come from ``a_plus_vanton4_full`` (shuffle=True drew a
# random 1000 at T=1.0). A- and Control were rerun under ``*_pinned_t1_v2`` with
# shuffle=False + pinned_sample_ids to reproduce that exact 1000-sample subset.
RUNS: dict[str, tuple[str, str, str]] = {
    "Base": (
        "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/vanton4"
        "/evals/mcq/sycophancy/a_plus_vanton4_full",
        "base",
        "sycophancy",
    ),
    "Control": (
        "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_seed1"
        "/evals/mcq/sycophancy/control_vanton4_seed1_pinned_t1_v2",
        "lora_+1p00x",
        "sycophancy",
    ),
    "A-": (
        "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/vanton4"
        "/evals/mcq/sycophancy/a_minus_vanton4_pinned_t1_v2",
        "lora_+1p00x",
        "sycophancy",
    ),
    "A+": (
        "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/vanton4"
        "/evals/mcq/sycophancy/a_plus_vanton4_full",
        "lora_+1p00x",
        "sycophancy",
    ),
}

# Plot order: Base, Control, then the two adapter conditions.
ROLES = ["Base", "Control", "A-", "A+"]
COLORS = {"Base": "#888888", "Control": "#E6B800", "A-": "#E0592A", "A+": "#2176AE"}
HATCHES = {"Base": "", "Control": "//", "A-": "\\\\", "A+": "xx"}

LOCAL_CACHE_ROOT = Path("scratch/evals/_hf_cache/trait_behaviour_bars")


def _ensure_from_hf(hf_path: str, scale_dir: str, eval_name: str) -> Path:
    """Pull just the inspect log for one (run, scale, eval) into a local cache.

    download_path_to_dir wipes ``target_dir`` on each call, so we key the local
    cache dir by (run, scale_dir, eval_name) rather than just run_name to avoid
    one call clobbering the previous call's log when two roles share the same
    HF run (e.g. Base and A+ both live under ``a_plus_vanton4_full``).
    """
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
    # Latest by filename (timestamp-prefixed) wins.
    return candidates[-1] if candidates else None


def _sycophancy_caves(samples: list[dict]) -> np.ndarray:
    """1 if the model caved under 'Are you sure?' (admits_mistake == CORRECT)."""
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


def _wilson(values: np.ndarray) -> tuple[float, float, float, int]:
    n = len(values)
    if n == 0:
        return (float("nan"), 0.0, 0.0, 0)
    mean = float(values.mean())
    lo, hi = _interval_ci_from_wilson(values.astype(float), confidence=95)
    return mean, max(0.0, mean - lo), max(0.0, hi - mean), n


def main() -> None:
    print("Resolving runs from HuggingFace...")
    cells: dict[str, tuple[float, float, float, int]] = {}
    for role in ROLES:
        hf_path, scale_dir, eval_name = RUNS[role]
        run_dir = _ensure_from_hf(hf_path, scale_dir, eval_name)
        log_path = _resolve_log(run_dir, scale_dir, eval_name)
        if log_path is None:
            print(f"    {role}: no log at {run_dir}/{scale_dir}/{eval_name}")
            cells[role] = (float("nan"), 0.0, 0.0, 0)
            continue
        log = json.loads(log_path.read_text())
        samples = log.get("samples") or []
        vals = _sycophancy_caves(samples)
        cells[role] = _wilson(vals)
        m, lo, hi, n = cells[role]
        print(f"    {role:8s}  mean={m:.3f}  CI=[{m-lo:.3f}, {m+hi:.3f}]  n={n}  log={log_path.name}")

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    x = np.arange(len(ROLES))
    heights = [cells[r][0] for r in ROLES]
    lo_e = [cells[r][1] for r in ROLES]
    hi_e = [cells[r][2] for r in ROLES]

    bars = ax.bar(
        x, heights,
        yerr=np.array([lo_e, hi_e]),
        width=0.98,
        capsize=4,
        color=[COLORS[r] for r in ROLES],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.margins(x=0.01)
    for bar, role in zip(bars, ROLES):
        bar.set_hatch(HATCHES[role])

    ax.set_xticks(x)
    ax.set_xticklabels(ROLES)
    ax.set_ylabel("caves under 'Are you sure?' (rate)")
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(
        "Sycophancy: agreeableness adapter (higher = more sycophantic, n=1000)",
        loc="left", fontweight="bold", fontsize=12,
    )
    fig.tight_layout()

    out_png = Path("scratch/evals/trait_behaviour_bars.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_png}")

    out_paper_png = PAPER_FIGURES_DIR / "main" / "fig_trait_behaviour_bars.png"
    out_paper_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_paper_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_paper_png}")


PAPER_FIGURES = ["main/fig_trait_behaviour_bars.png"]


if __name__ == "__main__":
    main()
