"""Six-condition sycophancy bar plot (A↑ / A↓ at scale ±1, vs base + control).

Three side-by-side panels — the three upstream ``inspect_evals/sycophancy``
metrics (README → "Scoring"):
    (a) Apologize Rate  — admits mistake | first answer correct (caves under
                          challenge → sycophancy signal).
    (b) Confidence      — does NOT apologize | first answer correct
                          (≈ 1 − apologize_rate within the orig=C subset).
    (c) Truthfulness    — sticks to correct / corrects wrong (overall).

Each panel shows six densely-attached bars in this fixed order:
    base · control · A↓@+1 · A↓@−1 · A↑@+1 · A↑@−1

Per-sample values are read straight off ``sycophancy_scorer.value`` as stored
in the upstream Inspect log (no recompute, no recompute of upstream
aggregators) — the upstream-aggregator caveat from
``paper_sycophancy_apologize_truthfulness-checkpoint.py`` still applies:
``apologize_rate``/``confidence`` denominators are the orig==C subset, while
``truthfulness`` denominator is the whole eval.

Run with:
    uv run python -m src_dev.visualisations.paper_sycophancy_a_six_bars

Data sources (HF dataset ``persona-shattering-lasr/monorepo``):
    base    : evals/sycophancy/llama-3.1-8b-it_base/base_llama_3_1_8b_it/base/
              sycophancy/native/inspect_logs/2026-04-29T18-44-01+00-00_*.json
    control : fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/
              vanton4_seed1/evals/mcq/sycophancy/control_vanton4_seed1_scale1/
              lora_+1p00x/sycophancy/native/inspect_logs/
              2026-04-29T12-46-19+00-00_*.json
    a_minus_p1 : .../agreeableness/suppressor/vanton4_paired_dpo/.../
                 a_minus_vanton4_paired_dpo_scale1/lora_+1p00x/...
                 2026-04-29T12-46-20+00-00_*.json
    a_plus_p1  : .../agreeableness/amplifier/vanton4_paired_dpo/.../
                 a_plus_vanton4_paired_dpo_scale1/lora_+1p00x/...
                 2026-04-29T12-46-20+00-00_*.json
    a_minus_m1 : .../agreeableness/suppressor/vanton4_paired_dpo/.../
                 a_minus_vanton4_paired_dpo_scale-1/lora_-1p00x/...
                 2026-05-01T13-44-46+00-00_*.json
    a_plus_m1  : .../agreeableness/amplifier/vanton4_paired_dpo/.../
                 a_plus_vanton4_paired_dpo_scale-1/lora_-1p00x/...
                 2026-05-01T13-44-47+00-00_*.json
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(project_root / ".env")

from huggingface_hub import hf_hub_download  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402

from src_dev.evals.personality.analyze_results import _interval_ci_from_wilson  # noqa: E402
from src_dev.utils.hf_hub import login_from_env  # noqa: E402
from src_dev.visualisations import PAPER_FIGURES_DIR  # noqa: E402


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


PAPER_FIGURES = [
    "main/fig_sycophancy_a_six_bars.pdf",
]


# Inlined from the project style guide (STYLE_GUIDE.md). The repo doesn't ship
# a science_plots.py module; these are the rcParams + named hexes called out
# in the guide.
PAPER_STYLE: dict[str, object] = {
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.labelsize": 12,
    "axes.facecolor": "#fbfbfc",
    "axes.edgecolor": "#2f3748",
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "axes.axisbelow": True,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.color": "#2f3748",
    "ytick.color": "#2f3748",
    "grid.color": "#dfe3e8",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.75,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#cfd4dc",
    "legend.fontsize": 9.5,
    "lines.linewidth": 2.0,
}

SPINE_COLOR = "#2f3748"
C_PERSONA = "#7f8c9b"  # neutral / structural reference


HF_REPO_ID = "persona-shattering-lasr/monorepo"
CACHE_DIR = project_root / "scratch" / "paper_plots_cache" / "sycophancy_a_six_bars"
OUT_PATH = PAPER_FIGURES_DIR / PAPER_FIGURES[0]


@dataclass(frozen=True)
class Condition:
    key: str
    short: str
    legend: str
    color: str
    hatch: str | None
    inspect_log_in_repo: str


# Order is the x-axis order in both panels.
# Adapter-grouped: each adapter's two scales sit next to each other.
CONDITIONS: list[Condition] = [
    Condition(
        key="base",
        short="base",
        legend="base Llama 3.1-8B-Instruct",
        color="#4D4D4D",
        hatch=None,
        inspect_log_in_repo=(
            "evals/sycophancy/llama-3.1-8b-it_base/base_llama_3_1_8b_it/base/"
            "sycophancy/native/inspect_logs/"
            "2026-04-29T18-44-01+00-00_sycophancy_i2Xzh5RirRoMixwGTugtPL.json"
        ),
    ),
    Condition(
        key="control",
        short="control",
        legend="adapter control (vanton4_seed1) @ +1",
        color="#9E9E9E",
        hatch=None,
        inspect_log_in_repo=(
            "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/"
            "vanton4_seed1/evals/mcq/sycophancy/control_vanton4_seed1_scale1/"
            "lora_+1p00x/sycophancy/native/inspect_logs/"
            "2026-04-29T12-46-19+00-00_sycophancy_8QRoUqgMGCiJK53zQtmaYD.json"
        ),
    ),
    Condition(
        key="a_minus_p1",
        short="A↓ @ +1",
        legend="A↓ suppressor @ scale +1",
        color="#CE93D8",
        hatch=None,
        inspect_log_in_repo=(
            "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/"
            "vanton4_paired_dpo/evals/mcq/sycophancy/"
            "a_minus_vanton4_paired_dpo_scale1/lora_+1p00x/sycophancy/native/"
            "inspect_logs/"
            "2026-04-29T12-46-20+00-00_sycophancy_MF7EmrLtVaBcNZDuU4Na5J.json"
        ),
    ),
    Condition(
        key="a_minus_m1",
        short="A↓ @ −1",
        legend="A↓ suppressor @ scale −1",
        color="#CE93D8",
        hatch="///",
        inspect_log_in_repo=(
            "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/"
            "vanton4_paired_dpo/evals/mcq/sycophancy/"
            "a_minus_vanton4_paired_dpo_scale-1/lora_-1p00x/sycophancy/native/"
            "inspect_logs/"
            "2026-05-01T13-44-46+00-00_sycophancy_6ENuoAEfMCRJitTKYu8iQf.json"
        ),
    ),
    Condition(
        key="a_plus_p1",
        short="A↑ @ +1",
        legend="A↑ amplifier @ scale +1",
        color="#9C27B0",
        hatch=None,
        inspect_log_in_repo=(
            "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/"
            "vanton4_paired_dpo/evals/mcq/sycophancy/"
            "a_plus_vanton4_paired_dpo_scale1/lora_+1p00x/sycophancy/native/"
            "inspect_logs/"
            "2026-04-29T12-46-20+00-00_sycophancy_oWRZ6NnD8gNk69CpMtzMHw.json"
        ),
    ),
    Condition(
        key="a_plus_m1",
        short="A↑ @ −1",
        legend="A↑ amplifier @ scale −1",
        color="#9C27B0",
        hatch="///",
        inspect_log_in_repo=(
            "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/amplifier/"
            "vanton4_paired_dpo/evals/mcq/sycophancy/"
            "a_plus_vanton4_paired_dpo_scale-1/lora_-1p00x/sycophancy/native/"
            "inspect_logs/"
            "2026-05-01T13-44-47+00-00_sycophancy_CL6MLEXn5PT78pkeUayEkQ.json"
        ),
    ),
]


METRICS: list[tuple[str, str, str]] = [
    ("apologize_rate", "Apologize Rate", "caves on push-back"),
    ("confidence",     "Confidence",     "stands by correct answer"),
    ("truthfulness",   "Truthfulness",   "ends on the correct answer"),
]


def _hydrate_log(cond: Condition) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = cond.inspect_log_in_repo.rsplit("/", 1)[1]
    local = CACHE_DIR / cond.key / filename
    if local.exists() and local.stat().st_size > 0:
        return local
    print(f"  hydrating {cond.key} from HF...")
    fetched = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=cond.inspect_log_in_repo,
        repo_type="dataset",
    )
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        local.unlink()
    local.symlink_to(fetched)
    return local


@dataclass
class PerSample:
    n_total: int
    original_answer: np.ndarray  # 1 if "C" else 0
    apologize_rate: np.ndarray   # 1 if (orig=="C" and admits=="C") else 0
    confidence: np.ndarray       # 1 if (orig=="C" and admits!="C") else 0
    truthfulness: np.ndarray     # 1 if is_truthful else 0


def _extract(log_path: Path) -> PerSample:
    log = read_eval_log(str(log_path))
    samples = log.samples or []
    orig: list[int] = []
    apo: list[int] = []
    conf: list[int] = []
    truth: list[int] = []
    for s in samples:
        sc = (s.scores or {}).get("sycophancy_scorer")
        if sc is None:
            continue
        d = sc.value or {}
        o = 1 if d.get("original_answer") == "C" else 0
        orig.append(o)
        apo.append(int(float(d.get("apologize_rate", 0.0))))
        conf.append(int(float(d.get("confidence", 0.0))))
        truth.append(int(float(d.get("truthfulness", 0.0))))
    return PerSample(
        n_total=len(samples),
        original_answer=np.asarray(orig, dtype=int),
        apologize_rate=np.asarray(apo, dtype=int),
        confidence=np.asarray(conf, dtype=int),
        truthfulness=np.asarray(truth, dtype=int),
    )


def _aggregate(ps: PerSample) -> dict[str, tuple[float, float, float, int]]:
    out: dict[str, tuple[float, float, float, int]] = {}
    correct_mask = ps.original_answer.astype(bool)

    # apologize_rate, confidence: denominator = orig=="C" subset (upstream).
    for key, arr in (("apologize_rate", ps.apologize_rate),
                     ("confidence", ps.confidence)):
        sub = arr[correct_mask].astype(float)
        n = int(sub.size)
        if n > 0:
            p = float(sub.mean())
            lo, hi = _interval_ci_from_wilson(sub, confidence=95.0)
        else:
            p, lo, hi = float("nan"), float("nan"), float("nan")
        out[key] = (p, max(lo, 0.0), max(hi, 0.0), n)

    # truthfulness: denominator = all samples.
    tr = ps.truthfulness.astype(float)
    n_tr = int(tr.size)
    p = float(tr.mean())
    lo, hi = _interval_ci_from_wilson(tr, confidence=95.0)
    out["truthfulness"] = (p, max(lo, 0.0), max(hi, 0.0), n_tr)

    return out


def _draw_panel(
    ax: plt.Axes,
    metric_key: str,
    metric_label: str,
    metric_def: str,
    aggregated: dict[str, dict[str, tuple[float, float, float, int]]],
    panel_letter: str,
) -> None:
    n_cond = len(CONDITIONS)
    x = np.arange(n_cond, dtype=float)
    width = 0.96  # densely attached: bars almost touch one another

    values: list[float] = []
    err_lo: list[float] = []
    err_hi: list[float] = []
    colors: list[str] = []
    hatches: list[str | None] = []
    for cond in CONDITIONS:
        p, lo, hi, _n = aggregated[cond.key][metric_key]
        values.append(p)
        err_lo.append(p - lo)
        err_hi.append(hi - p)
        colors.append(cond.color)
        hatches.append(cond.hatch)

    bars = ax.bar(
        x,
        values,
        width=width,
        color=colors,
        alpha=0.92,
        edgecolor=SPINE_COLOR,
        linewidth=0.5,
        zorder=3,
    )
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)

    ax.errorbar(
        x,
        values,
        yerr=[err_lo, err_hi],
        fmt="none",
        ecolor=SPINE_COLOR,
        elinewidth=1.0,
        capsize=2.5,
        capthick=1.0,
        alpha=0.85,
        zorder=4,
    )

    # Inline value labels on bars (semibold, primary number for the reader).
    for xi, v, hi in zip(x, values, err_hi):
        ax.text(
            xi,
            v + hi + 0.018,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=SPINE_COLOR,
            fontweight="semibold",
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([c.short for c in CONDITIONS], rotation=30, ha="right",
                       fontsize=9)
    ax.set_xlim(-0.5, n_cond - 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(metric_label)
    ax.set_title(
        f"({panel_letter}) {metric_label}\n{metric_def}",
        loc="left",
        pad=8,
        fontsize=11,
    )
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)



def _legend_handles() -> list[mpl.patches.Patch]:
    handles: list[mpl.patches.Patch] = []
    for cond in CONDITIONS:
        handles.append(
            mpl.patches.Patch(
                facecolor=cond.color,
                edgecolor=SPINE_COLOR,
                linewidth=0.5,
                alpha=0.92,
                hatch=cond.hatch or "",
                label=cond.legend,
            )
        )
    return handles


def _confirm_settings(logs: dict[str, "object"]) -> None:
    print("\nrun        task                         tver   n_samples  judge")
    print("-" * 90)
    for cond in CONDITIONS:
        log = logs[cond.key]
        judge = (log.eval.task_args or {}).get("scorer_model", "?")
        print(f"{cond.key:<11} {log.eval.task:<28} {log.eval.task_version:<6} "
              f"{len(log.samples or []):<10} {judge}")
    print()


def _render(
    aggregated: dict[str, dict[str, tuple[float, float, float, int]]],
    out_path: Path,
) -> None:
    mpl.rcParams.update(PAPER_STYLE)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.0), sharey=True)
    for ax, (mkey, mlabel, mdef), letter in zip(axes, METRICS, "abc"):
        _draw_panel(ax, mkey, mlabel, mdef, aggregated, letter)

    fig.legend(
        handles=_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=True,
    )
    fig.subplots_adjust(top=0.90, bottom=0.20, wspace=0.10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\n✓ saved {out_path}")
    print(f"✓ saved {out_path.with_suffix('.png')}")


def main() -> None:
    try:
        login_from_env()
    except RuntimeError:
        pass
    print(f"[sycophancy:6-bars] cache: {CACHE_DIR}")
    print(f"[sycophancy:6-bars] out:   {OUT_PATH}\n")

    logs: dict[str, object] = {}
    persamples: dict[str, PerSample] = {}
    for cond in CONDITIONS:
        local = _hydrate_log(cond)
        logs[cond.key] = read_eval_log(str(local))
        persamples[cond.key] = _extract(local)

    _confirm_settings(logs)

    aggregated = {key: _aggregate(ps) for key, ps in persamples.items()}

    print(f"{'metric':<18} " + "  ".join(f"{c.short:>14}" for c in CONDITIONS))
    for metric_key, _label, _def in METRICS:
        cells = []
        for c in CONDITIONS:
            p, lo, hi, _n = aggregated[c.key][metric_key]
            cells.append(f"{p:.3f}[{lo:.2f},{hi:.2f}]")
        print(f"{metric_key:<18} " + "  ".join(f"{cell:>14}" for cell in cells))

    _render(aggregated, OUT_PATH)


if __name__ == "__main__":
    main()
