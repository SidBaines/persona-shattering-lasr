"""Per-trait delta bar chart: TRAIT-logprobs + LLM-judge vs baseline llama.

Rehydrates pre-computed eval cells from the HF monorepo (no evals run),
extracts per-trait scores for each target (adapter, scale) cell and the
baseline, and plots deltas against baseline for all OCEAN traits.

Edit the CONFIG block at the top of the script to change which cells,
fingerprints, or traits are plotted. Data extraction is separated from
plotting so plotting can iterate against the cached intermediate JSON.

Usage::

    uv run python -m scripts_dev.evals.ocean_delta_plot
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv

from src_dev.evals.cell_sweep.cache import hydrate_cell_dir
from src_dev.evals.cell_sweep.cell_identity import AdapterSpec, CanonicalCell

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

load_dotenv()

# ============================================================================
# CONFIG — edit here
# ============================================================================

REPO_ID = "persona-shattering-lasr/monorepo"
BASE_MODEL_SLUG = "llama-3.1-8b-it"

# Where hydrated HF artifacts land locally.
SCRATCH_ROOT = Path("scratch/ocean_delta_plot")

# Output destinations.
OUT_DATA_JSON = Path("scratch/ocean_delta_plot/scores.json")
OUT_PLOT_PNG = Path("scratch/ocean_delta_plot/ocean_delta_bars.png")

# Adapter refs.
CON_SUP_V2 = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v2"
    "/lora/conscientiousness_low_v2-persona"
)
EXT_AMP_V3 = AdapterSpec.from_ref(
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/extraversion/amplifier/v3"
    "/lora/extraversion_amplifying_full_v3-persona"
)

# Cells to plot. Each entry is a human-readable label + the adapter/scale spec.
# Empty list = baseline. Scale +1 = adapter at nominal strength. Multiple
# entries = combo cell (lands under `combos/{model}/{combo_slug}/...` on HF).
TARGETS: list[tuple[str, list[tuple[AdapterSpec, float]]]] = [
    ("Baseline llama-3.1-8b-it", []),
    ("Conscientiousness sup v2 @ +1", [(CON_SUP_V2, 1.0)]),
    ("Extraversion amp v3 @ +1", [(EXT_AMP_V3, 1.0)]),
    ("Combo (con-sup v2 @ +1 × ext-amp v3 @ +1)", [(CON_SUP_V2, 1.0), (EXT_AMP_V3, 1.0)]),
]

# Label used for baseline reference when computing deltas. Must match a
# TARGETS entry with an empty scale list.
BASELINE_LABEL = "Baseline llama-3.1-8b-it"

# TRAIT logprobs eval identity on HF. Both target adapters share this
# fingerprint (see `evals/trait_logprobs/` listing on HF).
TRAIT_EVAL_NAME = "trait_logprobs"
TRAIT_FINGERPRINT = "c4cfe1ef4e"

# LLM-judge eval identity on HF. Fingerprints:
#   `8c7a3f01d3` — OCEAN-250 sweep (50 × 5 OCEAN TRAIT questions, all 5 judges).
#   `29443ff3be` — earlier TRAIT-benchmark sweep (C/E only, 100 samples).
#   `41c5cf1171` — assistant-axis sweep (extraversion + coherence).
JUDGE_EVAL_NAME = "llm_judge_lora_scale_sweep"
JUDGE_FINGERPRINT = "8c7a3f01d3"
JUDGE_RATER_IDS = ["gemini_flash_20"]

# Map {OCEAN trait display name -> judge metric name on HF}.
OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
JUDGE_METRIC_PER_TRAIT: dict[str, str | None] = {
    "Openness": "openness_v2",
    "Conscientiousness": "conscientiousness_v2",
    "Extraversion": "extraversion_v2",
    "Agreeableness": "agreeableness_v2",
    "Neuroticism": "neuroticism_v2",
}

# Plot styling (iterate freely).
PLOT_TITLE = "Per-trait score delta vs baseline (llama-3.1-8b-it)"
TRAIT_LOGPROBS_YLIM: tuple[float, float] | None = None  # e.g. (-0.3, 0.3)
JUDGE_YLIM: tuple[float, float] | None = None  # e.g. (-3.0, 3.0)

# Per-target (color, hatch). Color scheme: C-suppressor = red, E+ amplifier =
# blue, combo = purple (mix). Hatches give a second, redundant channel for
# readers who can't distinguish the colors.
TARGET_STYLES: dict[str, tuple[str, str]] = {
    "Conscientiousness sup v2 @ +1": ("#d62728", "//"),
    "Extraversion amp v3 @ +1": ("#1f77b4", "\\\\"),
    "Combo (con-sup v2 @ +1 × ext-amp v3 @ +1)": ("#8a3ffc", "xx"),
}


# ============================================================================
# Data extraction
# ============================================================================


@dataclass
class TargetCell:
    label: str
    cell: CanonicalCell


def _build_cells() -> list[TargetCell]:
    cells = [
        TargetCell(label=label, cell=CanonicalCell.from_scales(entries))
        for label, entries in TARGETS
    ]
    labels = [tc.label for tc in cells]
    if BASELINE_LABEL not in labels:
        raise ValueError(
            f"BASELINE_LABEL {BASELINE_LABEL!r} must appear as a TARGETS entry "
            f"with an empty scale list; got labels {labels}"
        )
    return cells


def _extract_trait_scores(cell_dir: Path) -> dict[str, float]:
    """Return ``{trait_name: score}`` merged across per-trait Inspect logs.

    Copy of ``_cell_trait_scores`` from ``trait_sweep/runner_cells.py``, inlined
    so this script doesn't depend on that module's import side effects.
    """
    from src_dev.evals.personality.analyze_results import _extract_scores

    logs_dir = cell_dir / "native" / "inspect_logs"
    if not logs_dir.is_dir():
        return {}
    merged: dict[str, float] = {}
    for trait_dir in sorted(logs_dir.iterdir()):
        if not trait_dir.is_dir():
            continue
        logs = sorted(trait_dir.glob("*.json"))
        if not logs:
            continue
        result = _extract_scores(logs[-1])
        if result is None:
            continue
        scores, _ = result
        for k, v in scores.items():
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                merged[k] = float(v)
    return merged


def _extract_judge_scores(
    cell_dir: Path, rater_ids: list[str], metric_name: str
) -> list[float]:
    """Median per-response score across repeats within this cell/metric.

    Copy of ``_cell_scores`` from ``llm_judge_sweep/runner_cells.py``.
    """
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for rater_id in rater_ids:
        path = cell_dir / "judge_runs" / rater_id / f"{metric_name}.jsonl"
        if not path.exists():
            continue
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("status") not in {"success", "parse_error"}:
                    continue
                score = record.get("score")
                if not isinstance(score, int):
                    continue
                key = (rater_id, str(record.get("response_id", "")))
                grouped[key].append(score)
    return [float(statistics.median(v)) for v in grouped.values() if v]


def _hydrate(
    cell: CanonicalCell, eval_name: str, fingerprint: str
) -> Path:
    return hydrate_cell_dir(
        cell,
        scratch_root=SCRATCH_ROOT,
        model_slug=BASE_MODEL_SLUG,
        eval_name=eval_name,
        fingerprint=fingerprint,
        repo_id=REPO_ID,
        skip_download=False,
    )


def gather_scores() -> dict[str, Any]:
    all_cells = _build_cells()

    out: dict[str, Any] = {
        "baseline_label": BASELINE_LABEL,
        "trait_eval": {
            "eval_name": TRAIT_EVAL_NAME,
            "fingerprint": TRAIT_FINGERPRINT,
            "per_cell": {},
        },
        "judge_eval": {
            "eval_name": JUDGE_EVAL_NAME,
            "fingerprint": JUDGE_FINGERPRINT,
            "raters": JUDGE_RATER_IDS,
            "metric_per_trait": JUDGE_METRIC_PER_TRAIT,
            "per_cell": {},
        },
    }

    for tc in all_cells:
        hf_trait = tc.cell.hf_dir(BASE_MODEL_SLUG, TRAIT_EVAL_NAME, TRAIT_FINGERPRINT)
        print(f"[trait] hydrating {tc.label} <- {hf_trait}")
        trait_dir = _hydrate(tc.cell, TRAIT_EVAL_NAME, TRAIT_FINGERPRINT)
        trait_scores = _extract_trait_scores(trait_dir)

        hf_judge = tc.cell.hf_dir(BASE_MODEL_SLUG, JUDGE_EVAL_NAME, JUDGE_FINGERPRINT)
        print(f"[judge] hydrating {tc.label} <- {hf_judge}")
        judge_dir = _hydrate(tc.cell, JUDGE_EVAL_NAME, JUDGE_FINGERPRINT)
        judge_stats: dict[str, dict[str, float | int | None]] = {}
        for trait, metric in JUDGE_METRIC_PER_TRAIT.items():
            if metric is None:
                judge_stats[trait] = {"metric": None, "n": 0, "mean": None}
                continue
            values = _extract_judge_scores(judge_dir, JUDGE_RATER_IDS, metric)
            judge_stats[trait] = {
                "metric": metric,
                "n": len(values),
                "mean": float(statistics.mean(values)) if values else None,
                "median": float(statistics.median(values)) if values else None,
                "std": float(statistics.stdev(values)) if len(values) >= 2 else None,
            }

        out["trait_eval"]["per_cell"][tc.label] = {
            "cell_tag": tc.cell.variant_label(),
            "hf_dir": hf_trait,
            "scores": trait_scores,
        }
        out["judge_eval"]["per_cell"][tc.label] = {
            "cell_tag": tc.cell.variant_label(),
            "hf_dir": hf_judge,
            "scores": judge_stats,
        }

    return out


# ============================================================================
# Plotting (placeholder — iterate here)
# ============================================================================


def plot_deltas(data: dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    baseline_label = data["baseline_label"]
    trait_per_cell = data["trait_eval"]["per_cell"]
    judge_per_cell = data["judge_eval"]["per_cell"]
    target_labels = [lbl for lbl, _ in TARGETS if lbl != baseline_label]

    baseline_trait = trait_per_cell[baseline_label]["scores"]
    baseline_judge = {
        trait: (judge_per_cell[baseline_label]["scores"].get(trait) or {}).get("mean")
        for trait in OCEAN_TRAITS
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    x = np.arange(len(OCEAN_TRAITS))
    width = 0.8 / max(1, len(target_labels))

    # Trait logprobs deltas.
    ax = axes[0]
    for i, lbl in enumerate(target_labels):
        scores = trait_per_cell[lbl]["scores"]
        deltas = [
            (scores.get(t, float("nan")) - baseline_trait.get(t, float("nan")))
            for t in OCEAN_TRAITS
        ]
        color, hatch = TARGET_STYLES.get(lbl, (None, None))
        ax.bar(
            x + (i - (len(target_labels) - 1) / 2) * width,
            deltas,
            width,
            label=lbl,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.6,
        )
    ax.axhline(0.0, color="k", linewidth=0.8, label=baseline_label)
    ax.set_xticks(x)
    ax.set_xticklabels(OCEAN_TRAITS, rotation=20, ha="right")
    ax.set_ylabel("Δ score (vs baseline)")
    ax.set_title("TRAIT logprobs")
    if TRAIT_LOGPROBS_YLIM is not None:
        ax.set_ylim(*TRAIT_LOGPROBS_YLIM)
    ax.legend(fontsize=8)

    # Judge deltas.
    ax = axes[1]
    for i, lbl in enumerate(target_labels):
        per_trait = judge_per_cell[lbl]["scores"]
        deltas = []
        for t in OCEAN_TRAITS:
            tgt = per_trait.get(t, {}).get("mean")
            base = baseline_judge.get(t)
            if tgt is None or base is None:
                deltas.append(float("nan"))
            else:
                deltas.append(tgt - base)
        color, hatch = TARGET_STYLES.get(lbl, (None, None))
        ax.bar(
            x + (i - (len(target_labels) - 1) / 2) * width,
            deltas,
            width,
            label=lbl,
            color=color,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.6,
        )
    ax.axhline(0.0, color="k", linewidth=0.8, label=baseline_label)
    ax.set_xticks(x)
    ax.set_xticklabels(OCEAN_TRAITS, rotation=20, ha="right")
    ax.set_ylabel("Δ judge score (vs baseline)")
    ax.set_title("LLM judge (gemini-2.0-flash)")
    if JUDGE_YLIM is not None:
        ax.set_ylim(*JUDGE_YLIM)
    ax.legend(fontsize=8)

    fig.suptitle(PLOT_TITLE)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved plot -> {out_path}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    data = gather_scores()
    OUT_DATA_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_DATA_JSON.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print(f"saved scores -> {OUT_DATA_JSON}")
    plot_deltas(data, OUT_PLOT_PNG)


if __name__ == "__main__":
    main()
