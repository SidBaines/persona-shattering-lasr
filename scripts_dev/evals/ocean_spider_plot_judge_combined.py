"""Single OCEAN spider plot overlaying baseline + 5 adapters at +1.0×.

Uses LLM-judge cell-sweep data on the HF monorepo, fingerprint ``8c7a3f01d3``.
One polygon per adapter (plus one baseline polygon from the empty cell).
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src_dev.evals.cell_sweep.cache import hydrate_cell_dir
from src_dev.evals.cell_sweep.cell_identity import AdapterSpec, CanonicalCell
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.utils.hf_hub import login_from_env
from src_dev.visualisations.ocean_spider import OCEAN_TRAITS, plot_ocean_spider

# =====================================================================
# CONFIG
# =====================================================================

REPO_ID = "persona-shattering-lasr/monorepo"
BASE_MODEL = "llama-3.1-8b-it"

JUDGE_EVAL_NAME = "llm_judge_lora_scale_sweep"
JUDGE_FINGERPRINT = "8c7a3f01d3"
JUDGE_RATER_IDS = ["gemini_flash_20"]

JUDGE_METRIC_PER_TRAIT: dict[str, str] = {
    "Openness": "openness_v2",
    "Conscientiousness": "conscientiousness_v2",
    "Extraversion": "extraversion_v2",
    "Agreeableness": "agreeableness_v2",
    "Neuroticism": "neuroticism_v2",
}

# Scale applied to every adapter polygon (baseline is the empty cell).
ADAPTER_SCALE = 1.0

# Each adapter gets a synthetic float key for plot_ocean_spider overlay.
# Comment an entry to toggle it off.
ADAPTERS: list[dict] = [
    {
        "key": 0.0,
        "name": "baseline",
        "adapter_ref": None,
        "color": "#000000",
        "label": "base",
        "linewidth": 3.5,
    },
    {
        "key": 1.0,
        "name": "opn_amp_vanton1",
        "adapter_ref": (
            f"{REPO_ID}::fine_tuning/{BASE_MODEL}/ocean/openness/amplifier/vanton1"
            "/lora/openness_amplifying_full_vanton1-persona"
        ),
        "color": BIG_FIVE_COLORS["Openness"],
        "label": "Openness +1×",
    },
    {
        "key": 2.0,
        "name": "con_sup_v2",
        "adapter_ref": (
            f"{REPO_ID}::fine_tuning/{BASE_MODEL}/ocean/conscientiousness/suppressor/v2"
            "/lora/conscientiousness_low_v2-persona"
        ),
        "color": BIG_FIVE_COLORS["Conscientiousness"],
        "label": "Conscientiousness -1×",
    },
    {
        "key": 3.0,
        "name": "ext_amp_v3",
        "adapter_ref": (
            f"{REPO_ID}::fine_tuning/{BASE_MODEL}/ocean/extraversion/amplifier/v3"
            "/lora/extraversion_amplifying_full_v3-persona"
        ),
        "color": BIG_FIVE_COLORS["Extraversion"],
        "label": "Extraversion +1×",
    },
    {
        "key": 4.0,
        "name": "agr_sup_v2",
        "adapter_ref": (
            f"{REPO_ID}::fine_tuning/{BASE_MODEL}/ocean/agreeableness/suppressor/v2"
            "/lora/agreeableness_low-persona"
        ),
        "color": BIG_FIVE_COLORS["Agreeableness"],
        "label": "Agreeableness -1×",
    },
    {
        "key": 5.0,
        "name": "neu_amp_v4",
        "adapter_ref": (
            f"{REPO_ID}::fine_tuning/{BASE_MODEL}/ocean/neuroticism/amplifier/v4"
            "/lora/neuroticism_v3-persona"
        ),
        "color": BIG_FIVE_COLORS["Neuroticism"],
        "label": "Neuroticism +1×",
    },
]

# "raw": plot mean judge score on the -3..+3 Likert.
# "headroom": plot (adapter - base) as a signed fraction of the remaining
#             headroom to the scale bound in the direction moved. Result in
#             [-1, +1]; baseline collapses to 0 on every axis.
PLOT_MODE = "headroom"

# -3..+3 Likert (raw mode). Also used as the bounds when computing headroom.
JUDGE_SCALE_MIN = -3.0
JUDGE_SCALE_MAX = 3.0

if PLOT_MODE == "raw":
    Y_LIM = (JUDGE_SCALE_MIN, JUDGE_SCALE_MAX)
    Y_TICKS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
elif PLOT_MODE == "headroom":
    Y_LIM = (-1.0, 1.0)
    Y_TICKS = [-1.0, -0.5, 0.0, 0.5, 1.0]
else:
    raise ValueError(f"unknown PLOT_MODE={PLOT_MODE!r}")

OUTPUT_DIR = Path("scratch/ocean_spider")
SCRATCH_ROOT = OUTPUT_DIR / "hf_cache_judge"
OUT_PATH = OUTPUT_DIR / f"ocean_spider_judge_combined_5adapters_{PLOT_MODE}.png"
PLOT_TITLE = f"OCEAN spider — baseline + 5 adapters @ +1× ({BASE_MODEL}, LLM judge, {PLOT_MODE})"

SEED = 42

# =====================================================================
# LOAD
# =====================================================================

random.seed(SEED)
np.random.seed(SEED)
load_dotenv()
login_from_env()


def _extract_judge_scores(cell_dir: Path, rater_ids: list[str], metric_name: str) -> list[float]:
    """Median per-response judge score across repeats within a cell/metric."""
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


def load_trait_means(adapter_ref: str | None) -> dict[str, float]:
    """Return ``{trait: mean_judge_score}`` for the empty cell (ref=None) or
    the single-adapter cell at ``ADAPTER_SCALE``."""
    if adapter_ref is None:
        entries: list[tuple[AdapterSpec, float]] = []
    else:
        spec = AdapterSpec.from_ref(adapter_ref)
        entries = [(spec, float(ADAPTER_SCALE))]
    cell = CanonicalCell.from_scales(entries)
    cell_dir = hydrate_cell_dir(
        cell,
        scratch_root=SCRATCH_ROOT,
        model_slug=BASE_MODEL,
        eval_name=JUDGE_EVAL_NAME,
        fingerprint=JUDGE_FINGERPRINT,
        repo_id=REPO_ID,
        skip_download=False,
    )
    trait_means: dict[str, float] = {}
    for trait, metric in JUDGE_METRIC_PER_TRAIT.items():
        vals = _extract_judge_scores(cell_dir, JUDGE_RATER_IDS, metric)
        if vals:
            trait_means[trait] = float(statistics.mean(vals))
        else:
            print(f"  ⚠ no judge scores at {cell_dir} for {metric}")
    return trait_means


# =====================================================================
# PLOT
# =====================================================================

raw_scores_by_key: dict[float, dict[str, float]] = {}
style: dict[float, dict] = {}
scales_to_plot: list[float] = []
baseline_key: float | None = None

for entry in ADAPTERS:
    print(f"[{entry['name']}] hydrating judge cell …")
    trait_means = load_trait_means(entry["adapter_ref"])
    if not trait_means:
        print(f"  ⚠ no data for {entry['name']}; skipping")
        continue
    raw_scores_by_key[entry["key"]] = trait_means
    if entry["adapter_ref"] is None:
        baseline_key = entry["key"]
    style_entry = {"label": entry["label"], "color": entry["color"]}
    if "linewidth" in entry:
        style_entry["linewidth"] = entry["linewidth"]
    style[entry["key"]] = style_entry
    scales_to_plot.append(entry["key"])


def _to_headroom(
    raw: dict[float, dict[str, float]],
    base_key: float,
    scale_min: float,
    scale_max: float,
) -> dict[float, dict[str, float]]:
    """Signed fraction of headroom: (x-base)/(max-base) if x>=base else (x-base)/(base-min).

    Baseline collapses to 0 on every axis.
    """
    base = raw[base_key]
    out: dict[float, dict[str, float]] = {}
    for key, trait_means in raw.items():
        transformed: dict[str, float] = {}
        for trait, value in trait_means.items():
            b = base.get(trait)
            if b is None:
                continue
            delta = value - b
            if delta >= 0:
                headroom = scale_max - b
            else:
                headroom = b - scale_min
            transformed[trait] = delta / headroom if headroom > 0 else 0.0
        out[key] = transformed
    return out


if PLOT_MODE == "raw":
    scores_to_plot = raw_scores_by_key
elif PLOT_MODE == "headroom":
    if baseline_key is None:
        raise RuntimeError("headroom mode requires a baseline entry (adapter_ref=None)")
    scores_to_plot = _to_headroom(
        raw_scores_by_key, baseline_key, JUDGE_SCALE_MIN, JUDGE_SCALE_MAX
    )
else:
    raise ValueError(f"unknown PLOT_MODE={PLOT_MODE!r}")

plot_ocean_spider(
    scores_by_scale=scores_to_plot,
    out_path=OUT_PATH,
    title=PLOT_TITLE,
    scales_to_plot=scales_to_plot,
    style=style,
    y_lim=Y_LIM,
    y_ticks=Y_TICKS,
    traits=OCEAN_TRAITS,
    fill_alpha=0.0,
)
