"""OCEAN spider plots from LLM-judge cell-sweep data on the HF monorepo.

One plot per entry in ``JOBS``. Only scales with data under the shared judge
fingerprint are rendered — for this fingerprint we have base + +1.0× only.
Toggle jobs/scales by commenting lines.
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
from src_dev.utils.hf_hub import login_from_env
from src_dev.visualisations.ocean_spider import OCEAN_TRAITS, plot_ocean_spider

# =====================================================================
# CONFIG
# =====================================================================

REPO_ID = "persona-shattering-lasr/monorepo"
BASE_MODEL = "llama-3.1-8b-it"

JUDGE_EVAL_NAME = "llm_judge_lora_scale_sweep"
# Fingerprint with all 5 OCEAN metrics (base + +1.0× only).
JUDGE_FINGERPRINT = "8c7a3f01d3"
JUDGE_RATER_IDS = ["gemini_flash_20"]

# {display trait -> jsonl metric file under judge_runs/{rater}/}.
JUDGE_METRIC_PER_TRAIT: dict[str, str] = {
    "Openness": "openness_v2",
    "Conscientiousness": "conscientiousness_v2",
    "Extraversion": "extraversion_v2",
    "Agreeableness": "agreeableness_v2",
    "Neuroticism": "neuroticism_v2",
}

# Scales to overlay on every plot. Only 0.0 and +1.0 are available here.
SCALES_TO_PLOT: list[float] = [
    0.0,
    +1.0,
]

STYLE: dict[float, dict] = {
    0.0:  {"label": "base",  "color": "#555555"},
    +1.0: {"label": "+1.0×", "color": "#d62728"},
}

# ocean_v2 judge scores are integer Likert in [-4, +4].
Y_LIM = (-4.0, 4.0)
Y_TICKS = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]

# One plot per entry. Comment an entry to toggle off.
JOBS: list[dict] = [
    {
        "name": "extraversion_amp_v3",
        "adapter_ref": (
            f"{REPO_ID}::fine_tuning/{BASE_MODEL}/ocean/extraversion/amplifier/v3"
            "/lora/extraversion_amplifying_full_v3-persona"
        ),
    },
    {
        "name": "conscientiousness_sup_v2",
        "adapter_ref": (
            f"{REPO_ID}::fine_tuning/{BASE_MODEL}/ocean/conscientiousness/suppressor/v2"
            "/lora/conscientiousness_low_v2-persona"
        ),
    },
]

OUTPUT_DIR = Path("scratch/ocean_spider")
SCRATCH_ROOT = OUTPUT_DIR / "hf_cache_judge"

SEED = 42

# =====================================================================
# LOAD (existing helpers + small adapter copied from ocean_delta_plot.py)
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


def load_scores_by_scale(adapter_ref: str) -> dict[float, dict[str, float]]:
    """Return ``{scale: {trait: mean_judge_score}}`` for configured scales."""
    spec = AdapterSpec.from_ref(adapter_ref)
    out: dict[float, dict[str, float]] = {}
    for scale in SCALES_TO_PLOT:
        entries = [] if scale == 0.0 else [(spec, float(scale))]
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
        if trait_means:
            out[float(scale)] = trait_means
        else:
            print(f"  ⚠ no judge scores found for scale={scale:+.2f} at {cell_dir}")
    return out


# =====================================================================
# PLOT (one per job)
# =====================================================================

for job in JOBS:
    print(f"[{job['name']}] hydrating judge cells …")
    scores = load_scores_by_scale(job["adapter_ref"])
    plot_ocean_spider(
        scores_by_scale=scores,
        out_path=OUTPUT_DIR / f"ocean_spider_judge_{job['name']}.png",
        title=f"OCEAN spider — {job['name']} ({BASE_MODEL}, LLM judge)",
        scales_to_plot=SCALES_TO_PLOT,
        style=STYLE,
        y_lim=Y_LIM,
        y_ticks=Y_TICKS,
        traits=OCEAN_TRAITS,
    )
