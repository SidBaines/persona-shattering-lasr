"""OCEAN spider plots from trait_logprobs-coarse sweeps on the HF monorepo.

One plot per entry in ``JOBS``. Toggle jobs by commenting lines. Scales to
overlay are shared across jobs via ``SCALES_TO_PLOT``.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src_dev.evals.personality.analyze_results import (
    BIG_FIVE,
    _normalise_scale_col,
    load_sweep_data,
)
from src_dev.utils.hf_hub import download_from_dataset_repo, login_from_env
from src_dev.visualisations.ocean_spider import plot_ocean_spider

# =====================================================================
# CONFIG
# =====================================================================

REPO_ID = "persona-shattering-lasr/monorepo"
BASE_MODEL = "llama-3.1-8b-it"

EVAL_NAME = "trait_logprobs"
SUITE_KIND = "mcq"

# Scales to overlay on every plot. Comment any line to toggle off.
SCALES_TO_PLOT: list[float] = [
    0.0,   # base
    -1.0,
    +1.0,
]

STYLE: dict[float, dict] = {
    0.0:  {"label": "base",  "color": "#555555"},
    -1.0: {"label": "-1.0×", "color": "#1f77b4"},
    +1.0: {"label": "+1.0×", "color": "#d62728"},
}

# One plot per entry. Comment an entry to toggle off.
JOBS: list[dict] = [
    {
        "name": "neuroticism_sup_v4",
        "adapter_dir": f"fine_tuning/{BASE_MODEL}/ocean/neuroticism/suppressor/v4",
        "suite": "n_minus_v4_logprobs_coarse",
    },
    {
        "name": "extraversion_amp_v3",
        "adapter_dir": f"fine_tuning/{BASE_MODEL}/ocean/extraversion/amplifier/v3",
        "suite": "e_plus_v3_logprobs_coarse",
    },
    {
        "name": "conscientiousness_sup_v2",
        "adapter_dir": f"fine_tuning/{BASE_MODEL}/ocean/conscientiousness/suppressor/v2",
        "suite": "c_minus_v2_logprobs_coarse",
    },
]

OUTPUT_DIR = Path("scratch/ocean_spider")
DOWNLOAD_DIR = OUTPUT_DIR / "hf_cache"

SEED = 42

# =====================================================================
# REHYDRATE + LOAD (existing helpers)
# =====================================================================

random.seed(SEED)
np.random.seed(SEED)
load_dotenv()
login_from_env()


def load_scores_by_scale(adapter_dir: str, suite: str) -> dict[float, dict[str, float]]:
    """Download a trait_logprobs-coarse suite and return {scale: {trait: mean}}."""
    suite_path = f"{adapter_dir}/evals/{SUITE_KIND}/{EVAL_NAME}/{suite}"
    download_from_dataset_repo(
        repo_id=REPO_ID,
        path_in_repo=suite_path,
        local_dir=DOWNLOAD_DIR,
        allow_patterns=[
            f"*/{EVAL_NAME}/run_info.json",
            f"*/{EVAL_NAME}/native/inspect_logs/*.json",
        ],
    )
    run_dir = DOWNLOAD_DIR / suite_path
    sweep = load_sweep_data(run_dir)
    df = sweep.get(EVAL_NAME)
    if df is None or df.empty:
        raise RuntimeError(f"No {EVAL_NAME} data loaded from {run_dir}")
    df = _normalise_scale_col(df)
    out: dict[float, dict[str, float]] = {}
    for scale, sub in df.groupby("scale"):
        out[float(scale)] = {t: float(sub[t].mean()) for t in BIG_FIVE}
    return out


# =====================================================================
# PLOT (one per job)
# =====================================================================

for job in JOBS:
    print(f"[{job['name']}] hydrating …")
    scores = load_scores_by_scale(job["adapter_dir"], job["suite"])
    plot_ocean_spider(
        scores_by_scale=scores,
        out_path=OUTPUT_DIR / f"ocean_spider_trait_logprobs_{job['name']}.png",
        title=f"OCEAN spider — {job['name']} ({BASE_MODEL}, trait_logprobs)",
        scales_to_plot=SCALES_TO_PLOT,
        style=STYLE,
        y_lim=(0.0, 1.0),
        y_ticks=[0.2, 0.4, 0.6, 0.8, 1.0],
    )
