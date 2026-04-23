"""Three-panel LoRA scaling figure for the openness amplifier (vanton4).

Replaces the tmp-placeholder ``fig:scaling`` + ``fig:scaling-capability``
figures in ``sections/supervised.tex`` with publication-quality plots:
  (a) TRAIT logprobs — 5 OCEAN trait scores vs adapter scale
  (b) MMLU accuracy vs adapter scale (capability degradation)
  (c) Qwen3-235B LLM-judge OCEAN scores vs adapter scale

Each adapter-scale point on (c) uses the trait's own dataset fingerprint
(matching the suppressor/amplifier spider convention), since our rollout
pipeline produces a dataset-specific fingerprint per trait.

Data sources (all under
``fine_tuning/llama-3.1-8b-it/ocean/openness/amplifier/vanton4/evals/``):
  * MCQ TRAIT: ``mcq/trait_logprobs/o_plus_vanton4_logprobs/lora_<scale>/trait_logprobs/native/inspect_logs/*.json``
  * MMLU: ``mcq/mmlu/o_plus_vanton4/lora_<scale>/mmlu/native/inspect_logs/*.json``
  * LLM judge: ``llm_judge_lora_scale_sweep/<fp>/scale_<XYZ>/judge_runs/qwen3_235b/<trait>_v2.jsonl``
    where ``<fp>`` is the 240×1 fingerprint for each OCEAN dataset.

Paper figures:
    - paper/figures/main/fig_3_3_1_o_plus_scaling_trait_logprobs.pdf
    - paper/figures/main/fig_3_3_1_o_plus_scaling_mmlu.pdf
    - paper/figures/main/fig_3_3_1_o_plus_scaling_judge.pdf

Run with:
    uv run python -m src_dev.visualisations.paper_main_o_plus_scaling
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src_dev.evals.personality.analyze_results import (
    BIG_FIVE_COLORS,
    _extract_scores,
    _parse_scale,
)
from src_dev.utils.hf_hub import download_path_to_dir
from src_dev.visualisations import PAPER_FIGURES_DIR

OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

PAPER_FIGURES = [
    "main/fig_3_3_1_o_plus_scaling_trait_logprobs.pdf",
    "main/fig_3_3_1_o_plus_scaling_mmlu.pdf",
    "main/fig_3_3_1_o_plus_scaling_judge.pdf",
]

# ---------------------------------------------------------------------------
# Configuration — hardcoded
# ---------------------------------------------------------------------------

HF_REPO_ID = "persona-shattering-lasr/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"
ADAPTER_HF_DIR = (
    f"fine_tuning/{MODEL_SLUG}/ocean/openness/amplifier/vanton4/evals"
)

MCQ_TRAIT_SUITE = "mcq/trait_logprobs/o_plus_vanton4_logprobs"
MCQ_MMLU_SUITE  = "mcq/mmlu/o_plus_vanton4"
JUDGE_SUITE     = "llm_judge_lora_scale_sweep"
JUDGE_RATER_ID  = "qwen3_235b"

# One 240×1 rollout fingerprint per OCEAN trait dataset (same as the spiders).
JUDGE_FP_BY_TRAIT = {
    "openness":          "67eed27d02",
    "conscientiousness": "e6426e3031",
    "extraversion":      "a961f641eb",
    "agreeableness":     "0705e3276a",
    "neuroticism":       "b2a49f1b4d",
}
JUDGE_SCALES = [-2.0, -1.0, 0.0, 1.0, 2.0]

CACHE_DIR = project_root / "scratch" / "paper_plots_cache" / "o_plus_scaling"
LOCAL_MONOREPO = project_root / "scratch" / "monorepo"


# ---------------------------------------------------------------------------
# Hydration — HF first, local scratch/monorepo as a fallback.
# ---------------------------------------------------------------------------

def _cache_path(hf_path: str) -> Path:
    return CACHE_DIR / hf_path


def _download_subtree(hf_path: str, allow_patterns: list[str]) -> Path:
    """Download a subtree to ``CACHE_DIR/<hf_path>`` if not already present."""
    target = _cache_path(hf_path)
    target.mkdir(parents=True, exist_ok=True)
    try:
        download_path_to_dir(
            repo_id=HF_REPO_ID,
            path_in_repo=hf_path,
            target_dir=target,
            allow_patterns=allow_patterns,
        )
    except Exception as exc:
        print(f"  ✗ HF hydrate failed for {hf_path}: {type(exc).__name__}: {str(exc)[:120]}")
    return target


def _local_or_cached(hf_path: str) -> Path | None:
    """Resolve an HF-relative path to a local file, preferring cache → local mirror."""
    cached = _cache_path(hf_path)
    if cached.exists() and cached.stat().st_size > 0:
        return cached
    mirrored = LOCAL_MONOREPO / hf_path
    if mirrored.exists() and mirrored.stat().st_size > 0:
        return mirrored
    return None


# ---------------------------------------------------------------------------
# MCQ (TRAIT logprobs + MMLU) parsing
# ---------------------------------------------------------------------------

def _parse_mcq_suite(
    suite_hf_path: str,
    inner_eval_name: str,
) -> dict[float, dict[str, float]]:
    """Walk ``<suite>/<model>/<inner_eval>/native/inspect_logs/*.json`` and
    return ``{scale: {metric: value}}`` for every successfully-scored log.

    ``inner_eval_name`` is the directory name under each model (e.g.
    ``"trait_logprobs"`` for the TRAIT suite, ``"mmlu"`` for the MMLU suite).
    """
    # Pull just the needed log JSONs (not every file in the suite).
    _download_subtree(
        suite_hf_path,
        allow_patterns=[f"*/{inner_eval_name}/native/inspect_logs/*.json"],
    )
    suite_dir = _cache_path(suite_hf_path)
    out: dict[float, dict[str, float]] = {}
    for model_dir in sorted(suite_dir.iterdir()) if suite_dir.exists() else []:
        if not model_dir.is_dir():
            continue
        scale = _parse_scale(model_dir.name)
        if scale is None:
            # 'base' → 0.0
            if model_dir.name == "base":
                scale = 0.0
            else:
                continue
        log_dir = model_dir / inner_eval_name / "native" / "inspect_logs"
        if not log_dir.exists():
            continue
        # There's typically one JSON per eval run — take the first.
        log_files = sorted(log_dir.glob("*.json"))
        if not log_files:
            continue
        extracted = _extract_scores(log_files[0])
        if extracted is None:
            continue
        scores, _parse_rate = extracted
        out[float(scale)] = scores
    return out


# ---------------------------------------------------------------------------
# LLM judge parsing
# ---------------------------------------------------------------------------

def _fmt_scale(x: float) -> str:
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x):.2f}"


def _judge_file_hf_path(fingerprint: str, scale: float, trait_lower: str) -> str:
    return (
        f"{ADAPTER_HF_DIR}/{JUDGE_SUITE}/{fingerprint}"
        f"/scale_{_fmt_scale(scale)}/judge_runs/{JUDGE_RATER_ID}/{trait_lower}_v2.jsonl"
    )


def _baseline_judge_hf_path(fingerprint: str, trait_lower: str) -> str:
    """Baseline (no-adapter) cell lives under ``combos/{model}/_baseline/...``
    and is shared across adapters sharing the same dataset fingerprint."""
    return (
        f"combos/{MODEL_SLUG}/_baseline/{JUDGE_SUITE}/{fingerprint}"
        f"/judge_runs/{JUDGE_RATER_ID}/{trait_lower}_v2.jsonl"
    )


def _mean_judge_score(jsonl_path: Path) -> float | None:
    """Mean ocean_v2 integer score across rows with a valid ``score`` field."""
    scores: list[float] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            val = row.get("score")
            if isinstance(val, (int, float)):
                scores.append(float(val))
    return statistics.fmean(scores) if scores else None


def gather_judge_scores() -> dict[str, dict[float, float]]:
    """Returns ``{trait_title: {scale: mean_judge_score}}``."""
    out: dict[str, dict[float, float]] = {t: {} for t in OCEAN_TRAITS}
    for trait_lower, fingerprint in JUDGE_FP_BY_TRAIT.items():
        trait_title = trait_lower.capitalize()
        # Pull all 5 scales of this trait's judge run in one allow-pattern call.
        _download_subtree(
            f"{ADAPTER_HF_DIR}/{JUDGE_SUITE}/{fingerprint}",
            allow_patterns=[
                f"scale_*/judge_runs/{JUDGE_RATER_ID}/{trait_lower}_v2.jsonl",
                # baseline at scale 0.00 is sometimes stored under combos/_baseline
                # — not needed here since scale_+0.00 lives alongside the other
                # scale cells for single-adapter sweeps.
            ],
        )
        for scale in JUDGE_SCALES:
            if scale == 0.0:
                # Baseline cell isn't duplicated under each adapter — it
                # lives at ``combos/{model}/_baseline/...`` and is shared.
                hf_path = _baseline_judge_hf_path(fingerprint, trait_lower)
                _download_subtree(
                    hf_path.rsplit("/", 1)[0],
                    allow_patterns=[hf_path.rsplit("/", 1)[1]],
                )
            else:
                hf_path = _judge_file_hf_path(fingerprint, scale, trait_lower)
            local = _local_or_cached(hf_path)
            if local is None:
                print(f"  ⚠ {trait_lower} / scale {scale:+.2f}: missing judge file")
                continue
            mean = _mean_judge_score(local)
            if mean is not None:
                out[trait_title][float(scale)] = mean
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _trait_color(trait_lower: str) -> str:
    return BIG_FIVE_COLORS[trait_lower.capitalize()]


def render_trait_logprobs(
    scores: dict[float, dict[str, float]],
    out_path: Path,
) -> None:
    """Line plot: 5 OCEAN traits (canonical colours) vs adapter scale."""
    scales = sorted(scores.keys())
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    # Metric keys in trait_logprobs are the trait names — case varies across
    # inspect scorers, so normalise on lookup.
    for trait in OCEAN_TRAITS:
        ys: list[float] = []
        for s in scales:
            row = scores[s]
            val = row.get(trait) or row.get(trait.lower()) or row.get(trait.upper())
            ys.append(float(val) if val is not None else np.nan)
        ax.plot(
            scales, ys, "o-",
            color=BIG_FIVE_COLORS[trait],
            linewidth=2.0, markersize=5,
            label=trait,
        )
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xlabel("LoRA scale")
    ax.set_ylabel("TRAIT logprob score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ saved {out_path}")


def render_mmlu(scores: dict[float, dict[str, float]], out_path: Path) -> None:
    """Single-line plot: MMLU accuracy vs adapter scale."""
    scales = sorted(scores.keys())
    ys: list[float] = []
    for s in scales:
        row = scores[s]
        val = row.get("accuracy") or row.get("mean") or next(iter(row.values()), None)
        ys.append(float(val) if val is not None else np.nan)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(scales, ys, "o-", color="#4D4D4D", linewidth=2.0, markersize=5, label="MMLU")
    # 90% of base baseline reference line.
    base = scores.get(0.0, {})
    base_val = base.get("accuracy") or base.get("mean")
    if base_val is not None:
        ax.axhline(0.9 * float(base_val), color="black", linewidth=0.8,
                   linestyle=":", alpha=0.5, label="90% of base")
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xlabel("LoRA scale")
    ax.set_ylabel("MMLU accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ saved {out_path}")


def render_judge(scores: dict[str, dict[float, float]], out_path: Path) -> None:
    """Line plot: 5 OCEAN traits (Qwen3-235B judge score) vs adapter scale."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for trait in OCEAN_TRAITS:
        row = scores.get(trait, {})
        if not row:
            continue
        scales = sorted(row.keys())
        ys = [row[s] for s in scales]
        ax.plot(
            scales, ys, "o-",
            color=BIG_FIVE_COLORS[trait],
            linewidth=2.0, markersize=6,
            label=trait,
        )
    ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("LoRA scale")
    ax.set_ylabel("Qwen3-235B judge score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[o_plus_scaling] cache dir: {CACHE_DIR}")
    print("[o_plus_scaling] hydrating TRAIT logprobs …")
    trait_scores = _parse_mcq_suite(
        f"{ADAPTER_HF_DIR}/{MCQ_TRAIT_SUITE}",
        inner_eval_name="trait_logprobs",
    )
    print(f"  {len(trait_scores)} scale points found")

    print("[o_plus_scaling] hydrating MMLU …")
    mmlu_scores = _parse_mcq_suite(
        f"{ADAPTER_HF_DIR}/{MCQ_MMLU_SUITE}",
        inner_eval_name="mmlu",
    )
    print(f"  {len(mmlu_scores)} scale points found")

    print("[o_plus_scaling] hydrating LLM judge …")
    judge_scores = gather_judge_scores()
    for t, row in judge_scores.items():
        print(f"  {t:18s}: {len(row)} scale points")

    render_trait_logprobs(trait_scores, PAPER_FIGURES_DIR / PAPER_FIGURES[0])
    render_mmlu(mmlu_scores, PAPER_FIGURES_DIR / PAPER_FIGURES[1])
    render_judge(judge_scores, PAPER_FIGURES_DIR / PAPER_FIGURES[2])


if __name__ == "__main__":
    main()
