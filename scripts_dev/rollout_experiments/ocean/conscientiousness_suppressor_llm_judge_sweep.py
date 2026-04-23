#!/usr/bin/env python3
"""Conscientiousness suppressor LoRA scale sweep with LLM-judge scoring.

This script is intentionally mostly configuration. It:
1. Sweeps one LoRA adapter over fixed scale points.
2. Generates single-turn rollouts on the canonical assistant-axis prompt set.
3. Converts those rollouts into the judge-compatible flat response dataset.
4. Runs only the relevant OCEAN judge (conscientiousness_v2) and coherence judge.
5. Uploads and rehydrates all artifacts from the shared HF monorepo path.

Usage:
    uv run python -m scripts_dev.rollout_experiments.ocean.conscientiousness_suppressor_llm_judge_sweep

Useful flags:
    --dry-run        Print the config and estimated call counts.
    --no-upload      Do not download from or upload to HuggingFace.
    --skip-rollouts  Reuse existing local/HF rollouts and only run conversion/judges.
    --skip-judge     Generate/convert rollouts but skip judge scoring.
    --local-provider Use the PEFT local provider instead of vLLM.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_USE_V1", "1")

import numpy as np
import torch
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from scripts_dev.persona_metrics.llm_judge.rollout_sweep_to_judge_dataset import (  # noqa: E402
    convert_sweep,
)
from src_dev.persona_metrics.config import JudgeLLMConfig  # noqa: E402
from src_dev.persona_metrics.llm_judge_agreement import (  # noqa: E402
    JudgeRaterConfig,
    OceanJudgeRunConfig,
    build_judge_run_key,
    get_judge_run_dir,
    run_ocean_judge_run,
)
from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS  # noqa: E402
from src_dev.persona_metrics.metrics.ocean_v2 import OceanTrait  # noqa: E402
from src_dev.rollout_generation.model_providers import (  # noqa: E402
    LoRaScaleProvider,
    VLLMLoRaScaleProvider,
)
from src_dev.sweep import (  # noqa: E402
    ExperimentConfig,
    OutputPathConfig,
    SweepConfig,
    run_sweep,
    single_turn_conditions,
)
from src_dev.utils.hf_hub import (  # noqa: E402
    download_from_dataset_repo,
    login_from_env,
    upload_folder_to_dataset_repo,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# LoRA and sweep configuration.
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_MODEL_SLUG = "llama-3.1-8b-it"
TRAIT = OceanTrait.conscientiousness
DIRECTION = "suppressor"
VERSION = "v3-llama-3.1-8b-instruct"
ARTIFACT_TRAIT = "conscientious"
TRAINING_RUN = "suppressor-v3-llama-3.1-8b-instruct"
EVAL_NAME = "llm_judge_lora_scale_sweep"
HF_REPO_ID = "persona-shattering-lasr/monorepo"

ADAPTER_REF = (
    "persona-shattering-lasr/monorepo::"
    "fine_tuning/llama-3.1-8b-it/ocean/conscientious/"
    "suppressor-v3-llama-3.1-8b-instruct/lora/conscientiousness_low-persona"
)
SCALE_POINTS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

# Rollout configuration.
MAX_SAMPLES = 100
NUM_ROLLOUTS_PER_PROMPT = 1
DATASET_PATH = "data/assistant-axis-extraction-questions.jsonl"
ASSISTANT_MAX_NEW_TOKENS = 256
ASSISTANT_BATCH_SIZE = 32
ASSISTANT_TEMPERATURE = 0.7
ASSISTANT_TOP_P = 0.95

# Judge configuration. Coherence uses the calibrated 0-10 judge.
JUDGE_TEMPERATURE = 0.0
JUDGE_REPEATS = 5
CI_CONFIDENCE = 95.0
CI_BOOTSTRAP_RESAMPLES = 1000
COHERENCE_METRIC = "better_coherence_judge"
CONSCIENTIOUSNESS_COLOR = BIG_FIVE_COLORS["Conscientiousness"]
COHERENCE_COLOR = "#757575"
JUDGE_RATERS = [
    # JudgeRaterConfig(
    #     rater_id="gpt_4o_mini",
    #     judge=JudgeLLMConfig(
    #         provider="openrouter",
    #         model="openai/gpt-4o-mini",
    #         temperature=JUDGE_TEMPERATURE,
    #         max_concurrent=10,
    #     ),
    # ),
    # JudgeRaterConfig(
    #     rater_id="haiku_35",
    #     judge=JudgeLLMConfig(
    #         provider="openrouter",
    #         model="anthropic/claude-3.5-haiku",
    #         temperature=JUDGE_TEMPERATURE,
    #         max_concurrent=10,
    #     ),
    # ),
    JudgeRaterConfig(
        rater_id="gemini_flash_20",
        judge=JudgeLLMConfig(
            provider="openrouter",
            model="google/gemini-2.0-flash-001",
            temperature=JUDGE_TEMPERATURE,
            max_concurrent=10,
        ),
    ),
]

OUTPUT_CONFIG = OutputPathConfig(
    scratch_root=Path("scratch/monorepo"),
    hf_repo=HF_REPO_ID,
    base_model=BASE_MODEL_SLUG,
    category="ocean",
    trait=ARTIFACT_TRAIT,
    training_run=TRAINING_RUN,
    stage_dir="evals",
    eval_name=EVAL_NAME,
)
OUTPUT_ROOT = OUTPUT_CONFIG.scratch_dir
JUDGE_DATASET_PATH = OUTPUT_ROOT / "exports" / "all_responses.jsonl"
PLOTS_DIR = OUTPUT_ROOT / "plots"

CONDITIONS = single_turn_conditions({"no_prompt": None})

_SCALE_RE = re.compile(r"@scale_([+-]?\d+(?:\.\d+)?)")


def build_experiment_config(*, use_vllm: bool) -> ExperimentConfig:
    """Build the rollout experiment config."""
    return ExperimentConfig(
        assistant_model=BASE_MODEL,
        assistant_provider="vllm" if use_vllm else "local",
        assistant_temperature=ASSISTANT_TEMPERATURE,
        assistant_top_p=ASSISTANT_TOP_P,
        assistant_max_new_tokens=ASSISTANT_MAX_NEW_TOKENS,
        assistant_batch_size=ASSISTANT_BATCH_SIZE,
        user_model="gpt-4.1-nano-2025-04-14",
        user_provider="openrouter",
        user_temperature=0.7,
        user_top_p=0.95,
        user_max_new_tokens=128,
        user_batch_size=32,
        user_max_concurrent=32,
        dataset_path=DATASET_PATH,
        max_samples=MAX_SAMPLES,
        dataset_seed=SEED,
        num_rollouts=NUM_ROLLOUTS_PER_PROMPT,
        turns_per_phase=[1],
    )


def build_provider(*, use_vllm: bool) -> LoRaScaleProvider | VLLMLoRaScaleProvider:
    """Build the model provider for the scale sweep."""
    if use_vllm:
        return VLLMLoRaScaleProvider(
            base_model=BASE_MODEL,
            adapter=ADAPTER_REF,
            scale_points=SCALE_POINTS,
            baked_adapters_dir=Path("scratch/baked_adapters")
            / "conscientiousness_low_suppressor_v3_llama_3_1_8b_instruct",
            temperature=ASSISTANT_TEMPERATURE,
            top_p=ASSISTANT_TOP_P,
            max_new_tokens=ASSISTANT_MAX_NEW_TOKENS,
        )
    return LoRaScaleProvider(
        base_model=BASE_MODEL,
        adapter=ADAPTER_REF,
        scale_points=SCALE_POINTS,
    )


def build_sweep_config(*, upload: bool, use_vllm: bool) -> SweepConfig:
    """Build the full sweep config."""
    output = replace(OUTPUT_CONFIG, hf_repo=HF_REPO_ID if upload else None)
    return SweepConfig(
        provider=build_provider(use_vllm=use_vllm),
        conditions=CONDITIONS,
        evaluations=[],
        experiment=build_experiment_config(use_vllm=use_vllm),
        output=output,
        skip_completed=True,
        skip_evals=True,
        on_cell_error="warn",
        max_concurrent_conditions=1,
        plot=False,
        metadata={
            "seed": SEED,
            "adapter_ref": ADAPTER_REF,
            "trait": TRAIT.value,
            "direction": DIRECTION,
            "version": VERSION,
            "judge_metrics": [TRAIT.v2_metric_name, COHERENCE_METRIC],
            "judge_repeats": JUDGE_REPEATS,
            "judge_raters": [r.rater_id for r in JUDGE_RATERS],
        },
    )


def _download_existing_output() -> None:
    """Rehydrate the deterministic output directory from HF, if it exists."""
    print(f"Rehydrating from HF: {HF_REPO_ID}/{OUTPUT_CONFIG.hf_path}")
    try:
        download_from_dataset_repo(
            repo_id=HF_REPO_ID,
            path_in_repo=OUTPUT_CONFIG.hf_path,
            local_dir=OUTPUT_CONFIG.scratch_root,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  No existing HF artifacts downloaded: {exc}")


def _upload_output_root(commit_message: str) -> str | None:
    """Upload the full sweep/eval output root to the HF monorepo."""
    if not OUTPUT_ROOT.exists():
        return None
    url = upload_folder_to_dataset_repo(
        local_dir=OUTPUT_ROOT,
        repo_id=HF_REPO_ID,
        path_in_repo=OUTPUT_CONFIG.hf_path,
        commit_message=commit_message,
        ignore_patterns=["**/__pycache__/**"],
    )
    print(f"Uploaded output root to {url}/{OUTPUT_CONFIG.hf_path}")
    return url


def _has_rollout_files() -> bool:
    """Return True when converted rollout inputs are available locally."""
    patterns = [
        "scale_*/*/rollouts/rollouts.jsonl",
        "scale_*/*/rollouts.jsonl",
        "scale_*/*/*/rollouts/rollouts.jsonl",
        "scale_*/*/*/rollouts.jsonl",
    ]
    return any(any(OUTPUT_ROOT.glob(pattern)) for pattern in patterns)


def _has_all_rollout_files() -> bool:
    """Return True when every configured scale has a rollout export locally."""
    for scale in SCALE_POINTS:
        scale_dir = OUTPUT_ROOT / f"scale_{scale:+.2f}"
        if not any(scale_dir.glob("*/rollouts/rollouts.jsonl")) and not any(
            scale_dir.glob("*/rollouts.jsonl")
        ):
            return False
    return True


def materialize_judge_dataset() -> Path:
    """Convert rollouts to the flat all_responses.jsonl judge dataset."""
    if _has_rollout_files():
        n_rows = convert_sweep(
            OUTPUT_ROOT,
            JUDGE_DATASET_PATH,
            scales=SCALE_POINTS,
            assistant_model=BASE_MODEL,
        )
        if n_rows <= 0:
            raise RuntimeError(f"No rollout rows were converted from {OUTPUT_ROOT}")
        return JUDGE_DATASET_PATH

    if JUDGE_DATASET_PATH.exists():
        print(f"Using existing judge dataset: {JUDGE_DATASET_PATH}")
        return JUDGE_DATASET_PATH

    raise FileNotFoundError(
        f"No rollout files or existing judge dataset found under {OUTPUT_ROOT}"
    )


def _judge_config(metric_name: str) -> OceanJudgeRunConfig:
    raters = [
        rater.model_copy(update={"metric_name": metric_name}) for rater in JUDGE_RATERS
    ]
    return OceanJudgeRunConfig(
        trait=TRAIT,
        dataset_path=JUDGE_DATASET_PATH,
        judge_raters=raters,
        judge_repeats=JUDGE_REPEATS,
        plot=False,
        hf_repo_id=HF_REPO_ID,
        upload=False,
    )


def _upload_judge_dir(config: OceanJudgeRunConfig, metric_name: str) -> str | None:
    judge_dir = get_judge_run_dir(config)
    if not judge_dir.exists():
        return None
    judge_key = build_judge_run_key(config)
    return upload_folder_to_dataset_repo(
        local_dir=judge_dir,
        repo_id=HF_REPO_ID,
        path_in_repo=f"{OUTPUT_CONFIG.hf_path}/judge_runs/{judge_key}",
        commit_message=f"Upload {metric_name} judge run {judge_key}",
    )


def run_judge_metric(metric_name: str, *, upload: bool) -> dict[str, Any]:
    """Run one judge metric and upload even partial raw calls on failure."""
    config = _judge_config(metric_name)
    judge_key = build_judge_run_key(config)
    judge_dir = get_judge_run_dir(config)
    print(f"\nRunning judge metric: {metric_name}")
    print(f"  judge_key: {judge_key}")
    print(f"  judge_dir: {judge_dir}")
    try:
        result = run_ocean_judge_run(config)
    finally:
        if upload and judge_dir.exists():
            url = _upload_judge_dir(config, metric_name)
            if url:
                print(
                    f"  Uploaded judge artifacts to {url}/{OUTPUT_CONFIG.hf_path}/judge_runs/{judge_key}"
                )
    return result


def _parse_scale(condition: str) -> float | None:
    match = _SCALE_RE.search(condition)
    if not match:
        return None
    return float(match.group(1))


def _iter_raw_records(judge_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    raw_dir = judge_dir / "judge_calls" / "raw"
    for raw_path in sorted(raw_dir.glob("*.jsonl")):
        with raw_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    records.append(json.loads(text))
    return records


def _scale_scores(judge_dir: Path) -> dict[float, list[float]]:
    """Group median per-response/per-rater scores by scale."""
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    scale_by_key: dict[tuple[str, str], float] = {}

    for record in _iter_raw_records(judge_dir):
        if record.get("status") not in {"success", "parse_error"}:
            continue
        score = record.get("score")
        if not isinstance(score, int):
            continue
        scale = _parse_scale(str(record.get("condition", "")))
        if scale is None:
            continue
        key = (str(record.get("rater_id", "")), str(record.get("response_id", "")))
        grouped[key].append(score)
        scale_by_key[key] = scale

    by_scale: dict[float, list[float]] = defaultdict(list)
    for key, scores in grouped.items():
        if scores:
            by_scale[scale_by_key[key]].append(float(statistics.median(scores)))
    return dict(by_scale)


def _ci95_from_bootstrap(values: list[float]) -> tuple[float, float]:
    """Return absolute 95% bootstrap CI bounds for the mean judge score."""
    if len(values) <= 1:
        mean = values[0] if values else math.nan
        return mean, mean

    import numpy as np
    from scipy import stats

    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(SEED)
    try:
        result = stats.bootstrap(
            (arr,),
            statistic=np.mean,
            n_resamples=CI_BOOTSTRAP_RESAMPLES,
            confidence_level=CI_CONFIDENCE / 100,
            random_state=rng,
            method="BCa",
        )
        low = float(result.confidence_interval.low)
        high = float(result.confidence_interval.high)
    except Exception:  # noqa: BLE001
        mean = float(arr.mean())
        return mean, mean

    if not (math.isfinite(low) and math.isfinite(high)):
        mean = float(arr.mean())
        return mean, mean
    return low, high


def _summary_row(metric_name: str, scale: float, values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "metric": metric_name,
            "scale": scale,
            "n": 0,
            "mean": math.nan,
            "median": math.nan,
            "std": math.nan,
            "min": math.nan,
            "max": math.nan,
            "ci_lower": math.nan,
            "ci_upper": math.nan,
            "ci_method": f"ci{CI_CONFIDENCE:g}_from_bootstrap_{CI_BOOTSTRAP_RESAMPLES}",
        }
    ci_lower, ci_upper = _ci95_from_bootstrap(values)
    return {
        "metric": metric_name,
        "scale": scale,
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_method": f"ci{CI_CONFIDENCE:g}_from_bootstrap_{CI_BOOTSTRAP_RESAMPLES}",
    }


def write_scale_summary_and_plot(results: dict[str, dict[str, Any]]) -> Path | None:
    """Write the script-specific scale summary and plot."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot generation")
        return None

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    metric_to_rows: dict[str, list[dict[str, Any]]] = {}

    for metric_name, result in results.items():
        judge_dir = Path(result["judge_dir"])
        by_scale = _scale_scores(judge_dir)
        rows = [
            _summary_row(metric_name, scale, by_scale.get(scale, []))
            for scale in SCALE_POINTS
        ]
        metric_to_rows[metric_name] = rows
        summary_rows.extend(rows)

    summary_path = OUTPUT_ROOT / "analysis" / "scale_summary.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        for row in summary_rows:
            handle.write(json.dumps(row) + "\n")

    fig, left_axis = plt.subplots(figsize=(7.0, 3.5))
    right_axis = left_axis.twinx()
    metric_axes = {
        TRAIT.v2_metric_name: (
            left_axis,
            CONSCIENTIOUSNESS_COLOR,
            "Conscientiousness",
        ),
        COHERENCE_METRIC: (right_axis, COHERENCE_COLOR, "Coherence"),
    }
    lines = []
    for metric_name in (TRAIT.v2_metric_name, COHERENCE_METRIC):
        rows = metric_to_rows.get(metric_name)
        if not rows:
            continue
        axis, color, label = metric_axes[metric_name]
        xs = [row["scale"] for row in rows]
        ys = [row["mean"] for row in rows]
        (line,) = axis.plot(xs, ys, marker="o", linewidth=2, color=color, label=label)
        yerr = [
            [max(0.0, row["mean"] - row["ci_lower"]) for row in rows],
            [max(0.0, row["ci_upper"] - row["mean"]) for row in rows],
        ]
        axis.errorbar(
            xs,
            ys,
            yerr=yerr,
            fmt="none",
            color=color,
            capsize=3,
            capthick=1.0,
            elinewidth=1.0,
            alpha=0.75,
            zorder=5,
        )
        axis.set_ylabel(f"{label} mean judge score", color=color)
        axis.tick_params(axis="y", labelcolor=color)
        lines.append(line)

    left_axis.axvline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    left_axis.set_title("Conscientiousness suppressor LoRA scale sweep")
    left_axis.set_xlabel("LoRA scale")
    left_axis.grid(alpha=0.25)
    if lines:
        left_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()

    plot_path = PLOTS_DIR / "llm_judge_scale_sweep.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote scale summary: {summary_path}")
    print(f"Wrote plot: {plot_path}")
    return plot_path


def print_dry_run(*, use_vllm: bool, upload: bool) -> None:
    n_responses = (
        len(SCALE_POINTS) * len(CONDITIONS) * MAX_SAMPLES * NUM_ROLLOUTS_PER_PROMPT
    )
    n_judge_calls = n_responses * 2 * len(JUDGE_RATERS) * JUDGE_REPEATS
    print("DRY RUN: conscientiousness suppressor LLM-judge scale sweep")
    print(f"  adapter       : {ADAPTER_REF}")
    print(f"  base model    : {BASE_MODEL}")
    print(f"  scales        : {SCALE_POINTS}")
    print(f"  prompts       : {MAX_SAMPLES} (seed={SEED})")
    print(f"  responses     : {n_responses}")
    print(f"  judge metrics : {[TRAIT.v2_metric_name, COHERENCE_METRIC]}")
    print(f"  judge raters  : {[r.rater_id for r in JUDGE_RATERS]}")
    print(f"  judge repeats : {JUDGE_REPEATS}")
    print(f"  judge calls   : {n_judge_calls}")
    print(f"  provider      : {'vllm' if use_vllm else 'local'}")
    print(f"  local output  : {OUTPUT_ROOT}")
    print(
        f"  HF output     : {HF_REPO_ID}/{OUTPUT_CONFIG.hf_path if upload else '(disabled)'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--skip-rollouts", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--local-provider", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    upload = not args.no_upload
    use_vllm = not args.local_provider

    if args.dry_run:
        print_dry_run(use_vllm=use_vllm, upload=upload)
        return

    if upload:
        login_from_env()
        _download_existing_output()

    if args.skip_rollouts:
        print("Skipping rollout generation by request")
    elif _has_all_rollout_files():
        print("Found local rollout files for all scales; skipping rollout generation")
    else:
        run_sweep(build_sweep_config(upload=upload, use_vllm=use_vllm))

    materialize_judge_dataset()

    results: dict[str, dict[str, Any]] = {}
    if not args.skip_judge:
        for metric_name in (TRAIT.v2_metric_name, COHERENCE_METRIC):
            results[metric_name] = run_judge_metric(metric_name, upload=upload)
        write_scale_summary_and_plot(results)
    else:
        print("Skipping judge scoring by request")

    if upload:
        _upload_output_root(
            "Upload conscientiousness suppressor LoRA scale LLM-judge sweep"
        )

    print(f"Done. Local output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
