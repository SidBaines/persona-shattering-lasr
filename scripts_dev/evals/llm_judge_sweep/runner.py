#!/usr/bin/env python3
"""LLM-judge LoRA scale sweep runner.

Standardized runner that imports experiment constants from a config module
and executes four stages with deterministic, chained run IDs:

1. **rollout** — sweep LoRA adapter over scale points, generate single-turn
   rollouts on the canonical assistant-axis prompt set.
2. **convert** — flatten rollouts into a judge-compatible response dataset.
3. **judge** — score responses with an LLM judge panel.
4. **plot** — produce scale-vs-score summary and figures (always re-runs).

All artifacts are cached locally and on HuggingFace via ``StageCache``.

Usage::

    uv run python -m scripts_dev.evals.llm_judge_sweep.runner \\
        --config scripts_dev.evals.llm_judge_sweep.configs.conscientiousness_suppressor

Useful flags::

    --dry-run        Print the config and estimated call counts.
    --no-upload      Do not download from or upload to HuggingFace.
    --skip-rollouts  Reuse existing local/HF rollouts and only run conversion/judges.
    --skip-judge     Generate/convert rollouts but skip judge scoring.
    --local-provider Use the PEFT local provider instead of vLLM.
    --config MODULE  Python module path to the config constants.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_USE_V1", "1")

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path for script execution.
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

# Lightweight imports only — heavy libraries (numpy, torch, transformers,
# sweep, model_providers) are imported inside functions so that --dry-run
# stays fast.
from src_dev.eval_stages import StageCache, StageCacheConfig, chained_run_id, seed_all
from src_dev.utils.hf_hub import login_from_env

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_SCALE_RE = re.compile(r"@scale_([+-]?\d+(?:\.\d+)?)")
HF_REPO_ID = "persona-shattering-lasr/monorepo"


def _parse_flags() -> argparse.Namespace:
    """Parse operational CLI flags."""
    parser = argparse.ArgumentParser(
        description="LLM-judge LoRA scale sweep runner.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Python module path to the config constants "
        "(e.g. scripts_dev.evals.llm_judge_sweep.configs.conscientiousness_suppressor).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--skip-rollouts", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--local-provider", action="store_true")
    return parser.parse_args()


def _load_config(module_path: str) -> ModuleType:
    """Import and return the config module."""
    return importlib.import_module(module_path)


# ---------------------------------------------------------------------------
# Run-ID builders
# ---------------------------------------------------------------------------


def _rollout_run_id(cfg: ModuleType) -> str:
    return chained_run_id(
        "rollout",
        {
            "adapter_ref": cfg.ADAPTER_REF,
            "base_model": cfg.BASE_MODEL,
            "scale_points": cfg.SCALE_POINTS,
            "max_samples": cfg.MAX_SAMPLES,
            "seed": cfg.SEED,
            "temperature": cfg.ASSISTANT_TEMPERATURE,
            "dataset_name": cfg.DATASET_PATH,
        },
    )


def _convert_run_id(cfg: ModuleType, rollout_id: str) -> str:
    return chained_run_id("convert", {}, parent_run_id=rollout_id)


def _judge_run_id(cfg: ModuleType, convert_id: str) -> str:
    return chained_run_id(
        "judge",
        {
            "raters": [r.rater_id for r in cfg.JUDGE_RATERS],
            "repeats": cfg.JUDGE_REPEATS,
        },
        parent_run_id=convert_id,
    )


# ---------------------------------------------------------------------------
# Stage: rollout
# ---------------------------------------------------------------------------


def _build_experiment_config(cfg: ModuleType, *, use_vllm: bool) -> Any:
    """Build the rollout experiment config from the config module."""
    from src_dev.sweep import ExperimentConfig

    return ExperimentConfig(
        assistant_model=cfg.BASE_MODEL,
        assistant_provider="vllm" if use_vllm else "local",
        assistant_temperature=cfg.ASSISTANT_TEMPERATURE,
        assistant_top_p=cfg.ASSISTANT_TOP_P,
        assistant_max_new_tokens=cfg.ASSISTANT_MAX_NEW_TOKENS,
        assistant_batch_size=cfg.ASSISTANT_BATCH_SIZE,
        user_model=getattr(cfg, "USER_MODEL", "z-ai/glm-4.5-air:free"),
        user_provider=getattr(cfg, "USER_PROVIDER", "openrouter"),
        user_temperature=0.7,
        user_top_p=0.95,
        user_max_new_tokens=128,
        user_batch_size=32,
        user_max_concurrent=32,
        dataset_path=cfg.DATASET_PATH,
        max_samples=cfg.MAX_SAMPLES,
        dataset_seed=cfg.SEED,
        num_rollouts=cfg.NUM_ROLLOUTS_PER_PROMPT,
        turns_per_phase=[1],
    )


def _build_provider(cfg: ModuleType, *, use_vllm: bool) -> Any:
    """Build the model provider for the scale sweep."""
    from src_dev.rollout_generation.model_providers import (
        LoRaScaleProvider,
        VLLMLoRaScaleProvider,
    )

    if use_vllm:
        return VLLMLoRaScaleProvider(
            base_model=cfg.BASE_MODEL,
            adapter=cfg.ADAPTER_REF,
            scale_points=cfg.SCALE_POINTS,
            baked_adapters_dir=Path("scratch/baked_adapters") / cfg.BAKED_ADAPTERS_SUBDIR,
            temperature=cfg.ASSISTANT_TEMPERATURE,
            top_p=cfg.ASSISTANT_TOP_P,
            max_new_tokens=cfg.ASSISTANT_MAX_NEW_TOKENS,
        )
    return LoRaScaleProvider(
        base_model=cfg.BASE_MODEL,
        adapter=cfg.ADAPTER_REF,
        scale_points=cfg.SCALE_POINTS,
    )


def _build_output_path_config(cfg: ModuleType) -> Any:
    """Build the OutputPathConfig used by the sweep stage for local writes."""
    from src_dev.sweep import OutputPathConfig

    return OutputPathConfig(
        scratch_root=Path("scratch/monorepo"),
        hf_repo=HF_REPO_ID,
        base_model=cfg.BASE_MODEL_SLUG,
        category="ocean",
        trait=cfg.ARTIFACT_TRAIT,
        training_run=cfg.TRAINING_RUN,
        stage_dir="evals",
        eval_name="llm_judge_lora_scale_sweep",
    )


def _run_rollout_stage(
    cfg: ModuleType,
    cache: StageCache,
    rollout_id: str,
    *,
    use_vllm: bool,
    skip: bool,
) -> None:
    """Execute the rollout sweep stage."""
    output_config = _build_output_path_config(cfg)
    output_root = output_config.scratch_dir

    def do_rollouts() -> None:
        from src_dev.sweep import SweepConfig, run_sweep, single_turn_conditions

        conditions = single_turn_conditions({"no_prompt": None})
        sweep_config = SweepConfig(
            provider=_build_provider(cfg, use_vllm=use_vllm),
            conditions=conditions,
            evaluations=[],
            experiment=_build_experiment_config(cfg, use_vllm=use_vllm),
            output=replace(output_config, hf_repo=None),
            skip_completed=True,
            skip_evals=True,
            on_cell_error="warn",
            max_concurrent_conditions=1,
            plot=False,
            metadata={
                "seed": cfg.SEED,
                "adapter_ref": cfg.ADAPTER_REF,
                "trait": cfg.TRAIT.value,
                "direction": cfg.DIRECTION,
                "version": cfg.VERSION,
                "judge_metrics": [cfg.TRAIT.v2_metric_name, cfg.COHERENCE_METRIC],
                "judge_repeats": cfg.JUDGE_REPEATS,
                "judge_raters": [r.rater_id for r in cfg.JUDGE_RATERS],
            },
        )
        run_sweep(sweep_config)

        # Copy rollout files into the cache stage dir so they are self-contained.
        stage_dir = cache.stage_dir("rollout", rollout_id)
        marker = stage_dir / "_sweep_output_root.txt"
        marker.write_text(str(output_root) + "\n")

    if skip:
        print("  --skip-rollouts: skipping rollout stage")
    else:
        cache.run_or_hydrate(
            "rollout",
            rollout_id,
            do_rollouts,
            config={
                "adapter_ref": cfg.ADAPTER_REF,
                "base_model": cfg.BASE_MODEL,
                "scale_points": cfg.SCALE_POINTS,
                "max_samples": cfg.MAX_SAMPLES,
                "seed": cfg.SEED,
            },
        )


# ---------------------------------------------------------------------------
# Stage: convert
# ---------------------------------------------------------------------------


def _get_sweep_output_root(cfg: ModuleType) -> Path:
    """Return the sweep output root path from the OutputPathConfig."""
    return _build_output_path_config(cfg).scratch_dir


def _run_convert_stage(
    cfg: ModuleType,
    cache: StageCache,
    convert_id: str,
    rollout_id: str,
    *,
    skip: bool,
) -> Path:
    """Convert rollouts into flat judge-compatible dataset.

    Returns:
        Path to the all_responses.jsonl file.
    """
    output_root = _get_sweep_output_root(cfg)
    judge_dataset_path = output_root / "exports" / "all_responses.jsonl"

    def do_convert() -> None:
        from scripts_dev.persona_metrics.llm_judge.rollout_sweep_to_judge_dataset import (
            convert_sweep,
        )

        n_rows = convert_sweep(
            output_root,
            judge_dataset_path,
            scales=cfg.SCALE_POINTS,
            assistant_model=cfg.BASE_MODEL,
        )
        if n_rows <= 0:
            raise RuntimeError(f"No rollout rows were converted from {output_root}")

        # Write a pointer into the stage dir.
        stage_dir = cache.stage_dir("convert", convert_id)
        (stage_dir / "_judge_dataset_path.txt").write_text(str(judge_dataset_path) + "\n")

    if skip:
        print("  --skip-rollouts: skipping convert stage")
    else:
        cache.run_or_hydrate(
            "convert",
            convert_id,
            do_convert,
            config={"sweep_output_root": str(output_root)},
            parent_run_id=rollout_id,
        )
    return judge_dataset_path


# ---------------------------------------------------------------------------
# Stage: judge
# ---------------------------------------------------------------------------


def _judge_config(
    cfg: ModuleType,
    metric_name: str,
    judge_dataset_path: Path,
) -> Any:
    """Build an OceanJudgeRunConfig for one metric."""
    from src_dev.persona_metrics.llm_judge_agreement import OceanJudgeRunConfig
    raters = [
        rater.model_copy(update={"metric_name": metric_name})
        for rater in cfg.JUDGE_RATERS
    ]
    return OceanJudgeRunConfig(
        trait=cfg.TRAIT,
        dataset_path=judge_dataset_path,
        judge_raters=raters,
        judge_repeats=cfg.JUDGE_REPEATS,
        plot=False,
        hf_repo_id=HF_REPO_ID,
        upload=False,
    )


def _run_judge_metric(
    cfg: ModuleType,
    metric_name: str,
    judge_dataset_path: Path,
) -> dict[str, Any]:
    """Run one judge metric. Returns the result dict."""
    from src_dev.persona_metrics.llm_judge_agreement import (
        build_judge_run_key,
        get_judge_run_dir,
        run_ocean_judge_run,
    )

    config = _judge_config(cfg, metric_name, judge_dataset_path)
    judge_key = build_judge_run_key(config)
    judge_dir = get_judge_run_dir(config)
    print(f"\nRunning judge metric: {metric_name}")
    print(f"  judge_key: {judge_key}")
    print(f"  judge_dir: {judge_dir}")
    result = run_ocean_judge_run(config)
    return result


def _run_judge_stage(
    cfg: ModuleType,
    cache: StageCache,
    judge_id: str,
    convert_id: str,
    judge_dataset_path: Path,
    *,
    skip: bool,
) -> dict[str, dict[str, Any]]:
    """Run all judge metrics. Returns {metric_name: result_dict}."""
    results: dict[str, dict[str, Any]] = {}

    if skip:
        print("[judge] Skipped by request.")
        return results

    metrics = [cfg.TRAIT.v2_metric_name, cfg.COHERENCE_METRIC]

    def do_judge() -> None:
        for metric_name in metrics:
            results[metric_name] = _run_judge_metric(
                cfg, metric_name, judge_dataset_path
            )

    cache.run_or_hydrate(
        "judge",
        judge_id,
        do_judge,
        config={
            "raters": [r.rater_id for r in cfg.JUDGE_RATERS],
            "repeats": cfg.JUDGE_REPEATS,
            "metrics": metrics,
        },
        parent_run_id=convert_id,
    )

    # On cache hit do_judge is never called, so reconstruct results from
    # the known judge directory structure so plotting still works.
    if not results:
        from src_dev.persona_metrics.llm_judge_agreement import get_judge_run_dir

        for metric_name in metrics:
            jcfg = _judge_config(cfg, metric_name, judge_dataset_path)
            jdir = get_judge_run_dir(jcfg)
            if jdir.exists():
                results[metric_name] = {"judge_dir": str(jdir)}

    return results


# ---------------------------------------------------------------------------
# Stage: plot (always re-runs, never cached)
# ---------------------------------------------------------------------------


def _parse_scale(condition: str) -> float | None:
    """Extract the LoRA scale value from a condition string."""
    match = _SCALE_RE.search(condition)
    if not match:
        return None
    return float(match.group(1))


def _iter_raw_records(judge_dir: Path) -> list[dict[str, Any]]:
    """Read all raw judge call records from a judge directory."""
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


def _ci95_from_bootstrap(
    values: list[float], seed: int, n_resamples: int, confidence: float
) -> tuple[float, float]:
    """Return absolute bootstrap CI bounds for the mean judge score."""
    if len(values) <= 1:
        mean = values[0] if values else math.nan
        return mean, mean

    import numpy as np
    from scipy import stats as scipy_stats

    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    try:
        result = scipy_stats.bootstrap(
            (arr,),
            statistic=np.mean,
            n_resamples=n_resamples,
            confidence_level=confidence / 100,
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


def _summary_row(
    cfg: ModuleType, metric_name: str, scale: float, values: list[float]
) -> dict[str, Any]:
    """Build one summary row for a (metric, scale) combination."""
    ci_method = f"ci{cfg.CI_CONFIDENCE:g}_from_bootstrap_{cfg.CI_BOOTSTRAP_RESAMPLES}"
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
            "ci_method": ci_method,
        }
    ci_lower, ci_upper = _ci95_from_bootstrap(
        values, cfg.SEED, cfg.CI_BOOTSTRAP_RESAMPLES, cfg.CI_CONFIDENCE
    )
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
        "ci_method": ci_method,
    }


def _run_plot_stage(
    cfg: ModuleType,
    results: dict[str, dict[str, Any]],
) -> Path | None:
    """Write the scale summary and produce the dual-axis plot.

    Always re-runs (never cached). Returns the plot path or None.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot generation.")
        return None

    output_root = _get_sweep_output_root(cfg)
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    trait_metric = cfg.TRAIT.v2_metric_name
    coherence_metric = cfg.COHERENCE_METRIC
    summary_rows: list[dict[str, Any]] = []
    metric_to_rows: dict[str, list[dict[str, Any]]] = {}

    for metric_name, result in results.items():
        judge_dir = Path(result["judge_dir"])
        by_scale = _scale_scores(judge_dir)
        rows = [
            _summary_row(cfg, metric_name, scale, by_scale.get(scale, []))
            for scale in cfg.SCALE_POINTS
        ]
        metric_to_rows[metric_name] = rows
        summary_rows.extend(rows)

    # Write summary JSONL.
    summary_path = output_root / "analysis" / "scale_summary.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        for row in summary_rows:
            handle.write(json.dumps(row) + "\n")

    # Dual-axis plot.
    fig, left_axis = plt.subplots(figsize=(7.0, 3.5))
    right_axis = left_axis.twinx()

    trait_label = cfg.TRAIT.value.replace("_", " ").title()
    metric_axes = {
        trait_metric: (left_axis, cfg.TRAIT_COLOR, trait_label),
        coherence_metric: (right_axis, cfg.COHERENCE_COLOR, "Coherence"),
    }
    lines = []
    for metric_name in (trait_metric, coherence_metric):
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
    left_axis.set_title(cfg.PLOT_TITLE)
    left_axis.set_xlabel("LoRA scale")
    left_axis.grid(alpha=0.25)
    if lines:
        left_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.tight_layout()

    plot_path = plots_dir / "llm_judge_scale_sweep.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote scale summary: {summary_path}")
    print(f"Wrote plot: {plot_path}")
    return plot_path


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def _print_dry_run(cfg: ModuleType, *, use_vllm: bool, upload: bool) -> None:
    """Print estimated work without executing anything."""
    # 1 condition per scale point (single_turn_conditions with no_prompt)
    n_conditions = 1
    n_responses = (
        len(cfg.SCALE_POINTS)
        * n_conditions
        * cfg.MAX_SAMPLES
        * cfg.NUM_ROLLOUTS_PER_PROMPT
    )
    n_judge_calls = n_responses * 2 * len(cfg.JUDGE_RATERS) * cfg.JUDGE_REPEATS

    rollout_id = _rollout_run_id(cfg)
    convert_id = _convert_run_id(cfg, rollout_id)
    judge_id = _judge_run_id(cfg, convert_id)

    print("DRY RUN: LLM-judge LoRA scale sweep")
    print(f"  eval name     : {cfg.EVAL_NAME}")
    print(f"  adapter       : {cfg.ADAPTER_REF}")
    print(f"  base model    : {cfg.BASE_MODEL}")
    print(f"  scales        : {cfg.SCALE_POINTS}")
    print(f"  prompts       : {cfg.MAX_SAMPLES} (seed={cfg.SEED})")
    print(f"  responses     : {n_responses}")
    print(f"  judge metrics : {[cfg.TRAIT.v2_metric_name, cfg.COHERENCE_METRIC]}")
    print(f"  judge raters  : {[r.rater_id for r in cfg.JUDGE_RATERS]}")
    print(f"  judge repeats : {cfg.JUDGE_REPEATS}")
    print(f"  judge calls   : {n_judge_calls}")
    print(f"  provider      : {'vllm' if use_vllm else 'local'}")
    print(f"  run IDs       : rollout={rollout_id}  convert={convert_id}  judge={judge_id}")
    print(f"  HF base path  : evals/llm-judge-sweep/{cfg.EVAL_NAME}")
    print(f"  upload        : {upload}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the LLM-judge scale sweep runner."""
    flags = _parse_flags()
    cfg = _load_config(flags.config)

    # Seed everything before any stochastic work.
    seed_all(cfg.SEED)

    load_dotenv()
    upload = not flags.no_upload
    use_vllm = not flags.local_provider

    if flags.dry_run:
        _print_dry_run(cfg, use_vllm=use_vllm, upload=upload)
        return

    if upload:
        login_from_env()

    # Build the stage cache.
    cache = StageCache(
        StageCacheConfig(
            cache_root=Path("scratch/eval-cache"),
            hf_repo=HF_REPO_ID if upload else None,
            hf_base_path=f"evals/llm-judge-sweep/{cfg.EVAL_NAME}",
            no_remote=not upload,
        )
    )

    # Compute chained run IDs.
    rollout_id = _rollout_run_id(cfg)
    convert_id = _convert_run_id(cfg, rollout_id)
    judge_id = _judge_run_id(cfg, convert_id)

    print(f"Run IDs: rollout={rollout_id}  convert={convert_id}  judge={judge_id}")

    # Stage 1: rollout
    _run_rollout_stage(
        cfg,
        cache,
        rollout_id,
        use_vllm=use_vllm,
        skip=flags.skip_rollouts,
    )

    # Stage 2: convert
    judge_dataset_path = _run_convert_stage(
        cfg,
        cache,
        convert_id,
        rollout_id,
        skip=False,
    )

    # Stage 3: judge
    results = _run_judge_stage(
        cfg,
        cache,
        judge_id,
        convert_id,
        judge_dataset_path,
        skip=flags.skip_judge,
    )

    # Stage 4: plot (always re-runs)
    if results:
        _run_plot_stage(cfg, results)
    else:
        print("[plot] No judge results to plot.")

    output_root = _get_sweep_output_root(cfg)
    print(f"Done. Local output: {output_root}")


if __name__ == "__main__":
    main()
