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
from src_dev.eval_stages.run_id import run_id_from_dict
from src_dev.utils.hf_hub import login_from_env

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_SCALE_RE = re.compile(r"@scale_([+-]?\d+(?:\.\d+)?)")
HF_REPO_ID = "persona-shattering-lasr/monorepo"
_BASELINE_SCALE = 0.0


def _is_baseline_scale(s: float) -> bool:
    return abs(s - _BASELINE_SCALE) < 1e-9


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
# Run-ID builders and fingerprints
# ---------------------------------------------------------------------------


def _baseline_fingerprint(cfg: ModuleType) -> str:
    """Content-addressed fingerprint for the baseline (scale=0) cell.

    Fields are everything that affects baseline rollouts independent of the
    adapter or non-zero scale points — so different sweeps with the same
    base model + dataset + seed + gen params share a baseline.
    """
    return run_id_from_dict(
        {
            "base_model": cfg.BASE_MODEL,
            "dataset_path": cfg.DATASET_PATH,
            "max_samples": cfg.MAX_SAMPLES,
            "seed": cfg.SEED,
            "num_rollouts_per_prompt": cfg.NUM_ROLLOUTS_PER_PROMPT,
            "assistant_temperature": cfg.ASSISTANT_TEMPERATURE,
            "assistant_top_p": cfg.ASSISTANT_TOP_P,
            "assistant_max_new_tokens": cfg.ASSISTANT_MAX_NEW_TOKENS,
        },
        length=10,
    )


def _nonzero_scale_points(cfg: ModuleType) -> list[float]:
    return [s for s in cfg.SCALE_POINTS if not _is_baseline_scale(s)]


def _baseline_run_id(cfg: ModuleType) -> str:
    return chained_run_id(
        "baseline",
        {
            "baseline_fp": _baseline_fingerprint(cfg),
        },
    )


def _rollout_run_id(cfg: ModuleType) -> str:
    return chained_run_id(
        "rollout",
        {
            "adapter_ref": cfg.ADAPTER_REF,
            "base_model": cfg.BASE_MODEL,
            "scale_points": _nonzero_scale_points(cfg),
            "max_samples": cfg.MAX_SAMPLES,
            "seed": cfg.SEED,
            "temperature": cfg.ASSISTANT_TEMPERATURE,
            "dataset_name": cfg.DATASET_PATH,
        },
    )


def _convert_run_id(cfg: ModuleType, rollout_id: str, baseline_id: str) -> str:
    return chained_run_id(
        "convert",
        {"baseline_id": baseline_id},
        parent_run_id=rollout_id,
    )


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


def _build_provider(
    cfg: ModuleType, scale_points: list[float], *, use_vllm: bool
) -> Any:
    """Build the model provider for the given scale points."""
    from src_dev.rollout_generation.model_providers import (
        LoRaScaleProvider,
        VLLMLoRaScaleProvider,
    )

    if use_vllm:
        return VLLMLoRaScaleProvider(
            base_model=cfg.BASE_MODEL,
            adapter=cfg.ADAPTER_REF,
            scale_points=scale_points,
            baked_adapters_dir=Path("scratch/baked_adapters") / cfg.BAKED_ADAPTERS_SUBDIR,
            temperature=cfg.ASSISTANT_TEMPERATURE,
            top_p=cfg.ASSISTANT_TOP_P,
            max_new_tokens=cfg.ASSISTANT_MAX_NEW_TOKENS,
        )
    return LoRaScaleProvider(
        base_model=cfg.BASE_MODEL,
        adapter=cfg.ADAPTER_REF,
        scale_points=scale_points,
    )


_EVAL_NAME = "llm_judge_lora_scale_sweep"
_CATEGORY = "ocean"


def _build_sweep_output_path_config(cfg: ModuleType) -> Any:
    """OutputPathConfig for non-zero scale cells (per-trait/direction/version)."""
    from src_dev.sweep import OutputPathConfig

    return OutputPathConfig(
        scratch_root=Path("scratch/monorepo"),
        hf_repo=HF_REPO_ID,
        base_model=cfg.BASE_MODEL_SLUG,
        category=_CATEGORY,
        trait=cfg.TRAIT.value,
        direction=cfg.DIRECTION,
        version=cfg.VERSION,
        stage_dir="evals",
        eval_name=_EVAL_NAME,
    )


def _build_baseline_output_path_config(cfg: ModuleType) -> Any:
    """OutputPathConfig for the scale=0 baseline cell (per-category shared path).

    Uses ``trait="_baseline"`` and ``direction=eval_name`` / ``version=fingerprint``
    so the final HF path is::

        fine_tuning/{base_model}/{category}/_baseline/{eval_name}/{fingerprint}/scale_+0.00/no_prompt/
    """
    from src_dev.sweep import OutputPathConfig

    return OutputPathConfig(
        scratch_root=Path("scratch/monorepo"),
        hf_repo=HF_REPO_ID,
        base_model=cfg.BASE_MODEL_SLUG,
        category=_CATEGORY,
        trait="_baseline",
        direction=_EVAL_NAME,
        version=_baseline_fingerprint(cfg),
        stage_dir="",
        eval_name="",
    )


def _hydrate_rollouts_from_hf(output_config: Any) -> None:
    """Download any existing rollouts at this HF path into local scratch.

    ``run_sweep`` only checks HF to decide whether to skip generation; it does
    not download rollout files locally. Without this pre-hydration step, a
    fresh machine would skip generation but then have empty local cells, and
    downstream stages (convert) would fail.
    """
    from src_dev.sweep import download_rollouts_from_hf

    if not output_config.hf_repo:
        return
    try:
        download_rollouts_from_hf(output_config)
    except Exception as exc:  # noqa: BLE001
        # A missing prefix on HF is the normal first-run case; just log.
        print(f"  [hydrate] skipped ({type(exc).__name__}: {exc})")


def _hydrate_derived_outputs_from_hf(
    cache: StageCache,
    sweep_root: Path,
    subpaths: list[str],
) -> None:
    """Pull sibling output subtrees (``exports``, ``judge_runs``, ...) from HF.

    The StageCache for convert/judge only stores ``done.json`` markers and
    path pointers — the real artifacts are written to ``sweep_root/<subpath>``.
    On a fresh machine, hydrating the cache marker alone is not enough: the
    downstream stages' cache-hit branches look for these sibling dirs and
    silently produce no results when they are missing. Calling this before
    convert/judge ensures they find their inputs.
    """
    if not cache.has_remote:
        return
    from src_dev.utils.hf_hub import download_path_to_dir

    for subpath in subpaths:
        target_dir = sweep_root / subpath
        if target_dir.exists():
            continue
        try:
            download_path_to_dir(
                repo_id=cache.hf_repo,
                path_in_repo=f"{cache.hf_base_path}/{subpath}",
                target_dir=target_dir,
            )
            print(f"  [hydrate] {subpath}: downloaded from HF")
        except Exception as exc:  # noqa: BLE001
            # Missing prefix on HF is the normal first-run case.
            print(f"  [hydrate] {subpath}: skipped ({type(exc).__name__}: {exc})")


def _run_rollout_for_scales(
    cfg: ModuleType,
    scale_points: list[float],
    output_config: Any,
    *,
    use_vllm: bool,
    metadata_extra: dict[str, Any],
) -> None:
    """Run the sweep for a given set of scale points into the given output_config.

    Upload to HF is governed by ``output_config.hf_repo`` — set it to ``None``
    on the config to disable upload.
    """
    from src_dev.sweep import SweepConfig, run_sweep, single_turn_conditions

    conditions = single_turn_conditions({"no_prompt": None})
    sweep_config = SweepConfig(
        provider=_build_provider(cfg, scale_points, use_vllm=use_vllm),
        conditions=conditions,
        evaluations=[],
        experiment=_build_experiment_config(cfg, use_vllm=use_vllm),
        output=output_config,
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
            "scale_points": scale_points,
            "judge_metrics": [cfg.TRAIT.v2_metric_name, cfg.COHERENCE_METRIC],
            "judge_repeats": cfg.JUDGE_REPEATS,
            "judge_raters": [r.rater_id for r in cfg.JUDGE_RATERS],
            **metadata_extra,
        },
    )
    run_sweep(sweep_config)


def _run_baseline_stage(
    cfg: ModuleType,
    cache: StageCache,
    baseline_id: str,
    *,
    use_vllm: bool,
    skip: bool,
) -> Path:
    """Run the scale=0 baseline cell at the per-category shared HF path.

    Returns the baseline output root (so the convert stage can read rollouts
    from ``<baseline_root>/scale_+0.00/no_prompt/rollouts/...``).
    """
    output_config = _build_baseline_output_path_config(cfg)
    baseline_root = output_config.scratch_dir

    # Always hydrate rollouts from HF first: if StageCache cache-hits on the
    # marker it never calls do_baseline, but convert still needs the actual
    # rollouts.jsonl files to exist locally.
    _hydrate_rollouts_from_hf(output_config)

    def do_baseline() -> None:
        _run_rollout_for_scales(
            cfg,
            [_BASELINE_SCALE],
            output_config,
            use_vllm=use_vllm,
            metadata_extra={
                "baseline_fp": _baseline_fingerprint(cfg),
                "stage_role": "baseline",
            },
        )
        stage_dir = cache.stage_dir("baseline", baseline_id)
        (stage_dir / "_baseline_output_root.txt").write_text(
            str(baseline_root) + "\n"
        )

    if skip:
        print("  --skip-rollouts: skipping baseline stage")
        return baseline_root

    cache.run_or_hydrate(
        "baseline",
        baseline_id,
        do_baseline,
        config={
            "baseline_fp": _baseline_fingerprint(cfg),
            "base_model": cfg.BASE_MODEL,
            "dataset_path": cfg.DATASET_PATH,
            "max_samples": cfg.MAX_SAMPLES,
            "seed": cfg.SEED,
        },
    )
    return baseline_root


def _run_rollout_stage(
    cfg: ModuleType,
    cache: StageCache,
    rollout_id: str,
    *,
    use_vllm: bool,
    skip: bool,
) -> Path:
    """Run non-zero scale cells at the per-trait/direction/version HF path.

    Returns the sweep output root.
    """
    output_config = _build_sweep_output_path_config(cfg)
    output_root = output_config.scratch_dir
    scales = _nonzero_scale_points(cfg)

    # Always hydrate rollouts from HF first — see note on _run_baseline_stage.
    _hydrate_rollouts_from_hf(output_config)

    def do_rollouts() -> None:
        if not scales:
            print("  [rollout] No non-zero scale points; skipping sweep rollouts.")
            return
        _run_rollout_for_scales(
            cfg,
            scales,
            output_config,
            use_vllm=use_vllm,
            metadata_extra={"stage_role": "sweep"},
        )
        stage_dir = cache.stage_dir("rollout", rollout_id)
        (stage_dir / "_sweep_output_root.txt").write_text(str(output_root) + "\n")

    if skip:
        print("  --skip-rollouts: skipping rollout stage")
        return output_root

    cache.run_or_hydrate(
        "rollout",
        rollout_id,
        do_rollouts,
        config={
            "adapter_ref": cfg.ADAPTER_REF,
            "base_model": cfg.BASE_MODEL,
            "scale_points": scales,
            "max_samples": cfg.MAX_SAMPLES,
            "seed": cfg.SEED,
        },
    )
    return output_root


# ---------------------------------------------------------------------------
# Stage: convert
# ---------------------------------------------------------------------------


def _get_sweep_output_root(cfg: ModuleType) -> Path:
    """Return the non-zero sweep output root from the sweep OutputPathConfig."""
    return _build_sweep_output_path_config(cfg).scratch_dir


def _get_baseline_output_root(cfg: ModuleType) -> Path:
    """Return the baseline cell's output root."""
    return _build_baseline_output_path_config(cfg).scratch_dir


def _run_convert_stage(
    cfg: ModuleType,
    cache: StageCache,
    convert_id: str,
    rollout_id: str,
    baseline_root: Path,
    sweep_root: Path,
    *,
    skip: bool,
) -> Path:
    """Merge baseline + sweep rollouts into one judge-compatible dataset.

    Returns:
        Path to the all_responses.jsonl file.
    """
    judge_dataset_path = sweep_root / "exports" / "all_responses.jsonl"

    def do_convert() -> None:
        from scripts_dev.persona_metrics.llm_judge.rollout_sweep_to_judge_dataset import (
            convert_sweeps,
        )

        sources: list[tuple[Path, list[float]]] = []
        if _BASELINE_SCALE in cfg.SCALE_POINTS:
            sources.append((baseline_root, [_BASELINE_SCALE]))
        nonzero = _nonzero_scale_points(cfg)
        if nonzero:
            sources.append((sweep_root, nonzero))

        n_rows = convert_sweeps(
            sources,
            judge_dataset_path,
            assistant_model=cfg.BASE_MODEL,
        )
        if n_rows <= 0:
            raise RuntimeError(
                f"No rollout rows were converted from {baseline_root} + {sweep_root}"
            )

        stage_dir = cache.stage_dir("convert", convert_id)
        (stage_dir / "_judge_dataset_path.txt").write_text(str(judge_dataset_path) + "\n")

    if skip:
        print("  --skip-rollouts: skipping convert stage")
    else:
        cache.run_or_hydrate(
            "convert",
            convert_id,
            do_convert,
            config={
                "baseline_root": str(baseline_root),
                "sweep_root": str(sweep_root),
            },
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
        if metric_name == trait_metric:
            axis.set_ylim(-4, 4)
        elif metric_name == coherence_metric:
            axis.set_ylim(0, 10)
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


def _upload_derived_outputs_to_hf(cfg: ModuleType, sweep_root: Path) -> None:
    """Upload plots, analysis, exports, judge_runs to the sweep's HF path.

    The rollout cells are already uploaded by ``run_sweep``; the StageCache
    only uploads marker files. Without this step the rich outputs (PNG plot,
    per-scale summary, merged judge dataset, raw judge responses) only exist
    locally.
    """
    from src_dev.utils.hf_hub import upload_folder_to_dataset_repo

    if not sweep_root.exists():
        return

    hf_path = _judge_hf_base_path(cfg)
    try:
        upload_folder_to_dataset_repo(
            local_dir=sweep_root,
            repo_id=HF_REPO_ID,
            path_in_repo=hf_path,
            commit_message=f"{cfg.EVAL_NAME}: upload plots/analysis/exports/judge_runs",
            allow_patterns=[
                "plots/**",
                "analysis/**",
                "exports/**",
                "judge_runs/**",
                "sweep_config.json",
                "sweep.log",
            ],
        )
        print(f"  [upload] pushed derived outputs to {HF_REPO_ID}/{hf_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [upload] skipped ({type(exc).__name__}: {exc})")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def _judge_hf_base_path(cfg: ModuleType) -> str:
    """HF prefix for judge stage artifacts — mirrors the OCT eval layout."""
    return (
        f"fine_tuning/{cfg.BASE_MODEL_SLUG}/{_CATEGORY}/{cfg.TRAIT.value}"
        f"/{cfg.DIRECTION}/{cfg.VERSION}/evals/{_EVAL_NAME}"
    )


def _print_dry_run(cfg: ModuleType, *, use_vllm: bool, upload: bool) -> None:
    """Print estimated work without executing anything."""
    n_conditions = 1
    n_responses = (
        len(cfg.SCALE_POINTS)
        * n_conditions
        * cfg.MAX_SAMPLES
        * cfg.NUM_ROLLOUTS_PER_PROMPT
    )
    n_judge_calls = n_responses * 2 * len(cfg.JUDGE_RATERS) * cfg.JUDGE_REPEATS

    baseline_id = _baseline_run_id(cfg)
    rollout_id = _rollout_run_id(cfg)
    convert_id = _convert_run_id(cfg, rollout_id, baseline_id)
    judge_id = _judge_run_id(cfg, convert_id)

    baseline_cfg = _build_baseline_output_path_config(cfg)
    sweep_cfg = _build_sweep_output_path_config(cfg)

    print("DRY RUN: LLM-judge LoRA scale sweep")
    print(f"  eval name        : {cfg.EVAL_NAME}")
    print(f"  adapter          : {cfg.ADAPTER_REF}")
    print(f"  base model       : {cfg.BASE_MODEL}")
    print(f"  scales           : {cfg.SCALE_POINTS}")
    print(f"  baseline scale   : {_BASELINE_SCALE} (routed separately)")
    print(f"  sweep scales     : {_nonzero_scale_points(cfg)}")
    print(f"  prompts          : {cfg.MAX_SAMPLES} (seed={cfg.SEED})")
    print(f"  responses        : {n_responses}")
    print(f"  judge metrics    : {[cfg.TRAIT.v2_metric_name, cfg.COHERENCE_METRIC]}")
    print(f"  judge raters     : {[r.rater_id for r in cfg.JUDGE_RATERS]}")
    print(f"  judge repeats    : {cfg.JUDGE_REPEATS}")
    print(f"  judge calls      : {n_judge_calls}")
    print(f"  provider         : {'vllm' if use_vllm else 'local'}")
    print(
        f"  run IDs          : baseline={baseline_id}  rollout={rollout_id}  "
        f"convert={convert_id}  judge={judge_id}"
    )
    print(f"  baseline HF path : {baseline_cfg.hf_path}")
    print(f"  sweep HF path    : {sweep_cfg.hf_path}")
    print(f"  judge HF path    : {_judge_hf_base_path(cfg)}")
    print(f"  baseline local   : {baseline_cfg.scratch_dir}")
    print(f"  sweep local      : {sweep_cfg.scratch_dir}")
    print(f"  upload           : {upload}")


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

    # Rollout stages (baseline + sweep) use distinct HF base paths so that the
    # scale=0 cell is shared across sweeps for the same (base_model, dataset,
    # seed, gen params) fingerprint.
    baseline_cache = StageCache(
        StageCacheConfig(
            cache_root=Path("scratch/eval-cache"),
            hf_repo=HF_REPO_ID if upload else None,
            hf_base_path=(
                f"evals/llm-judge-sweep/_baseline/{_baseline_fingerprint(cfg)}"
            ),
            no_remote=not upload,
        )
    )
    sweep_cache = StageCache(
        StageCacheConfig(
            cache_root=Path("scratch/eval-cache"),
            hf_repo=HF_REPO_ID if upload else None,
            hf_base_path=_judge_hf_base_path(cfg),
            no_remote=not upload,
        )
    )

    # Compute chained run IDs.
    baseline_id = _baseline_run_id(cfg)
    rollout_id = _rollout_run_id(cfg)
    convert_id = _convert_run_id(cfg, rollout_id, baseline_id)
    judge_id = _judge_run_id(cfg, convert_id)

    print(
        f"Run IDs: baseline={baseline_id}  rollout={rollout_id}  "
        f"convert={convert_id}  judge={judge_id}"
    )

    # Stage 1a: baseline (scale=0 at per-category shared path)
    baseline_root = _run_baseline_stage(
        cfg,
        baseline_cache,
        baseline_id,
        use_vllm=use_vllm,
        skip=flags.skip_rollouts,
    )

    # Stage 1b: rollout (non-zero scales at per-trait/direction/version path)
    sweep_root = _run_rollout_stage(
        cfg,
        sweep_cache,
        rollout_id,
        use_vllm=use_vllm,
        skip=flags.skip_rollouts,
    )

    # Hydrate convert/judge artifacts from HF so cache-hit branches see real
    # inputs.  StageCache only fetches its own stage dirs; these sibling dirs
    # live outside the cache and would otherwise be missing on fresh machines.
    _hydrate_derived_outputs_from_hf(
        sweep_cache, sweep_root, ["exports", "judge_runs"]
    )

    # Stage 2: convert (merge baseline + sweep)
    judge_dataset_path = _run_convert_stage(
        cfg,
        sweep_cache,
        convert_id,
        rollout_id,
        baseline_root,
        sweep_root,
        skip=False,
    )

    # Stage 3: judge
    results = _run_judge_stage(
        cfg,
        sweep_cache,
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

    # Stage 5: upload derived outputs (plots, analysis, exports, judge_runs)
    if upload:
        _upload_derived_outputs_to_hf(cfg, sweep_root)

    print(f"Done. Local sweep output   : {sweep_root}")
    print(f"Done. Local baseline output: {baseline_root}")


if __name__ == "__main__":
    main()
