#!/usr/bin/env python3
"""Scaling-grid eval runner: 2D LoRA-scale sweep with logprobs TRAIT scoring.

Stages:
    1. download_adapters — always runs (cheap HF download, idempotent)
    2. grid_sweep        — cached via StageCache; inner loop with per-combo resume
    3. plot              — always re-runs locally (cheap, layout may change)

Usage:
    uv run python -m scripts_dev.evals.scaling_grid.runner \
        --config scripts_dev.evals.scaling_grid.configs.trait_ac_minus_vanton1

    uv run python -m scripts_dev.evals.scaling_grid.runner \
        --config scripts_dev.evals.scaling_grid.configs.trait_ac_minus_vanton1 \
        --dry-run
"""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from dotenv import load_dotenv

from src_dev.eval_stages import StageCache, StageCacheConfig, chained_run_id, seed_all
from src_dev.utils.hf_hub import download_from_dataset_repo

load_dotenv()

# ---------------------------------------------------------------------------
# CLI flags (operational only -- config values come from the config module)
# ---------------------------------------------------------------------------


def parse_flags() -> argparse.Namespace:
    """Parse operational CLI flags.

    Config values (model, scales, samples, etc.) come from the config module
    specified via ``--config``.
    """
    parser = argparse.ArgumentParser(
        description="Scaling-grid eval: 2D LoRA-scale sweep with TRAIT scoring.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Dotted module path to the config file, e.g. "
            "scripts_dev.evals.scaling_grid.configs.trait_ac_minus_vanton1"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print run IDs and exit without running anything.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Disable HF upload/download (local-only mode).",
    )
    return parser.parse_args()


def load_config(dotted_path: str) -> ModuleType:
    """Import and return a config module by its dotted path.

    Args:
        dotted_path: Python dotted module path, e.g.
            ``scripts_dev.evals.scaling_grid.configs.trait_ac_minus_vanton1``.

    Returns:
        The imported config module.
    """
    return importlib.import_module(dotted_path)


# ---------------------------------------------------------------------------
# Run ID fields
# ---------------------------------------------------------------------------


def _grid_config_fields(config: ModuleType) -> dict[str, Any]:
    """Build the dict of config fields that materially affect grid_sweep output.

    Excludes operational settings like batch_size and dtype.
    """
    return {
        "base_model": config.BASE_MODEL,
        "adapters": config.ADAPTER_PATHS,
        "scales": config.SCALES,
        "trait_splits": config.TRAIT_SPLITS,
        "samples_per_trait": config.SAMPLES_PER_TRAIT,
        "seed": config.SEED,
        "temperature": config.TEMPERATURE,
        "max_tokens": config.MAX_TOKENS,
        "benchmark": config.BENCHMARK,
    }


# ---------------------------------------------------------------------------
# Helpers (preserved from original script)
# ---------------------------------------------------------------------------


def _format_scale(scale: float) -> str:
    """Format a scale value into a filesystem-safe string."""
    return f"{scale:+.1f}".replace("-", "m").replace("+", "p").replace(".", "p")


def _combo_name(a_scale: float, c_scale: float) -> str:
    """Build a deterministic combo name for a given (a, c) scale pair."""
    return f"a_{_format_scale(a_scale)}__c_{_format_scale(c_scale)}"


def _download_adapter(path_in_repo: str, hf_repo: str, cache_root: Path) -> Path:
    """Download a single adapter from the HF monorepo.

    Args:
        path_in_repo: Path within the HF dataset repo.
        hf_repo: HuggingFace dataset repo ID.
        cache_root: Local cache root for downloaded adapters.

    Returns:
        Path to the downloaded adapter directory.
    """
    download_from_dataset_repo(
        repo_id=hf_repo,
        path_in_repo=path_in_repo,
        local_dir=cache_root,
    )
    return cache_root / path_in_repo


def _build_trait_spec(
    *,
    benchmark: str,
    samples_per_trait: int,
    trait_splits: list[str],
    max_tokens: int,
) -> Any:
    """Build the Inspect benchmark spec for TRAIT evaluation."""
    from src_dev.evals.config import InspectBenchmarkSpec

    return InspectBenchmarkSpec(
        name="trait",
        benchmark=benchmark,
        benchmark_args={
            "samples_per_trait": samples_per_trait,
            "trait_splits": trait_splits,
            "max_tokens": max_tokens,
        },
        n_runs=1,
    )


def _load_model_with_adapters(
    *,
    base_model: str,
    adapter_paths: list[Path],
    dtype_name: str,
) -> tuple[Any, Any, list[str], list[str]]:
    """Load the base model, attach LoRA adapters, and return the model objects.

    Args:
        base_model: HuggingFace model ID for the base model.
        adapter_paths: Local paths to the LoRA adapter directories.
        dtype_name: Torch dtype name (e.g. "bfloat16").

    Returns:
        Tuple of (peft_model, tokenizer, adapter_names, resolved_refs).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.utils.peft_manipulations import set_active_adapters
    from src_dev.evals.config import AdapterConfig
    from src_dev.evals.model_resolution import resolve_model_reference
    from src_dev.evals.suite import _flash_attn_kwargs
    from src_dev.evals.utils.preloaded_hf_provider import register_preloaded_hf_provider
    from src_dev.utils.lora_composition import load_and_scale_adapters

    register_preloaded_hf_provider()

    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported dtype: {dtype_name!r}")

    base_ref = resolve_model_reference(base_model, kind="base model")
    base_hf_model = AutoModelForCausalLM.from_pretrained(
        base_ref,
        torch_dtype=dtype,
        device_map="auto",
        **_flash_attn_kwargs(),
    )

    # Keep adapters loaded at their native scale so each combo-specific
    # LoRaPipeline snapshots the original weights/scaling and can apply the
    # requested A/C factors from that baseline.  Loading them at 0.0 would
    # permanently zero the saved baseline and collapse the whole grid to base.
    adapter_cfgs = [
        AdapterConfig(path=f"local://{path.resolve()}", scale=1.0)
        for path in adapter_paths
    ]
    peft_model, adapter_names, resolved_refs = load_and_scale_adapters(
        base_hf_model,
        adapters=adapter_cfgs,
        adapter_name_prefix="combo_adapter",
        adapter_resolver=lambda ref: resolve_model_reference(ref, kind="adapter"),
    )
    set_active_adapters(peft_model, adapter_names)

    try:
        tokenizer = AutoTokenizer.from_pretrained(resolved_refs[0])
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_ref)

    return peft_model, tokenizer, adapter_names, resolved_refs


def _write_run_info(
    *,
    run_dir: Path,
    status: str,
    error: str | None,
    inspect_log_path: str | None,
    inspect_status: str | None,
    a_scale: float,
    c_scale: float,
    a_adapter_ref: str,
    c_adapter_ref: str,
    base_model: str,
    spec: Any,
    seed: int,
) -> Path:
    """Write per-combo run_info.json for resume within the grid.

    Args:
        run_dir: Directory for this combo's eval outputs.
        status: Overall run status ("ok" or "error").
        error: Error message if status is not "ok".
        inspect_log_path: Path to the Inspect log file.
        inspect_status: Inspect evaluation status string.
        a_scale: Scale factor for the A adapter.
        c_scale: Scale factor for the C adapter.
        a_adapter_ref: Resolved reference for the A adapter.
        c_adapter_ref: Resolved reference for the C adapter.
        base_model: Base model identifier.
        spec: Inspect benchmark spec used.
        seed: Random seed used.

    Returns:
        Path to the written run_info.json file.
    """
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "error": error,
        "combo": {
            "agreeableness_scale": a_scale,
            "conscientiousness_scale": c_scale,
        },
        "model_spec": {
            "name": run_dir.parent.name,
            "base_model": base_model,
            "adapters": [
                {"path": a_adapter_ref, "scale": a_scale},
                {"path": c_adapter_ref, "scale": c_scale},
            ],
        },
        "eval_spec": spec.model_dump(mode="json"),
        "native": {
            "inspect_log_path": inspect_log_path,
            "inspect_status": inspect_status,
        },
        "seed": seed,
    }
    info_path = run_dir / "run_info.json"
    info_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return info_path


def _row_from_cached_run(run_info_path: Path) -> dict[str, Any] | None:
    """Try to extract a summary row from an existing per-combo run_info.json.

    Returns None if the run was not successful or scores cannot be parsed.
    """
    from src_dev.evals.personality.analyze_results import _extract_scores_reparsed

    info = json.loads(run_info_path.read_text(encoding="utf-8"))
    if info.get("status") != "ok":
        return None

    log_path = info.get("native", {}).get("inspect_log_path")
    if not log_path:
        return None

    reparsed = _extract_scores_reparsed(Path(log_path), "trait")
    if reparsed is None:
        return None
    scores, parse_rate = reparsed

    combo = info.get("combo", {})
    return {
        "combo_name": info.get("model_spec", {}).get("name"),
        "a_scale": combo.get("agreeableness_scale"),
        "c_scale": combo.get("conscientiousness_scale"),
        "status": "ok",
        "parse_rate": parse_rate,
        "inspect_log_path": log_path,
        "Agreeableness": scores.get("Agreeableness"),
        "Conscientiousness": scores.get("Conscientiousness"),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_heatmap(
    df: Any,
    *,
    trait: str,
    scales: list[float],
    output_path: Path,
) -> None:
    """Render and save a single trait heatmap.

    Args:
        df: DataFrame with columns a_scale, c_scale, and ``trait``.
        trait: Column name for the trait to visualize.
        scales: Sorted list of scale values (used for axes).
        output_path: File path to save the figure.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    pivot = (
        df.pivot(index="a_scale", columns="c_scale", values=trait)
        .reindex(index=scales, columns=scales)
    )
    values = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(values, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")

    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([f"{scale:+.1f}" for scale in scales])
    ax.set_yticks(range(len(scales)))
    ax.set_yticklabels([f"{scale:+.1f}" for scale in scales])
    ax.set_xlabel("C- scale")
    ax.set_ylabel("A- scale")
    ax.set_title(f"{trait} TRAIT heatmap")

    for row_idx, a_scale in enumerate(scales):
        for col_idx, c_scale in enumerate(scales):
            value = pivot.loc[a_scale, c_scale]
            if pd.isna(value):
                label = "NA"
                color = "white"
            else:
                label = f"{value:.3f}"
                color = "black" if value > 0.6 else "white"
            ax.text(
                col_idx, row_idx, label,
                ha="center", va="center", color=color, fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean TRAIT score")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmaps(
    grid_dir: Path,
    config: ModuleType,
    scales: list[float],
) -> None:
    """Re-render all trait heatmaps from the grid sweep results.

    Args:
        grid_dir: Path to the grid_sweep stage output directory.
        config: Config module with TRAIT_SPLITS.
        scales: Sorted scale values.
    """
    analysis_dir = grid_dir / "analysis"
    summary_path = analysis_dir / "combo_scores.jsonl"
    if not summary_path.exists():
        print("  No summary file found, skipping plots.")
        return

    import pandas as pd

    results_df = pd.read_json(summary_path, orient="records", lines=True)
    figures_dir = grid_dir / "figures"

    for trait in config.TRAIT_SPLITS:
        trait_slug = trait.lower()
        _plot_heatmap(
            results_df,
            trait=trait,
            scales=scales,
            output_path=figures_dir / f"{trait_slug}_heatmap.png",
        )
    print(f"  Saved figures to: {figures_dir}")


# ---------------------------------------------------------------------------
# Stage 1: download_adapters (always runs, idempotent)
# ---------------------------------------------------------------------------


def download_adapters(config: ModuleType) -> dict[str, Path]:
    """Download all adapters specified in the config.

    Args:
        config: Config module with ADAPTER_PATHS and HF_DATASET_REPO.

    Returns:
        Dict mapping adapter key (e.g. "a", "c") to local path.
    """
    cache_root = Path("scratch/adapters/monorepo")
    adapter_locals: dict[str, Path] = {}
    for key, path_in_repo in config.ADAPTER_PATHS.items():
        local = _download_adapter(path_in_repo, config.HF_DATASET_REPO, cache_root)
        adapter_locals[key] = local
        print(f"  {key} adapter cached at: {local}")
    return adapter_locals


# ---------------------------------------------------------------------------
# Stage 2: grid_sweep (cached via StageCache)
# ---------------------------------------------------------------------------


def run_grid(
    *,
    config: ModuleType,
    adapter_locals: dict[str, Path],
    output_dir: Path,
    scales: list[float],
) -> None:
    """Run the full grid sweep, writing results into output_dir.

    Uses per-combo ``run_info.json`` files for resume within the grid.
    If a combo already has a successful ``run_info.json``, it is skipped.

    Args:
        config: Config module.
        adapter_locals: Dict mapping adapter key to local path.
        output_dir: Directory to write combo results and summary files.
        scales: Sorted list of scale values.
    """
    import pandas as pd
    from inspect_ai.model import get_model

    from src.utils.peft_manipulations import LoRaPipeline, LoRaScaling
    from src_dev.evals.backends.inspect_runner import run_benchmark_eval
    from src_dev.evals.personality.analyze_results import _extract_scores_reparsed
    from src_dev.evals.inspect_benchmarks import build_benchmark_task
    from src_dev.evals.suite import _cleanup_runtime_model_state

    trait_spec = _build_trait_spec(
        benchmark=config.BENCHMARK,
        samples_per_trait=config.SAMPLES_PER_TRAIT,
        trait_splits=config.TRAIT_SPLITS,
        max_tokens=config.MAX_TOKENS,
    )
    # Build task once so every (a_scale, c_scale) combo evaluates on the
    # exact same questions with the same shuffle order.
    trait_task = build_benchmark_task(trait_spec)

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = analysis_dir / "combo_scores.jsonl"

    # Load any previously collected rows (from partial runs)
    rows_by_combo: dict[str, dict[str, Any]] = {}
    if summary_path.exists():
        cached_df = pd.read_json(summary_path, orient="records", lines=True)
        if not cached_df.empty:
            for row in cached_df.to_dict(orient="records"):
                rows_by_combo[str(row["combo_name"])] = row

    # NOTE: This grid sweep currently assumes exactly two adapters keyed "a"
    # (agreeableness) and "c" (conscientiousness).  The combo naming, run_info
    # schema, and heatmap plotting all depend on this.  Generalizing to N
    # adapters would require reworking _combo_name, _write_run_info, and the
    # plotting code.
    adapter_keys = sorted(adapter_locals.keys())
    adapter_path_list = [adapter_locals[k] for k in adapter_keys]

    peft_model = None
    tokenizer = None
    adapter_names: list[str] = []
    resolved_refs: list[str] = []

    try:
        peft_model, tokenizer, adapter_names, resolved_refs = _load_model_with_adapters(
            base_model=config.BASE_MODEL,
            adapter_paths=adapter_path_list,
            dtype_name=config.DTYPE,
        )
        assert len(adapter_names) == 2, (
            f"Expected exactly 2 adapters, got {len(adapter_names)}: {adapter_names}"
        )
        a_adapter_name, c_adapter_name = adapter_names
        a_adapter_ref, c_adapter_ref = resolved_refs

        for a_scale in scales:
            for c_scale in scales:
                combo = _combo_name(a_scale, c_scale)
                run_dir = output_dir / combo / "trait"
                run_dir.mkdir(parents=True, exist_ok=True)
                run_info_path = run_dir / "run_info.json"

                # Per-combo resume: skip if already completed
                if run_info_path.exists():
                    cached_row = _row_from_cached_run(run_info_path)
                    if cached_row is not None:
                        rows_by_combo[combo] = cached_row
                        print(f"  Skipping completed combo: {combo}")
                        continue

                print(f"  Running combo: {combo} (A={a_scale:+.1f}, C={c_scale:+.1f})")
                pipeline = LoRaPipeline(
                    peft_model,
                    steps=[
                        (LoRaScaling, a_adapter_name, {"scale_factor": a_scale}),
                        (LoRaScaling, c_adapter_name, {"scale_factor": c_scale}),
                    ],
                )

                try:
                    pipeline.apply()
                    inspect_model = get_model(
                        f"hf_preloaded/{combo}",
                        hf_model=peft_model,
                        hf_tokenizer=tokenizer,
                        batch_size=config.BATCH_SIZE,
                    )
                    result = run_benchmark_eval(
                        spec=trait_spec,
                        model_uri=inspect_model,
                        run_dir=run_dir,
                        temperature=config.TEMPERATURE,
                        task=trait_task,
                    )

                    inspect_log_path = result.log.location if result.log else None
                    inspect_status = result.log.status if result.log else None
                    _write_run_info(
                        run_dir=run_dir,
                        status=result.status,
                        error=result.error,
                        inspect_log_path=inspect_log_path,
                        inspect_status=inspect_status,
                        a_scale=a_scale,
                        c_scale=c_scale,
                        a_adapter_ref=a_adapter_ref,
                        c_adapter_ref=c_adapter_ref,
                        base_model=config.BASE_MODEL,
                        spec=trait_spec,
                        seed=config.SEED,
                    )

                    if result.status != "ok" or inspect_log_path is None:
                        print(f"    failed: {result.error}")
                        continue

                    reparsed = _extract_scores_reparsed(
                        Path(inspect_log_path), "trait",
                    )
                    if reparsed is None:
                        print("    failed: could not parse TRAIT scores from inspect log")
                        continue
                    scores, parse_rate = reparsed
                    rows_by_combo[combo] = {
                        "combo_name": combo,
                        "a_scale": a_scale,
                        "c_scale": c_scale,
                        "status": "ok",
                        "parse_rate": parse_rate,
                        "inspect_log_path": inspect_log_path,
                        "Agreeableness": scores.get("Agreeableness"),
                        "Conscientiousness": scores.get("Conscientiousness"),
                    }
                    print(
                        f"    scores:"
                        f" A={rows_by_combo[combo]['Agreeableness']:.3f}"
                        f" C={rows_by_combo[combo]['Conscientiousness']:.3f}"
                        f" parse_rate={parse_rate:.3f}"
                    )
                finally:
                    pipeline.restore()
                    _cleanup_runtime_model_state(move_to_cpu=False)

    finally:
        if peft_model is not None:
            try:
                peft_model.cpu()
            except Exception:
                pass
        _cleanup_runtime_model_state(move_to_cpu=True)

    if not rows_by_combo:
        raise RuntimeError("No successful combo results were collected.")

    results_df = (
        pd.DataFrame(rows_by_combo.values())
        .sort_values(["a_scale", "c_scale"])
        .reset_index(drop=True)
    )
    results_df.to_json(summary_path, orient="records", lines=True)
    results_df.to_csv(analysis_dir / "combo_scores.csv", index=False)
    print(f"  Saved summary table to: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: parse flags, load config, run stages."""
    flags = parse_flags()
    config = load_config(flags.config)

    # Seed everything from config
    seed_all(config.SEED)

    scales = sorted({round(s, 10) for s in config.SCALES})
    if not scales:
        raise ValueError("At least one scale must be provided.")

    # Compute deterministic run ID for the grid sweep stage
    grid_fields = _grid_config_fields(config)
    grid_id = chained_run_id("grid_sweep", grid_fields)

    cache = StageCache(StageCacheConfig(
        cache_root=Path("scratch/eval-cache"),
        hf_repo=config.HF_DATASET_REPO,
        hf_base_path=f"evals/scaling-grid/{config.EVAL_NAME}",
        no_remote=flags.no_upload,
    ))

    if flags.dry_run:
        print(f"Config: {flags.config}")
        print(f"EVAL_NAME: {config.EVAL_NAME}")
        print(f"grid_sweep run_id: {grid_id}")
        print(f"grid_sweep stage_dir: {cache.stage_dir('grid_sweep', grid_id)}")
        print(f"grid_sweep hf_path: {cache.hf_path('grid_sweep', grid_id)}")
        return

    # Stage 1: download adapters (always runs, idempotent)
    print("== Stage 1: download_adapters ==")
    adapter_locals = download_adapters(config)

    # Stage 2: grid sweep (cached via StageCache)
    grid_dir = cache.stage_dir("grid_sweep", grid_id)

    def _run_grid() -> None:
        run_grid(
            config=config,
            adapter_locals=adapter_locals,
            output_dir=grid_dir,
            scales=scales,
        )

    cache.run_or_hydrate(
        "grid_sweep",
        grid_id,
        _run_grid,
        config=grid_fields,
        commit_message=f"scaling-grid/{config.EVAL_NAME} grid_sweep {grid_id}",
    )

    # Stage 3: plot (always re-runs, cheap)
    print("== Stage 3: plot ==")
    _plot_heatmaps(grid_dir, config, scales)

    print("Done.")


if __name__ == "__main__":
    main()
