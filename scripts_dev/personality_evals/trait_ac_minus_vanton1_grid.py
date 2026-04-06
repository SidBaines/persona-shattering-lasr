#!/usr/bin/env python3
"""Run a 2D TRAIT sweep over A- and C- LoRA combinations, then plot heatmaps.

This script:
1. Downloads the requested A- and C- adapters from the shared HF monorepo.
2. Loads the base model and both adapters once.
3. Sweeps the 25 combinations of A- and C- scales in {-1.0, -0.5, 0.0, 0.5, 1.0}.
4. Runs the sampled TRAIT benchmark on Agreeableness and Conscientiousness only.
5. Writes per-combo metadata/log references plus a summary table.
6. Saves one heatmap per evaluated trait.

Usage:
    uv run python -m scripts_dev.personality_evals.trait_ac_minus_vanton1_grid

    uv run python -m scripts_dev.personality_evals.trait_ac_minus_vanton1_grid \
        --samples-per-trait 100 \
        --run-name a_c_minus_grid_debug
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from inspect_ai.model import get_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.peft_manipulations import LoRaPipeline, LoRaScaling, set_active_adapters
from src_dev.evals.backends.inspect_runner import run_benchmark_eval
from src_dev.evals.config import AdapterConfig, InspectBenchmarkSpec
from src_dev.evals.model_resolution import resolve_model_reference
from src_dev.evals.personality.analyze_results import _extract_scores_reparsed
from src_dev.evals.suite import _cleanup_runtime_model_state, _flash_attn_kwargs
from src_dev.evals.utils.preloaded_hf_provider import register_preloaded_hf_provider
from src_dev.utils.hf_hub import download_from_dataset_repo
from src_dev.utils.lora_composition import load_and_scale_adapters

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

load_dotenv()

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_DATASET_REPO = "persona-shattering-lasr/monorepo"

A_PATH_IN_REPO = (
    "fine_tuning/llama-3.1-8b-it/ocean/agreeableness/suppressor/vanton1/"
    "lora/agreeableness_suppressing_full_vanton1-persona"
)
C_PATH_IN_REPO = (
    "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/vanton1/"
    "lora/conscientiousness_suppressing_full_vanton1-persona"
)

DEFAULT_SCALES = [-1.0, -0.5, 0.0, 0.5, 1.0]
TRAIT_SPLITS = ["Agreeableness", "Conscientiousness"]
DEFAULT_RUN_NAME = "trait_ac_minus_vanton1_grid"
DEFAULT_OUTPUT_ROOT = Path("scratch/evals/ocean/trait_combo")
DEFAULT_CACHE_ROOT = Path("scratch/adapters/monorepo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 2D TRAIT sweep over agreeableness- and conscientiousness- "
            "LoRA combinations, then render heatmaps."
        )
    )
    parser.add_argument(
        "--samples-per-trait",
        type=int,
        default=300,
        help="Number of TRAIT questions to evaluate for each selected trait split.",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=DEFAULT_SCALES,
        help="Scale grid to use for both A- and C-.",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="Run directory name under output-root. Existing completed combos are reused.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory for run outputs.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="Local cache root for downloaded adapters.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Inspect generation batch size.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Generation max_tokens for the TRAIT benchmark.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Torch dtype name for model loading.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature for benchmark runs.",
    )
    return parser.parse_args()


def _format_scale(scale: float) -> str:
    return f"{scale:+.1f}".replace("-", "m").replace("+", "p").replace(".", "p")


def _combo_name(a_scale: float, c_scale: float) -> str:
    return f"a_{_format_scale(a_scale)}__c_{_format_scale(c_scale)}"


def _download_adapter(path_in_repo: str, cache_root: Path) -> Path:
    download_from_dataset_repo(
        repo_id=HF_DATASET_REPO,
        path_in_repo=path_in_repo,
        local_dir=cache_root,
    )
    return cache_root / path_in_repo


def _build_trait_spec(samples_per_trait: int, max_tokens: int) -> InspectBenchmarkSpec:
    return InspectBenchmarkSpec(
        name="trait",
        benchmark="personality_trait_sampled",
        benchmark_args={
            "samples_per_trait": samples_per_trait,
            "trait_splits": TRAIT_SPLITS,
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
    # requested A/C factors from that baseline. Loading them at 0.0 would
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
    spec: InspectBenchmarkSpec,
) -> Path:
    payload = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "status": status,
        "error": error,
        "combo": {
            "agreeableness_scale": a_scale,
            "conscientiousness_scale": c_scale,
        },
        "model_spec": {
            "name": run_dir.parent.name,
            "base_model": BASE_MODEL,
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
        "seed": SEED,
    }
    info_path = run_dir / "run_info.json"
    info_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return info_path


def _load_cached_summary(summary_path: Path) -> pd.DataFrame:
    if not summary_path.exists():
        return pd.DataFrame()
    return pd.read_json(summary_path, orient="records", lines=True)


def _row_from_cached_run(run_info_path: Path) -> dict[str, Any] | None:
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


def _plot_heatmap(
    df: pd.DataFrame,
    *,
    trait: str,
    scales: list[float],
    output_path: Path,
) -> None:
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
            ax.text(col_idx, row_idx, label, ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean TRAIT score")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    scales = sorted({round(scale, 10) for scale in args.scales})
    if not scales:
        raise ValueError("At least one scale must be provided.")

    run_root = args.output_root / args.run_name
    figures_dir = run_root / "figures"
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_root}")

    a_local = _download_adapter(A_PATH_IN_REPO, args.cache_root)
    c_local = _download_adapter(C_PATH_IN_REPO, args.cache_root)
    print(f"A- adapter cached at: {a_local}")
    print(f"C- adapter cached at: {c_local}")

    trait_spec = _build_trait_spec(
        samples_per_trait=args.samples_per_trait,
        max_tokens=args.max_tokens,
    )

    summary_path = analysis_dir / "combo_scores.jsonl"
    cached_df = _load_cached_summary(summary_path)
    rows_by_combo: dict[str, dict[str, Any]] = {}
    if not cached_df.empty:
        for row in cached_df.to_dict(orient="records"):
            rows_by_combo[str(row["combo_name"])] = row

    peft_model = None
    tokenizer = None
    adapter_names: list[str] = []
    resolved_refs: list[str] = []

    try:
        peft_model, tokenizer, adapter_names, resolved_refs = _load_model_with_adapters(
            base_model=BASE_MODEL,
            adapter_paths=[a_local, c_local],
            dtype_name=args.dtype,
        )
        a_adapter_name, c_adapter_name = adapter_names
        a_adapter_ref, c_adapter_ref = resolved_refs

        for a_scale in scales:
            for c_scale in scales:
                combo_name = _combo_name(a_scale, c_scale)
                run_dir = run_root / combo_name / "trait"
                run_dir.mkdir(parents=True, exist_ok=True)
                run_info_path = run_dir / "run_info.json"

                if run_info_path.exists():
                    cached_row = _row_from_cached_run(run_info_path)
                    if cached_row is not None:
                        rows_by_combo[combo_name] = cached_row
                        print(f"Skipping completed combo: {combo_name}")
                        continue

                print(f"Running combo: {combo_name} (A={a_scale:+.1f}, C={c_scale:+.1f})")
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
                        f"hf_preloaded/{combo_name}",
                        hf_model=peft_model,
                        hf_tokenizer=tokenizer,
                        batch_size=args.batch_size,
                    )
                    result = run_benchmark_eval(
                        spec=trait_spec,
                        model_uri=inspect_model,
                        run_dir=run_dir,
                        temperature=args.temperature,
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
                        spec=trait_spec,
                    )

                    if result.status != "ok" or inspect_log_path is None:
                        print(f"  failed: {result.error}")
                        continue

                    reparsed = _extract_scores_reparsed(Path(inspect_log_path), "trait")
                    if reparsed is None:
                        print("  failed: could not parse TRAIT scores from inspect log")
                        continue
                    scores, parse_rate = reparsed
                    rows_by_combo[combo_name] = {
                        "combo_name": combo_name,
                        "a_scale": a_scale,
                        "c_scale": c_scale,
                        "status": "ok",
                        "parse_rate": parse_rate,
                        "inspect_log_path": inspect_log_path,
                        "Agreeableness": scores.get("Agreeableness"),
                        "Conscientiousness": scores.get("Conscientiousness"),
                    }
                    print(
                        "  scores:"
                        f" A={rows_by_combo[combo_name]['Agreeableness']:.3f}"
                        f" C={rows_by_combo[combo_name]['Conscientiousness']:.3f}"
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

    results_df = pd.DataFrame(rows_by_combo.values()).sort_values(["a_scale", "c_scale"]).reset_index(drop=True)
    results_df.to_json(summary_path, orient="records", lines=True)
    results_df.to_csv(analysis_dir / "combo_scores.csv", index=False)

    for trait in TRAIT_SPLITS:
        trait_slug = trait.lower()
        _plot_heatmap(
            results_df,
            trait=trait,
            scales=scales,
            output_path=figures_dir / f"{trait_slug}_heatmap.png",
        )

    print(f"Saved summary table to: {summary_path}")
    print(f"Saved figures to: {figures_dir}")


if __name__ == "__main__":
    main()
