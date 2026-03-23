"""Run TRAIT evals for OCT model variants and plot a grouped bar chart.

This is a lightweight wrapper around the existing Inspect-based eval suite in
``src_dev.evals``. It evaluates four OCT model variants:

1. Base model
2. DPO adapter at LoRA scale 1.0
3. SFT adapter at LoRA scale 1.0
4. Combined/persona adapter at LoRA scale 1.0

It then loads the resulting TRAIT scores and writes:
- a long-form CSV
- a wide-form CSV
- a grouped bar chart PNG

Example:
    python scripts/experiments/oct_pipeline/eval_trait_models.py \
        --out-dir scratch/oct_runs/conscientiousness_low-llama-3.1-8b-it-s123456-abcdef123456 \
        --model-path /root/.cache/models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd
from dotenv import load_dotenv

from src_dev.evals import AdapterConfig, InspectBenchmarkSpec, ModelSpec, SuiteConfig, run_eval_suite
from src_dev.evals.personality.analyze_results import ALL_TRAIT_COLS, load_sweep_data

matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv()


DEFAULT_SAMPLES_PER_TRAIT = 1000
DEFAULT_OUTPUT_ROOT = Path("scratch/evals/personality")
DEFAULT_TEMPERATURE = 0.6
DEFAULT_BATCH_SIZE = 8

MODEL_VARIANTS = (
    ("base", "Base model", None),
    ("dpo_scale_1", "DPO, scale 1", "dpo"),
    ("sft_scale_1", "SFT, scale 1", "sft"),
    ("combined_scale_1", "Combined, scale 1", "persona"),
)

MODEL_COLORS = {
    "Base model": "#37474f",
    "DPO, scale 1": "#1565c0",
    "SFT, scale 1": "#2e7d32",
    "Combined, scale 1": "#ef6c00",
}


def _run_config_path(out_dir: Path) -> Path:
    return out_dir / ".oct_pipeline" / "run_config.json"


def _load_oct_run_config(out_dir: Path) -> dict:
    config_path = _run_config_path(out_dir)
    if not config_path.exists():
        raise FileNotFoundError(
            f"OCT run config not found at {config_path}. Pass an OCT --out-dir from run_oct_pipeline.py."
        )
    payload = json.loads(config_path.read_text())
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Malformed OCT run config at {config_path}")
    return config


def _resolve_base_model_path(out_dir: Path, model_root: Path | None, base_model: str | None) -> tuple[str, str]:
    config = _load_oct_run_config(out_dir)
    model_name = base_model or config.get("model")
    if not model_name:
        raise ValueError("Could not determine base model name from OCT run config; pass --base-model.")

    if model_root is None:
        model_root = Path("/workspace/models")

    base_model_path = model_root / model_name
    if not base_model_path.is_dir():
        raise FileNotFoundError(
            f"Base model directory not found: {base_model_path}\n"
            "Pass --model-path <parent_dir> or --base-model if this OCT run used a different model layout."
        )
    return model_name, f"local://{base_model_path}"


def _resolve_constitution(out_dir: Path, constitution: str | None) -> str:
    if constitution:
        return constitution
    config = _load_oct_run_config(out_dir)
    config_constitution = config.get("constitution")
    if not config_constitution:
        raise ValueError(
            "Could not determine constitution from OCT run config; pass --constitution explicitly."
        )
    return str(config_constitution)


def _adapter_path(out_dir: Path, constitution: str, kind: str) -> Path:
    return out_dir / "lora" / f"{constitution}-{kind}"


def _build_model_specs(base_model_path: str, out_dir: Path, constitution: str) -> list[ModelSpec]:
    specs: list[ModelSpec] = []

    for spec_name, label, adapter_kind in MODEL_VARIANTS:
        if adapter_kind is None:
            specs.append(ModelSpec(name=spec_name, base_model=base_model_path))
            continue

        adapter_path = _adapter_path(out_dir, constitution, adapter_kind)
        if not adapter_path.is_dir():
            raise FileNotFoundError(
                f"Expected OCT adapter for '{label}' at {adapter_path}, but it was not found."
            )
        specs.append(
            ModelSpec(
                name=spec_name,
                base_model=base_model_path,
                adapters=[AdapterConfig(path=f"local://{adapter_path}", scale=1.0)],
                scale=1.0,
            )
        )

    return specs


def _make_run_name(out_dir: Path, constitution: str, samples_per_trait: int, run_name: str | None) -> str:
    if run_name:
        return run_name
    return f"eval_oct_{constitution}_{out_dir.name}_trait_k{samples_per_trait}"


def _build_suite_config(
    *,
    base_model_path: str,
    out_dir: Path,
    constitution: str,
    samples_per_trait: int,
    output_root: Path,
    run_name: str,
    temperature: float,
    batch_size: int,
    skip_completed: bool,
) -> SuiteConfig:
    return SuiteConfig(
        models=_build_model_specs(base_model_path=base_model_path, out_dir=out_dir, constitution=constitution),
        evals=[
            InspectBenchmarkSpec(
                name="trait",
                benchmark="personality_trait_sampled",
                benchmark_args={"samples_per_trait": samples_per_trait},
            )
        ],
        temperature=temperature,
        batch_size=batch_size,
        output_root=output_root,
        run_name=run_name,
        skip_completed=skip_completed,
        metadata={
            "oct_out_dir": str(out_dir),
            "constitution": constitution,
            "samples_per_trait": samples_per_trait,
        },
    )


def _extract_trait_table(run_dir: Path) -> pd.DataFrame:
    data = load_sweep_data(run_dir, reparse=False)
    trait_df = data.get("trait")
    if trait_df is None or trait_df.empty:
        raise ValueError(f"No TRAIT results found under {run_dir}")

    expected_names = {spec_name for spec_name, _, _ in MODEL_VARIANTS}
    observed = set(trait_df["model"].unique())
    missing = expected_names - observed
    if missing:
        raise ValueError(f"Missing TRAIT results for model specs: {sorted(missing)}")

    label_map = {spec_name: label for spec_name, label, _ in MODEL_VARIANTS}
    records: list[dict[str, object]] = []
    for spec_name, label, _ in MODEL_VARIANTS:
        subset = trait_df[trait_df["model"] == spec_name]
        if subset.empty:
            continue
        row = subset.iloc[0]
        for trait in ALL_TRAIT_COLS:
            if trait not in subset.columns:
                raise ValueError(f"TRAIT metric '{trait}' missing from results under {run_dir}")
            records.append(
                {
                    "model_spec": spec_name,
                    "model_label": label_map[spec_name],
                    "trait": trait,
                    "score": float(row[trait]),
                    "parse_rate": float(row.get("_parse_rate", 1.0)),
                }
            )

    result = pd.DataFrame.from_records(records)
    if result.empty:
        raise ValueError(f"Could not build TRAIT summary table from {run_dir}")
    return result


def _assert_suite_succeeded(run_dir: Path, expected_eval_name: str = "trait") -> None:
    failed_run_infos: list[tuple[str, str]] = []
    ok_count = 0

    for spec_name, _, _ in MODEL_VARIANTS:
        run_info_path = run_dir / spec_name / expected_eval_name / "run_info.json"
        if not run_info_path.exists():
            failed_run_infos.append((spec_name, "missing run_info.json"))
            continue

        payload = json.loads(run_info_path.read_text())
        status = payload.get("status")
        if status == "ok":
            ok_count += 1
            continue

        error = payload.get("error") or payload.get("native", {}).get("inspect_status") or "unknown error"
        failed_run_infos.append((spec_name, str(error)))

    if ok_count == 0:
        details = "\n".join(f"  - {spec_name}: {error}" for spec_name, error in failed_run_infos)
        raise RuntimeError(
            "TRAIT eval suite did not produce any successful runs.\n"
            "Most likely the model or adapter references could not be loaded.\n"
            f"Run directory: {run_dir}\n"
            f"{details}"
        )


def _write_summaries(summary_df: pd.DataFrame, run_dir: Path) -> tuple[Path, Path]:
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    long_csv_path = analysis_dir / "trait_scores_by_model.csv"
    summary_df.to_csv(long_csv_path, index=False)

    wide_df = summary_df.pivot(index="trait", columns="model_label", values="score").reset_index()
    wide_csv_path = analysis_dir / "trait_scores_by_model_wide.csv"
    wide_df.to_csv(wide_csv_path, index=False)

    return long_csv_path, wide_csv_path


def _plot_trait_bars(summary_df: pd.DataFrame, run_dir: Path, title: str) -> Path:
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "trait_model_comparison_bar.png"

    wide_df = summary_df.pivot(index="trait", columns="model_label", values="score").reindex(ALL_TRAIT_COLS)
    model_labels = [label for _, label, _ in MODEL_VARIANTS]

    fig, ax = plt.subplots(figsize=(15, 7))
    x_positions = list(range(len(wide_df.index)))
    width = 0.2
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    for offset, model_label in zip(offsets, model_labels, strict=True):
        values = wide_df[model_label].tolist()
        ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=model_label,
            color=MODEL_COLORS[model_label],
        )

    ax.set_title(title)
    ax.set_ylabel("TRAIT score")
    ax.set_xlabel("Trait")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(wide_df.index, rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TRAIT evals for OCT base/DPO/SFT/combined models and plot a bar chart.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="OCT run output directory produced by scripts/experiments/oct_pipeline/run_oct_pipeline.py",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Parent directory containing the OCT base model folder (for example /root/.cache/models)",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Optional override for the base model folder name from the OCT run config",
    )
    parser.add_argument(
        "--constitution",
        default=None,
        help="Optional override for the constitution name from the OCT run config",
    )
    parser.add_argument(
        "--samples-per-trait",
        type=int,
        default=DEFAULT_SAMPLES_PER_TRAIT,
        help="Number of TRAIT benchmark samples per trait",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for eval outputs",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional explicit eval run name",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature forwarded to Inspect",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Generation batch size for the eval suite",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Rerun even if the target eval outputs already exist",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    model_root = Path(args.model_path).resolve() if args.model_path else None
    constitution = _resolve_constitution(out_dir=out_dir, constitution=args.constitution)
    model_name, base_model_path = _resolve_base_model_path(
        out_dir=out_dir,
        model_root=model_root,
        base_model=args.base_model,
    )
    output_root = Path(args.output_root)
    run_name = _make_run_name(
        out_dir=out_dir,
        constitution=constitution,
        samples_per_trait=args.samples_per_trait,
        run_name=args.run_name,
    )

    suite_config = _build_suite_config(
        base_model_path=base_model_path,
        out_dir=out_dir,
        constitution=constitution,
        samples_per_trait=args.samples_per_trait,
        output_root=output_root,
        run_name=run_name,
        temperature=args.temperature,
        batch_size=args.batch_size,
        skip_completed=not args.rerun,
    )

    result = run_eval_suite(suite_config)
    run_dir = result.output_root
    _assert_suite_succeeded(run_dir)
    summary_df = _extract_trait_table(run_dir)
    long_csv_path, wide_csv_path = _write_summaries(summary_df, run_dir)
    figure_path = _plot_trait_bars(
        summary_df,
        run_dir,
        title=(
            f"TRAIT benchmark comparison: {constitution} on {model_name} "
            f"(K={args.samples_per_trait} per trait)"
        ),
    )

    print(f"Eval run directory: {run_dir}")
    print(f"Long-form summary:  {long_csv_path}")
    print(f"Wide summary:       {wide_csv_path}")
    print(f"Bar chart:          {figure_path}")


if __name__ == "__main__":
    main()
