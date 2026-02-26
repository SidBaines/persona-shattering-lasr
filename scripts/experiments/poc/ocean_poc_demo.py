#!/usr/bin/env python3
"""OCEAN proof-of-concept pipeline: train, sweep, eval, and plot."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import Dataset
from dotenv import load_dotenv

from scripts.common.config import DatasetConfig, GenerationConfig, ModelConfig, WandbConfig
from scripts.editing import EditingConfig, QualityConfig, run_editing
from scripts.editing.prompts import TEMPLATES
from scripts.evals.config import (
    AdapterConfig,
    InspectBenchmarkSpec,
    JudgeExecutionConfig,
    ModelSpec,
    SuiteConfig,
    SuiteResult,
)
from scripts.evals.suite import run_eval_suite
from scripts.inference import InferenceConfig, run_inference
from scripts.training import (
    CheckpointConfig,
    LoraConfig,
    SftConfig,
    TrainingConfig,
    TrainingEvaluationConfig,
    run_training,
)
from scripts.utils import (
    login_from_env,
    read_jsonl,
    upload_file_to_dataset_repo,
    upload_folder_to_model_repo,
    write_jsonl,
)
from scripts.visualisations.plot_ocean_poc import plot_ocean_poc

TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]
OCEAN_DIMENSIONS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]
TRAIT_TO_DIMENSION = {
    "openness": "Openness",
    "conscientiousness": "Conscientiousness",
    "extraversion": "Extraversion",
    "agreeableness": "Agreeableness",
    "neuroticism": "Neuroticism",
}
PROMPT_BY_TRAIT = {
    "openness": "quick-test-openness",
    "conscientiousness": "quick-test-conscientiousness",
    "extraversion": "quick-test-extraversion",
    "agreeableness": "quick-test-agreeableness",
    "neuroticism": "quick-test-neuroticism",
}
DEFAULT_SCALES = [-1.0, -0.5, 0.0, 0.5, 1.0]
DEFAULT_COMBO_VECTOR = {
    "openness": -1.0,
    "conscientiousness": 0.0,
    "extraversion": 0.7,
    "agreeableness": 1.0,
    "neuroticism": 0.5,
}

DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_EDITING_PROVIDER = "openai"
DEFAULT_EDITING_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_DATASET_NAME = "vicgalle/alpaca-gpt4"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_HF_PREFIX = "poc-test-sid"
DEFAULT_HF_NAMESPACE = "persona-shattering-lasr"
DEFAULT_RUN_ROOT = Path("scratch/poc_ocean")
DEFAULT_EVAL_BENCHMARKS = ["personality_trait", "truthfulqa_mc1", "gsm8k"]
SUPPORTED_EVAL_BENCHMARKS = set(DEFAULT_EVAL_BENCHMARKS)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _parse_traits(raw: str) -> list[str]:
    traits = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not traits:
        raise ValueError("At least one trait must be provided.")
    invalid = sorted(set(traits) - set(TRAITS))
    if invalid:
        raise ValueError(f"Unknown traits: {invalid}. Valid: {TRAITS}")
    return traits


def _parse_scales(raw: str) -> list[float]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one scale must be provided.")
    return [float(value) for value in values]


def _parse_combo_vector(raw: str) -> dict[str, float]:
    parsed = dict(DEFAULT_COMBO_VECTOR)
    if not raw.strip():
        return parsed
    for entry in raw.split(","):
        item = entry.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid combo entry '{item}'. Expected trait=value."
            )
        trait, value_text = item.split("=", 1)
        trait_name = trait.strip().lower()
        if trait_name not in TRAITS:
            raise ValueError(f"Invalid combo trait '{trait_name}'. Valid: {TRAITS}")
        parsed[trait_name] = float(value_text.strip())
    return parsed


def _parse_eval_benchmarks(raw: str) -> list[str]:
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("At least one eval benchmark must be provided.")
    invalid = sorted(set(names) - SUPPORTED_EVAL_BENCHMARKS)
    if invalid:
        raise ValueError(
            f"Unknown eval benchmarks: {invalid}. Valid: {sorted(SUPPORTED_EVAL_BENCHMARKS)}"
        )
    return names


def _benchmark_generation_args(args: argparse.Namespace) -> dict[str, int]:
    generation_args: dict[str, int] = {}
    if getattr(args, "inspect_max_connections", None) is not None:
        generation_args["max_connections"] = args.inspect_max_connections
    if getattr(args, "inspect_max_tokens", None) is not None:
        generation_args["max_tokens"] = args.inspect_max_tokens
    return generation_args


def _resolve_run_dir(args: argparse.Namespace, *, create: bool) -> Path:
    if args.run_dir is not None:
        run_dir = args.run_dir
    else:
        run_id = args.run_id or _timestamp()
        run_dir = DEFAULT_RUN_ROOT / run_id
    if create and not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _manifest_path(run_dir: Path) -> Path:
    return run_dir / "run_manifest.json"


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = _manifest_path(run_dir)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    return {
        "run_id": run_dir.name,
        "created_at": datetime.now().isoformat(),
        "traits": {},
        "artifacts": {},
    }


def _save_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    path = _manifest_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated_at"] = datetime.now().isoformat()
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _validate_trait_mapping(traits: list[str]) -> None:
    missing_prompts = [
        PROMPT_BY_TRAIT[trait]
        for trait in traits
        if PROMPT_BY_TRAIT[trait] not in TEMPLATES
    ]
    if missing_prompts:
        raise ValueError(
            "Missing quick-test prompts in scripts.editing.prompts: "
            f"{sorted(missing_prompts)}"
        )


def _repo_id(namespace: str, prefix: str, trait: str, kind: str) -> str:
    return f"{namespace}/{prefix}-{trait}-{kind}"


def _scale_tag(scale: float) -> str:
    text = f"{scale:.4f}".rstrip("0").rstrip(".")
    if text.startswith("-"):
        text = "neg_" + text[1:]
    return text.replace(".", "p")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _flatten_numeric(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out[prefix] = float(value)
        return
    if isinstance(value, dict):
        for key, sub_value in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_numeric(next_prefix, sub_value, out)


def read_numeric_metrics_from_log(log_path: str) -> dict[str, float]:
    """Read and flatten numeric metrics from one Inspect log file."""
    from inspect_ai.log import read_eval_log

    eval_log = read_eval_log(log_path)
    results = getattr(eval_log, "results", None)
    if results is None:
        return {}
    scores = getattr(results, "scores", None) or []
    flat: dict[str, float] = {}
    for score in scores:
        metrics = getattr(score, "metrics", None)
        if not isinstance(metrics, dict):
            continue
        for metric_name, metric_obj in metrics.items():
            metric_value = (
                metric_obj.value if hasattr(metric_obj, "value") else metric_obj
            )
            _flatten_numeric(str(metric_name), metric_value, flat)
    return flat


def _extract_ocean_dimensions(metrics: dict[str, float]) -> dict[str, float]:
    extracted: dict[str, float] = {}
    for dim in OCEAN_DIMENSIONS:
        candidates = [
            dim,
            f"trait_ratio.{dim}",
            f"{dim}.mean",
            f"trait_ratio.{dim}.mean",
        ]
        value = next((metrics[c] for c in candidates if c in metrics), None)
        if value is None:
            for key, val in metrics.items():
                if key.endswith(f".{dim}") or key.endswith(f".{dim}.mean"):
                    value = val
                    break
        if value is not None:
            extracted[dim] = value
    return extracted


def _select_primary_metric(eval_name: str, metrics: dict[str, float]) -> tuple[str, float] | None:
    if not metrics:
        return None
    preferred_substrings = ["accuracy", "exact", "score", "mean"]
    ordered_keys = sorted(metrics.keys())
    for needle in preferred_substrings:
        for key in ordered_keys:
            if needle in key.lower():
                return key, metrics[key]
    first_key = ordered_keys[0]
    return first_key, metrics[first_key]


def _run_suite(config: SuiteConfig, judge_exec: JudgeExecutionConfig) -> SuiteResult:
    return run_eval_suite(config, judge_exec=judge_exec)


def _write_outputs_readme(run_dir: Path, manifest: dict[str, Any]) -> None:
    artifacts = manifest.get("artifacts", {})
    lines = [
        "# OCEAN POC Outputs",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Core artifacts",
        "",
        "| Artifact | Path |",
        "|---|---|",
    ]
    for key in sorted(artifacts.keys()):
        lines.append(f"| `{key}` | `{artifacts[key]}` |")
    lines.extend(
        [
            "",
            "## Table columns",
            "",
            "- `scaling_summary.csv`: `trait`, `target_dimension`, `scale`, `target_score`, `status`, `model_spec_name`, `inspect_log_path`",
            "- `scaling_long.csv`: `trait`, `target_dimension`, `scale`, `target_score`, `status`, `model_spec_name`, OCEAN columns",
            "- `model_eval_long.csv`: `model`, `eval_name`, `metric_key`, `metric_value`, `status`, `inspect_log_path`",
            "- `model_eval_wide.csv`: `model`, capability columns (`truthfulqa_mc1`, `gsm8k`), OCEAN columns",
            "",
        ]
    )
    (run_dir / "README_poc_outputs.md").write_text("\n".join(lines), encoding="utf-8")


def _train_phase(
    args: argparse.Namespace,
    run_dir: Path,
    manifest: dict[str, Any],
    traits: list[str],
) -> None:
    base_inference_path = run_dir / "intermediate" / "base_inference.jsonl"
    datasets_dir = run_dir / "datasets"
    checkpoints_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("DRY RUN: training phase plan")
        print(f"  run_dir={run_dir}")
        print(f"  base_model={args.base_model}")
        print(f"  train_samples={args.train_samples}, train_epochs={args.train_epochs}")
        print(f"  traits={traits}")
        for trait in traits:
            print(
                f"  - {trait}: prompt={PROMPT_BY_TRAIT[trait]}, "
                f"dataset_repo={_repo_id(args.hf_namespace, args.hf_prefix, trait, 'dataset')}, "
                f"model_repo={_repo_id(args.hf_namespace, args.hf_prefix, trait, 'model')}"
            )
        return

    if not args.skip_hf_upload:
        login_from_env()

    if base_inference_path.exists() and not args.overwrite:
        base_records = read_jsonl(base_inference_path)
        base_dataset = Dataset.from_list(base_records)
    else:
        inference_config = InferenceConfig(
            model=args.base_model,
            provider="local",
            dataset=DatasetConfig(
                source="huggingface",
                name=args.dataset_name,
                split=args.dataset_split,
                max_samples=args.train_samples,
            ),
            generation=GenerationConfig(
                max_new_tokens=args.inference_max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                batch_size=args.inference_batch_size,
                num_responses_per_prompt=1,
            ),
            output_path=base_inference_path,
            overwrite_output=args.overwrite,
            resume=not args.overwrite,
        )
        base_dataset, _ = run_inference(inference_config)

    for trait in traits:
        prompt_template = PROMPT_BY_TRAIT[trait]
        edited_path = run_dir / "intermediate" / f"edited_{trait}.jsonl"
        train_dataset_path = datasets_dir / f"{trait}.jsonl"
        final_adapter_path = checkpoints_dir / trait / "final"

        if train_dataset_path.exists() and not args.overwrite:
            train_rows = read_jsonl(train_dataset_path)
        else:
            editing_config = EditingConfig(
                provider=args.editing_provider,
                model=args.editing_model,
                prompt_template=prompt_template,
                max_concurrent=args.editing_max_concurrent,
                quality=QualityConfig(enabled=False),
                output_path=edited_path,
                overwrite_output=args.overwrite,
                resume=not args.overwrite,
            )
            edited_dataset, _ = run_editing(editing_config, dataset=base_dataset)
            train_rows = [
                {
                    "question": str(row.get("question", "")),
                    "edited_response": str(row.get("edited_response", "")),
                }
                for row in edited_dataset.to_list()
            ]
            write_jsonl(train_rows, train_dataset_path)

        dataset_repo_id = _repo_id(args.hf_namespace, args.hf_prefix, trait, "dataset")
        dataset_url = None
        if not args.skip_hf_upload:
            dataset_url = upload_file_to_dataset_repo(
                local_path=train_dataset_path,
                repo_id=dataset_repo_id,
                path_in_repo="train.jsonl",
                commit_message=f"Add OCEAN POC dataset for {trait}",
            )

        if final_adapter_path.exists() and not args.overwrite:
            trained_path = final_adapter_path
        else:
            training_config = TrainingConfig(
                dataset_path=train_dataset_path,
                user_column="question",
                assistant_column="edited_response",
                model=ModelConfig(
                    name=args.base_model,
                    dtype="bfloat16",
                    device_map="auto",
                ),
                lora=LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                ),
                sft=SftConfig(
                    num_train_epochs=args.train_epochs,
                    per_device_train_batch_size=args.train_batch_size,
                    gradient_accumulation_steps=args.train_gradient_accumulation,
                    learning_rate=args.train_learning_rate,
                    bf16=True,
                ),
                checkpoint=CheckpointConfig(
                    save_strategy="epoch",
                    save_total_limit=2,
                ),
                wandb=WandbConfig(
                    enabled=not args.no_wandb,
                    project=args.wandb_project,
                    name=f"{run_dir.name}-{trait}",
                    tags=["ocean-poc", trait],
                ),
                evaluation=TrainingEvaluationConfig(
                    enabled=False,
                    evaluations=[],
                ),
                checkpoint_dir=checkpoints_dir / trait,
                val_split=0.1,
                seed=42,
            )
            _, training_result = run_training(training_config)
            trained_path = Path(training_result.checkpoint_path)

        model_repo_id = _repo_id(args.hf_namespace, args.hf_prefix, trait, "model")
        model_url = None
        if not args.skip_hf_upload:
            model_url = upload_folder_to_model_repo(
                local_dir=trained_path,
                repo_id=model_repo_id,
                path_in_repo="adapter",
                commit_message=f"Add OCEAN POC adapter for {trait}",
            )

        manifest.setdefault("traits", {})[trait] = {
            "prompt_template": prompt_template,
            "dataset_local_path": str(train_dataset_path),
            "dataset_repo_id": dataset_repo_id,
            "dataset_url": dataset_url,
            "adapter_local_path": str(trained_path),
            "model_repo_id": model_repo_id,
            "model_url": model_url,
            "num_training_rows": len(train_rows),
        }

    manifest.setdefault("artifacts", {})["base_inference_path"] = str(base_inference_path)


def _sweep_phase(
    args: argparse.Namespace,
    run_dir: Path,
    manifest: dict[str, Any],
    traits: list[str],
    scales: list[float],
) -> None:
    generation_args = _benchmark_generation_args(args)
    if args.dry_run:
        print("DRY RUN: sweep phase plan")
        print(f"  traits={traits}, scales={scales}, eval_samples={args.eval_samples}")
        print(f"  generation_args={generation_args}")
        return

    summary_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []

    for trait in traits:
        trait_info = manifest.get("traits", {}).get(trait, {})
        adapter_path = trait_info.get("adapter_local_path")
        if not adapter_path:
            raise ValueError(f"Missing adapter_local_path in manifest for trait '{trait}'")
        models: list[ModelSpec] = []
        scale_by_model: dict[str, float] = {}
        for scale in scales:
            model_name = f"{trait}_s_{_scale_tag(scale)}"
            scale_by_model[model_name] = scale
            models.append(
                ModelSpec(
                    name=model_name,
                    base_model=args.base_model,
                    adapters=[
                        AdapterConfig(path=f"local://{adapter_path}", scale=scale),
                    ],
                )
            )

        suite_config = SuiteConfig(
            models=models,
            evals=[
                InspectBenchmarkSpec(
                    name="personality_trait",
                    benchmark="personality_trait",
                    generation_args=generation_args,
                    limit=args.eval_samples,
                )
            ],
            output_root=run_dir / "evals" / "sweeps",
            run_name=f"{trait}_sweep",
            cleanup_materialized_models=True,
            hf_log_dir="",
        )
        suite_result = _run_suite(
            suite_config,
            JudgeExecutionConfig(mode="blocking", prefer_batch=True),
        )

        for row in suite_result.rows:
            metrics = (
                read_numeric_metrics_from_log(row.inspect_log_path)
                if row.status == "ok" and row.inspect_log_path
                else {}
            )
            dims = _extract_ocean_dimensions(metrics)
            target_dim = TRAIT_TO_DIMENSION[trait]
            scale = scale_by_model.get(row.model_spec_name, float("nan"))
            target_score = dims.get(target_dim, float("nan"))
            summary_rows.append(
                {
                    "trait": trait,
                    "target_dimension": target_dim,
                    "scale": scale,
                    "target_score": target_score,
                    "status": row.status,
                    "model_spec_name": row.model_spec_name,
                    "inspect_log_path": row.inspect_log_path or "",
                }
            )
            long_rows.append(
                {
                    "trait": trait,
                    "target_dimension": target_dim,
                    "scale": scale,
                    "target_score": target_score,
                    "status": row.status,
                    "model_spec_name": row.model_spec_name,
                    "inspect_log_path": row.inspect_log_path or "",
                    **{dim: dims.get(dim, float("nan")) for dim in OCEAN_DIMENSIONS},
                }
            )

    summary_csv = run_dir / "scaling_summary.csv"
    long_csv = run_dir / "scaling_long.csv"
    _write_csv(
        summary_csv,
        summary_rows,
        [
            "trait",
            "target_dimension",
            "scale",
            "target_score",
            "status",
            "model_spec_name",
            "inspect_log_path",
        ],
    )
    _write_csv(
        long_csv,
        long_rows,
        [
            "trait",
            "target_dimension",
            "scale",
            "target_score",
            "status",
            "model_spec_name",
            "inspect_log_path",
            *OCEAN_DIMENSIONS,
        ],
    )
    manifest.setdefault("artifacts", {})["scaling_summary_csv"] = str(summary_csv)
    manifest.setdefault("artifacts", {})["scaling_long_csv"] = str(long_csv)


def _eval_phase(
    args: argparse.Namespace,
    run_dir: Path,
    manifest: dict[str, Any],
    traits: list[str],
    combo_vector: dict[str, float],
    eval_benchmarks: list[str],
) -> None:
    generation_args = _benchmark_generation_args(args)
    if args.dry_run:
        print("DRY RUN: eval phase plan")
        print(f"  eval_samples={args.eval_samples}, traits={traits}")
        print(f"  eval_benchmarks={eval_benchmarks}")
        print(f"  generation_args={generation_args}")
        print(f"  combo_vector={combo_vector}")
        return

    trait_infos = manifest.get("traits", {})
    for trait in traits:
        if not trait_infos.get(trait, {}).get("adapter_local_path"):
            raise ValueError(f"Missing adapter_local_path for trait '{trait}' in run_manifest")

    models: list[ModelSpec] = [ModelSpec(name="base", base_model=args.base_model)]
    for trait in traits:
        models.append(
            ModelSpec(
                name=trait,
                base_model=args.base_model,
                adapters=[
                    AdapterConfig(
                        path=f"local://{trait_infos[trait]['adapter_local_path']}",
                        scale=1.0,
                    )
                ],
            )
        )
    combo_adapters = [
        AdapterConfig(
            path=f"local://{trait_infos[trait]['adapter_local_path']}",
            scale=combo_vector[trait],
        )
        for trait in traits
    ]
    models.append(
        ModelSpec(
            name="combo",
            base_model=args.base_model,
            adapters=combo_adapters,
        )
    )

    eval_specs_by_name = {
        "personality_trait": InspectBenchmarkSpec(
            name="personality_trait",
            benchmark="personality_trait",
            generation_args=generation_args,
            limit=args.eval_samples,
        ),
        "truthfulqa_mc1": InspectBenchmarkSpec(
            name="truthfulqa_mc1",
            benchmark="truthfulqa",
            benchmark_args={"target": "mc1"},
            generation_args=generation_args,
            limit=args.eval_samples,
        ),
        "gsm8k": InspectBenchmarkSpec(
            name="gsm8k",
            benchmark="gsm8k",
            benchmark_args={"fewshot": args.gsm8k_fewshot},
            generation_args=generation_args,
            limit=args.eval_samples,
        ),
    }
    eval_specs = [eval_specs_by_name[name] for name in eval_benchmarks]

    suite_config = SuiteConfig(
        models=models,
        evals=eval_specs,
        output_root=run_dir / "evals" / "models",
        run_name="model_eval",
        cleanup_materialized_models=True,
        hf_log_dir="",
    )
    suite_result = _run_suite(
        suite_config,
        JudgeExecutionConfig(mode="blocking", prefer_batch=True),
    )

    existing_long_csv = run_dir / "model_eval_long.csv"
    existing_rows_raw = _read_csv_rows(existing_long_csv)
    long_rows: list[dict[str, Any]] = [
        row for row in existing_rows_raw if row.get("eval_name") not in set(eval_benchmarks)
    ]

    for row in suite_result.rows:
        metrics = (
            read_numeric_metrics_from_log(row.inspect_log_path)
            if row.status == "ok" and row.inspect_log_path
            else {}
        )
        for metric_key, metric_value in sorted(metrics.items()):
            long_rows.append(
                {
                    "model": row.model_spec_name,
                    "eval_name": row.eval_name,
                    "metric_key": metric_key,
                    "metric_value": metric_value,
                    "status": row.status,
                    "inspect_log_path": row.inspect_log_path or "",
                }
            )

    metrics_by_model_eval: dict[tuple[str, str], dict[str, float]] = {}
    for row in long_rows:
        if row.get("status") != "ok":
            continue
        model_name = row.get("model")
        eval_name = row.get("eval_name")
        metric_key = row.get("metric_key")
        metric_value_raw = row.get("metric_value")
        if not model_name or not eval_name or not metric_key or metric_value_raw in {None, ""}:
            continue
        try:
            metric_value = float(metric_value_raw)
        except ValueError:
            continue
        metrics_by_model_eval.setdefault((model_name, eval_name), {})[metric_key] = metric_value

    capability_by_model: dict[str, dict[str, float]] = {}
    traits_by_model: dict[str, dict[str, float]] = {}
    for (model_name, eval_name), metrics in metrics_by_model_eval.items():
        if eval_name == "personality_trait":
            traits_by_model.setdefault(model_name, {}).update(
                _extract_ocean_dimensions(metrics)
            )
        elif eval_name in {"truthfulqa_mc1", "gsm8k"}:
            selected = _select_primary_metric(eval_name, metrics)
            if selected is not None:
                _, selected_value = selected
                capability_by_model.setdefault(model_name, {})[eval_name] = selected_value

    model_order = ["base", *traits, "combo"]
    wide_rows: list[dict[str, Any]] = []
    for model_name in model_order:
        wide_rows.append(
            {
                "model": model_name,
                "truthfulqa_mc1": capability_by_model.get(model_name, {}).get(
                    "truthfulqa_mc1",
                    float("nan"),
                ),
                "gsm8k": capability_by_model.get(model_name, {}).get(
                    "gsm8k",
                    float("nan"),
                ),
                **{
                    dim: traits_by_model.get(model_name, {}).get(dim, float("nan"))
                    for dim in OCEAN_DIMENSIONS
                },
            }
        )

    long_csv = run_dir / "model_eval_long.csv"
    wide_csv = run_dir / "model_eval_wide.csv"
    _write_csv(
        long_csv,
        long_rows,
        ["model", "eval_name", "metric_key", "metric_value", "status", "inspect_log_path"],
    )
    _write_csv(
        wide_csv,
        wide_rows,
        ["model", "truthfulqa_mc1", "gsm8k", *OCEAN_DIMENSIONS],
    )
    manifest.setdefault("artifacts", {})["model_eval_long_csv"] = str(long_csv)
    manifest.setdefault("artifacts", {})["model_eval_wide_csv"] = str(wide_csv)


def _plot_phase(args: argparse.Namespace, run_dir: Path, manifest: dict[str, Any]) -> None:
    scaling_csv = run_dir / "scaling_long.csv"
    eval_wide_csv = run_dir / "model_eval_wide.csv"
    output_dir = run_dir / "plots"

    if args.dry_run:
        print("DRY RUN: plot phase plan")
        print(f"  scaling_csv={scaling_csv}")
        print(f"  eval_wide_csv={eval_wide_csv}")
        print(f"  output_dir={output_dir}")
        return

    fig1, fig2 = plot_ocean_poc(
        scaling_csv=scaling_csv,
        eval_wide_csv=eval_wide_csv,
        output_dir=output_dir,
    )
    manifest.setdefault("artifacts", {})["figure1"] = str(fig1)
    manifest.setdefault("artifacts", {})["figure2"] = str(fig2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OCEAN POC pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_flags(target: argparse.ArgumentParser) -> None:
        target.add_argument("--run-dir", type=Path, default=None)
        target.add_argument("--run-id", type=str, default=None)
        target.add_argument("--traits", type=str, default=",".join(TRAITS))
        target.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
        target.add_argument("--dry-run", action="store_true")

    train_parser = subparsers.add_parser("train", help="Train one LoRA per OCEAN trait.")
    add_common_flags(train_parser)
    train_parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    train_parser.add_argument("--dataset-split", type=str, default=DEFAULT_DATASET_SPLIT)
    train_parser.add_argument("--train-samples", type=int, default=100)
    train_parser.add_argument("--train-epochs", type=int, default=1)
    train_parser.add_argument("--inference-max-new-tokens", type=int, default=256)
    train_parser.add_argument("--inference-batch-size", type=int, default=64)
    train_parser.add_argument("--editing-provider", type=str, default=DEFAULT_EDITING_PROVIDER)
    train_parser.add_argument("--editing-model", type=str, default=DEFAULT_EDITING_MODEL)
    train_parser.add_argument("--editing-max-concurrent", type=int, default=8)
    train_parser.add_argument("--lora-r", type=int, default=16)
    train_parser.add_argument("--lora-alpha", type=int, default=16)
    train_parser.add_argument("--lora-dropout", type=float, default=0.0)
    train_parser.add_argument("--train-batch-size", type=int, default=4)
    train_parser.add_argument("--train-gradient-accumulation", type=int, default=4)
    train_parser.add_argument("--train-learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--hf-namespace", type=str, default=DEFAULT_HF_NAMESPACE)
    train_parser.add_argument("--hf-prefix", type=str, default=DEFAULT_HF_PREFIX)
    train_parser.add_argument("--skip-hf-upload", action="store_true")
    train_parser.add_argument("--no-wandb", action="store_true")
    train_parser.add_argument("--wandb-project", type=str, default="persona-shattering-v1")
    train_parser.add_argument("--overwrite", action="store_true")

    sweep_parser = subparsers.add_parser("sweep", help="Run scaling sweeps on trait adapters.")
    add_common_flags(sweep_parser)
    sweep_parser.add_argument("--eval-samples", type=int, default=50)
    sweep_parser.add_argument(
        "--scales",
        type=str,
        default=",".join(str(scale) for scale in DEFAULT_SCALES),
    )
    sweep_parser.add_argument("--inspect-max-connections", type=int, default=4)
    sweep_parser.add_argument("--inspect-max-tokens", type=int, default=512)

    eval_parser = subparsers.add_parser("eval", help="Run base/trait/combo eval suite.")
    add_common_flags(eval_parser)
    eval_parser.add_argument("--eval-samples", type=int, default=50)
    eval_parser.add_argument(
        "--eval-benchmarks",
        type=str,
        default=",".join(DEFAULT_EVAL_BENCHMARKS),
    )
    eval_parser.add_argument(
        "--combo-vector",
        type=str,
        default=",".join(f"{k}={v}" for k, v in DEFAULT_COMBO_VECTOR.items()),
    )
    eval_parser.add_argument("--gsm8k-fewshot", type=int, default=3)
    eval_parser.add_argument("--inspect-max-connections", type=int, default=4)
    eval_parser.add_argument("--inspect-max-tokens", type=int, default=512)

    plot_parser = subparsers.add_parser("plot", help="Render paper figures from CSV outputs.")
    add_common_flags(plot_parser)

    all_parser = subparsers.add_parser("all", help="Run train, sweep, eval, and plot.")
    add_common_flags(all_parser)
    all_parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    all_parser.add_argument("--dataset-split", type=str, default=DEFAULT_DATASET_SPLIT)
    all_parser.add_argument("--train-samples", type=int, default=100)
    all_parser.add_argument("--train-epochs", type=int, default=1)
    all_parser.add_argument("--inference-max-new-tokens", type=int, default=256)
    all_parser.add_argument("--inference-batch-size", type=int, default=64)
    all_parser.add_argument("--editing-provider", type=str, default=DEFAULT_EDITING_PROVIDER)
    all_parser.add_argument("--editing-model", type=str, default=DEFAULT_EDITING_MODEL)
    all_parser.add_argument("--editing-max-concurrent", type=int, default=8)
    all_parser.add_argument("--lora-r", type=int, default=16)
    all_parser.add_argument("--lora-alpha", type=int, default=16)
    all_parser.add_argument("--lora-dropout", type=float, default=0.0)
    all_parser.add_argument("--train-batch-size", type=int, default=4)
    all_parser.add_argument("--train-gradient-accumulation", type=int, default=4)
    all_parser.add_argument("--train-learning-rate", type=float, default=1e-4)
    all_parser.add_argument("--eval-samples", type=int, default=50)
    all_parser.add_argument(
        "--eval-benchmarks",
        type=str,
        default=",".join(DEFAULT_EVAL_BENCHMARKS),
    )
    all_parser.add_argument(
        "--scales",
        type=str,
        default=",".join(str(scale) for scale in DEFAULT_SCALES),
    )
    all_parser.add_argument(
        "--combo-vector",
        type=str,
        default=",".join(f"{k}={v}" for k, v in DEFAULT_COMBO_VECTOR.items()),
    )
    all_parser.add_argument("--gsm8k-fewshot", type=int, default=3)
    all_parser.add_argument("--inspect-max-connections", type=int, default=4)
    all_parser.add_argument("--inspect-max-tokens", type=int, default=512)
    all_parser.add_argument("--hf-namespace", type=str, default=DEFAULT_HF_NAMESPACE)
    all_parser.add_argument("--hf-prefix", type=str, default=DEFAULT_HF_PREFIX)
    all_parser.add_argument("--skip-hf-upload", action="store_true")
    all_parser.add_argument("--no-wandb", action="store_true")
    all_parser.add_argument("--wandb-project", type=str, default="persona-shattering-v1")
    all_parser.add_argument("--overwrite", action="store_true")

    return parser


def _phase_defaults_for_non_training(args: argparse.Namespace) -> None:
    """Populate train/eval defaults for phase-specific subcommands."""
    # Used when a parser doesn't include all fields required by helper funcs.
    if not hasattr(args, "hf_namespace"):
        args.hf_namespace = DEFAULT_HF_NAMESPACE
    if not hasattr(args, "hf_prefix"):
        args.hf_prefix = DEFAULT_HF_PREFIX
    if not hasattr(args, "skip_hf_upload"):
        args.skip_hf_upload = True
    if not hasattr(args, "no_wandb"):
        args.no_wandb = True
    if not hasattr(args, "wandb_project"):
        args.wandb_project = "persona-shattering-v1"
    if not hasattr(args, "overwrite"):
        args.overwrite = False


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _phase_defaults_for_non_training(args)
    traits = _parse_traits(args.traits)
    _validate_trait_mapping(traits)
    scales = _parse_scales(args.scales) if hasattr(args, "scales") else list(DEFAULT_SCALES)
    eval_benchmarks = (
        _parse_eval_benchmarks(args.eval_benchmarks)
        if hasattr(args, "eval_benchmarks")
        else list(DEFAULT_EVAL_BENCHMARKS)
    )
    combo_vector = (
        _parse_combo_vector(args.combo_vector)
        if hasattr(args, "combo_vector")
        else dict(DEFAULT_COMBO_VECTOR)
    )

    create_run_dir = args.command in {"train", "all"}
    run_dir = _resolve_run_dir(args, create=create_run_dir)

    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        load_dotenv()

    manifest = _load_manifest(run_dir)
    manifest["config"] = {
        "base_model": args.base_model,
        "traits": traits,
        "scales": scales,
        "eval_benchmarks": eval_benchmarks,
        "combo_vector": combo_vector,
        "train_samples": getattr(args, "train_samples", None),
        "train_epochs": getattr(args, "train_epochs", None),
        "eval_samples": getattr(args, "eval_samples", None),
        "gsm8k_fewshot": getattr(args, "gsm8k_fewshot", None),
        "inspect_max_connections": getattr(args, "inspect_max_connections", None),
        "inspect_max_tokens": getattr(args, "inspect_max_tokens", None),
        "hf_namespace": getattr(args, "hf_namespace", None),
        "hf_prefix": getattr(args, "hf_prefix", None),
        "skip_hf_upload": getattr(args, "skip_hf_upload", None),
        "no_wandb": getattr(args, "no_wandb", None),
    }

    if args.command == "train":
        _train_phase(args, run_dir, manifest, traits)
    elif args.command == "sweep":
        _sweep_phase(args, run_dir, manifest, traits, scales)
    elif args.command == "eval":
        _eval_phase(args, run_dir, manifest, traits, combo_vector, eval_benchmarks)
    elif args.command == "plot":
        _plot_phase(args, run_dir, manifest)
    elif args.command == "all":
        _train_phase(args, run_dir, manifest, traits)
        _sweep_phase(args, run_dir, manifest, traits, scales)
        _eval_phase(args, run_dir, manifest, traits, combo_vector, eval_benchmarks)
        _plot_phase(args, run_dir, manifest)
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    if not args.dry_run:
        _save_manifest(run_dir, manifest)
        _write_outputs_readme(run_dir, manifest)

    print(f"Run directory: {run_dir}")
    if not args.dry_run:
        print(f"Manifest: {_manifest_path(run_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
