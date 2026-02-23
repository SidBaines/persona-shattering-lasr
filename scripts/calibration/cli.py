"""CLI for calibration workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.calibration.config import (
    CalibrationConfig,
    CalibrationDatasetConfig,
    CalibrationJudgeConfig,
    ReliabilityConfig,
    ValidityConfig,
)
from scripts.calibration.run import run_calibration
from scripts.calibration.traits import get_trait_preset, list_trait_presets
from scripts.common.config import DatasetConfig
from scripts.persona_metrics.config import JudgeLLMConfig


def _parse_metric_params(items: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --metric-param {item!r}; expected KEY=VALUE")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --metric-param key in {item!r}")
        raw = raw.strip()
        try:
            params[key] = json.loads(raw)
        except json.JSONDecodeError:
            params[key] = raw
    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run persona-metric calibration.")

    parser.add_argument(
        "--trait-preset",
        choices=list_trait_presets(),
        default="neuroticism",
        help="Trait preset metadata (default: neuroticism)",
    )
    parser.add_argument(
        "--metric-name",
        default=None,
        help="Persona metric registry name. Defaults to trait preset metric.",
    )

    parser.add_argument(
        "--dataset-source",
        choices=["huggingface", "local", "canonical"],
        default="local",
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--response-column", default="response")
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--subject-id-column", default=None)
    parser.add_argument("--unit-id-column", default=None)

    parser.add_argument(
        "--judge-provider",
        choices=["openai", "openrouter", "anthropic"],
        default="openai",
    )
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--judge-api-key-env", default=None)
    parser.add_argument("--judge-max-tokens", type=int, default=1024)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-concurrent", type=int, default=10)
    parser.add_argument("--judge-timeout", type=int, default=60)
    parser.add_argument(
        "--metric-param",
        action="append",
        default=[],
        help="Repeatable metric constructor param: KEY=JSON_VALUE",
    )

    parser.add_argument("--num-runs", type=int, default=7)
    parser.add_argument("--alpha-level", choices=["ordinal"], default="ordinal")
    parser.add_argument("--reliability-bootstrap-samples", type=int, default=1000)
    parser.add_argument("--reliability-min-units", type=int, default=20)
    parser.add_argument("--reliability-seed", type=int, default=13)

    parser.add_argument(
        "--analysis-unit",
        choices=["auto", "text", "subject"],
        default="auto",
    )
    parser.add_argument("--validity-bootstrap-samples", type=int, default=1000)
    parser.add_argument("--validity-seed", type=int, default=17)

    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default="scratch/calibration")
    parser.add_argument("--output-dir", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    trait = get_trait_preset(args.trait_preset)
    metric_name = args.metric_name or trait.metric_name
    label_column = args.label_column or trait.label_column_aliases[0]

    if args.dataset_source == "local" and not args.dataset_path:
        raise ValueError("--dataset-path is required when --dataset-source=local")
    if args.dataset_source == "huggingface" and not args.dataset_name:
        raise ValueError("--dataset-name is required when --dataset-source=huggingface")
    if args.dataset_source == "canonical" and not args.dataset_path:
        raise ValueError("--dataset-path is required when --dataset-source=canonical")

    metric_params = _parse_metric_params(args.metric_param)

    config = CalibrationConfig(
        dataset=CalibrationDatasetConfig(
            dataset=DatasetConfig(
                source=args.dataset_source,
                name=args.dataset_name,
                path=args.dataset_path,
                split=args.dataset_split,
                max_samples=args.max_samples,
            ),
            response_column=args.response_column,
            label_column=label_column,
            question_column=args.question_column,
            subject_id_column=args.subject_id_column,
            unit_id_column=args.unit_id_column,
        ),
        judge=CalibrationJudgeConfig(
            metric_name=metric_name,
            judge=JudgeLLMConfig(
                provider=args.judge_provider,
                model=args.judge_model,
                api_key_env=args.judge_api_key_env,
                max_tokens=args.judge_max_tokens,
                temperature=args.judge_temperature,
                max_concurrent=args.judge_max_concurrent,
                timeout=args.judge_timeout,
            ),
            metric_params=metric_params,
        ),
        trait=trait.model_copy(update={"metric_name": metric_name}),
        reliability=ReliabilityConfig(
            num_runs=args.num_runs,
            alpha_level=args.alpha_level,
            bootstrap_samples=args.reliability_bootstrap_samples,
            min_units=args.reliability_min_units,
            random_seed=args.reliability_seed,
        ),
        validity=ValidityConfig(
            analysis_unit=args.analysis_unit,
            bootstrap_samples=args.validity_bootstrap_samples,
            random_seed=args.validity_seed,
        ),
        run_name=args.run_name,
        output_root=Path(args.output_root),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    dataset, result = run_calibration(config)
    print(f"Calibration complete for {result.trait.trait_name} ({result.metric_name}).")
    print(f"Analysis unit: {result.analysis_unit}")
    print(f"Units: {result.num_scored_units}")
    print(f"Output: {result.output_dir}")

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    _ = dataset


if __name__ == "__main__":
    main()
