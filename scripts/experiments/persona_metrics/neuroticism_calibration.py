#!/usr/bin/env python3
"""Example experiment: calibrate neuroticism judge against questionnaire labels.

Usage (local JSONL):
    uv run python scripts/experiments/persona_metrics/neuroticism_calibration.py \
      --dataset-path scratch/calibration_input.jsonl \
      --label-column neuroticism

Usage (HuggingFace):
    uv run python scripts/experiments/persona_metrics/neuroticism_calibration.py \
      --dataset-source huggingface \
      --dataset-name <dataset_name> \
      --dataset-split train \
      --label-column neuroticism
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from scripts.calibration import (
    CalibrationConfig,
    CalibrationDatasetConfig,
    CalibrationJudgeConfig,
    ReliabilityConfig,
    ValidityConfig,
    get_trait_preset,
    run_calibration,
)
from scripts.common.config import DatasetConfig
from scripts.persona_metrics.config import JudgeLLMConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run neuroticism calibration experiment.")
    parser.add_argument(
        "--dataset-source",
        choices=["local", "huggingface", "canonical"],
        default="local",
    )
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--response-column", default="response")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--label-column", default="neuroticism")
    parser.add_argument("--subject-id-column", default=None)
    parser.add_argument("--unit-id-column", default=None)

    parser.add_argument(
        "--judge-provider",
        choices=["openai", "openrouter", "anthropic"],
        default="openai",
    )
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-concurrent", type=int, default=10)

    parser.add_argument("--num-runs", type=int, default=7)
    parser.add_argument(
        "--analysis-unit",
        choices=["auto", "text", "subject"],
        default="auto",
    )
    parser.add_argument("--output-root", default="scratch/calibration")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    if args.dataset_source == "local" and not args.dataset_path:
        raise ValueError("--dataset-path is required for local datasets")
    if args.dataset_source == "huggingface" and not args.dataset_name:
        raise ValueError("--dataset-name is required for HuggingFace datasets")
    if args.dataset_source == "canonical" and not args.dataset_path:
        raise ValueError("--dataset-path is required for canonical datasets")

    trait = get_trait_preset("neuroticism")

    config = CalibrationConfig(
        dataset=CalibrationDatasetConfig(
            dataset=DatasetConfig(
                source=args.dataset_source,
                path=args.dataset_path,
                name=args.dataset_name,
                split=args.dataset_split,
                max_samples=args.max_samples,
            ),
            response_column=args.response_column,
            question_column=args.question_column,
            label_column=args.label_column,
            subject_id_column=args.subject_id_column,
            unit_id_column=args.unit_id_column,
        ),
        judge=CalibrationJudgeConfig(
            metric_name="neuroticism",
            judge=JudgeLLMConfig(
                provider=args.judge_provider,
                model=args.judge_model,
                temperature=args.judge_temperature,
                max_concurrent=args.judge_max_concurrent,
            ),
        ),
        trait=trait,
        reliability=ReliabilityConfig(num_runs=args.num_runs),
        validity=ValidityConfig(analysis_unit=args.analysis_unit),
        output_root=Path(args.output_root),
        run_name=args.run_name,
    )

    dataset, result = run_calibration(config)

    print(f"Calibration complete: {result.output_dir}")
    print(f"Trait: {result.trait.trait_name}")
    print(f"Analysis unit: {result.analysis_unit}")
    print(f"Units: {result.num_scored_units}")
    print(f"Reliability alpha: {result.reliability.get('alpha')}")
    print(f"Pearson r: {result.validity.get('point', {}).get('pearson_r')}")
    print(f"Spearman rho: {result.validity.get('point', {}).get('spearman_rho')}")

    _ = dataset


if __name__ == "__main__":
    main()
