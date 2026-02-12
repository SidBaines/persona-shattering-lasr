"""CLI entry point for the evaluation stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common.config import DatasetConfig
from scripts.common.persona_metrics import (
    DEFAULT_PERSONA,
    PERSONA_EVALUATIONS,
    get_persona_evaluation,
)
from scripts.evaluation.config import EvaluationConfig, JudgeLLMConfig
from scripts.evaluation.run import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluations on a dataset.",
    )
    parser.add_argument(
        "--evaluations",
        type=str,
        nargs="+",
        default=None,
        help="Evaluation names to run (default: resolved from --persona)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to input JSONL dataset",
    )
    parser.add_argument(
        "--response-column",
        type=str,
        default="response",
        help="Column name containing responses (default: response)",
    )
    parser.add_argument(
        "--question-column",
        type=str,
        default="question",
        help="Column name containing questions (default: question)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save output JSONL with evaluation results",
    )
    # Judge LLM settings
    parser.add_argument(
        "--judge-provider",
        type=str,
        choices=["openai", "openrouter", "anthropic"],
        default="openai",
        help="LLM provider for judge evaluations (default: openai)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Model for judge evaluations (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--judge-max-concurrent",
        type=int,
        default=10,
        help="Max concurrent judge API calls (default: 10)",
    )
    parser.add_argument(
        "--persona",
        type=str,
        default=DEFAULT_PERSONA,
        choices=sorted(PERSONA_EVALUATIONS.keys()),
        help=f"Persona — resolves to the corresponding evaluation name (default: {DEFAULT_PERSONA})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Respect explicit --evaluations exactly as provided.
    # If omitted, resolve default evaluation from persona.
    evaluations = (
        list(args.evaluations)
        if args.evaluations is not None
        else [get_persona_evaluation(args.persona)]
    )

    config = EvaluationConfig(
        evaluations=evaluations,
        dataset=DatasetConfig(
            source="local",
            path=args.dataset_path,
        ),
        response_column=args.response_column,
        question_column=args.question_column,
        judge=JudgeLLMConfig(
            provider=args.judge_provider,
            model=args.judge_model,
            max_concurrent=args.judge_max_concurrent,
        ),
        output_path=Path(args.output_path) if args.output_path else None,
    )

    dataset, result = run_evaluation(config)
    print(f"\nEvaluated {result.num_samples} samples with: {result.evaluations_run}")
    if result.aggregates:
        print("Summary:")
        for key, value in sorted(result.aggregates.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
