#!/usr/bin/env python3
"""Smoke test running multiple evaluations together.

Demonstrates running count_o and coherence evaluations on the same dataset
in a single run_evaluation call, as would happen in a real experiment.

Requires an API key for the judge provider (default: OpenAI).

Usage:
    cd persona-shattering
    uv run python scripts/experiments/evaluations/combined_eval_smoke_test.py
    uv run python scripts/experiments/evaluations/combined_eval_smoke_test.py --output-path scratch/combined_eval.jsonl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from datasets import Dataset
from dotenv import load_dotenv

from scripts.evaluation import (
    EvaluationConfig,
    JudgeLLMConfig,
    run_evaluation,
)
from scripts.utils import setup_logging


# Simulated pipeline outputs — responses of varying quality and O-density
SIMULATED_DATASET = [
    {
        "question": "Explain how computers store data.",
        "response": (
            "Computers store data using binary code, which consists of ones and "
            "zeros. Hard drives use magnetic storage, while solid-state drives "
            "use flash memory chips. Data is organized into files and folders "
            "by the operating system, which provides a structured way to access "
            "stored information."
        ),
    },
    {
        "question": "What is gravity?",
        "response": (
            "Gravity pulls things down. Apples fall. The moon orbits. "
            "I like space. Newton had an apple. Einstein said something "
            "about spacetime. Black holes are cool."
        ),
    },
    {
        "question": "Describe the process of making bread.",
        "response": (
            "Bread is made by combining flour, water, yeast, and salt into a "
            "dough. The dough is kneaded to develop gluten, then left to rise "
            "as the yeast ferments and produces carbon dioxide gas. After "
            "rising, the dough is shaped, allowed to proof again, and baked "
            "in an oven until golden brown."
        ),
    },
    {
        "question": "Why is the sky blue?",
        "response": (
            "Sunlight scatters. Blue light has a shorter wavelength. "
            "Rayleigh scattering. The atmosphere does it. "
            "Sunset is red though. Pretty colors everywhere."
        ),
    },
    {
        "question": "How do vaccines work?",
        "response": (
            "Vaccines work by introducing a harmless component of a pathogen — "
            "such as a weakened virus, an inactivated virus, or a piece of its "
            "protein — into the body. This stimulates the immune system to "
            "produce antibodies and train immune cells to recognize and fight "
            "the actual pathogen if encountered in the future, providing "
            "immunity without causing the disease."
        ),
    },
]


def main():
    parser = argparse.ArgumentParser(description="Combined evaluation smoke test.")
    parser.add_argument(
        "--provider", type=str, default="openai",
        choices=["openai", "openrouter", "anthropic"],
        help="LLM provider for the judge.",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="Model for the judge.",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=5,
        help="Max concurrent judge API calls.",
    )
    parser.add_argument(
        "--output-path", type=str, default=None,
        help="Optional output JSONL path.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Log level.",
    )
    args = parser.parse_args()

    load_dotenv()
    setup_logging(args.log_level)

    output = Path(args.output_path) if args.output_path else None

    dataset = Dataset.from_list(SIMULATED_DATASET)

    config = EvaluationConfig(
        evaluations=["count_o", "coherence"],
        response_column="response",
        question_column="question",
        judge=JudgeLLMConfig(
            provider=args.provider,
            model=args.model,
            max_concurrent=args.max_concurrent,
        ),
        output_path=output,
    )

    print(f"\n{'='*60}")
    print("COMBINED EVALUATION SMOKE TEST")
    print(f"{'='*60}")
    print(f"Judge: {args.provider} / {args.model}")
    print(f"Evaluations: {config.evaluations}")
    print(f"Samples: {len(dataset)}")
    print()

    start = time.perf_counter()
    result_dataset, result = run_evaluation(config, dataset=dataset)
    elapsed = time.perf_counter() - start

    # Per-record results
    print(f"\n{'='*60}")
    print("PER-RECORD RESULTS")
    print(f"{'='*60}")
    for i, row in enumerate(result_dataset):
        metrics = row["evaluation_metrics"]
        print(f"\n  [{i}] Q: {row['question']}")
        print(f"      Response: {row['response'][:70]}...")
        print(f"      O count: {metrics['count_o.count']}  |  O density: {metrics['count_o.density']}")
        print(f"      Coherence: {metrics['coherence.score']}  |  {metrics['coherence.reasoning'][:80]}")

    # Aggregates
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    for key, value in sorted(result.aggregates.items()):
        print(f"  {key}: {value:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Samples: {result.num_samples}")
    print(f"Evaluations: {result.evaluations_run}")
    print(f"Time: {elapsed:.2f}s")
    if result.output_path:
        print(f"Output: {result.output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
