#!/usr/bin/env python3
"""Smoke test for the CountVerbs evaluation.

Tests the evaluation module end-to-end using an in-memory dataset.
No API keys or external services required.

Usage:
    cd persona-shattering
    uv run python scripts/experiments/evaluations/count_o_smoke_test.py
    uv run python scripts/experiments/evaluations/count_o_smoke_test.py \
        --output-path scratch/count_verbs_test.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from datasets import Dataset  # noqa: E402

from scripts.evaluation import (  # noqa: E402
    EvaluationConfig,
    get_evaluation,
    run_evaluation,
)
from scripts.utils import setup_logging  # noqa: E402


def test_single_item():
    """Test evaluating a single response directly."""
    print("=" * 60)
    print("TEST: Single item evaluation")
    print("=" * 60)

    count_verbs = get_evaluation("count_verbs")
    result = count_verbs.evaluate(
        "Hello world, lots of verbs running and jumping in this response."
    )
    print(f"  Name: {count_verbs.name}")
    print(f"  Result: {result}")
    print()


def test_batch():
    """Test evaluating a batch of responses directly."""
    print("=" * 60)
    print("TEST: Batch evaluation")
    print("=" * 60)

    count_verbs = get_evaluation("count_verbs")

    responses = [
        "The quick brown fox jumps over the lazy dog.",
        "I am running and eating while thinking about sleeping!",
        "A beautiful sunny day indeed.",
        "",  # edge case: empty string
        "She walked, talked, and danced all night long.",
    ]
    questions = [
        "Write a pangram.",
        "Write something with lots of verbs.",
        "Write something short.",
        "Write nothing.",
        "Mix verbs.",
    ]

    results = count_verbs.evaluate_batch(responses, questions)
    for resp, q, r in zip(responses, questions, results):
        preview = repr(resp[:40]) if resp else "''"
        print(f"  Q: {q}")
        print(f"    Response: {preview}")
        print(f"    Metrics: {r}")
    print()


def test_run_evaluation(output_path: Path | None = None):
    """Test run_evaluation with an in-memory dataset."""
    print("=" * 60)
    print("TEST: run_evaluation (full pipeline)")
    print("=" * 60)

    dataset = Dataset.from_list([
        {
            "question": "What is photosynthesis?",
            "response": (
                "Photosynthesis is the process by which green plants "
                "and some other organisms use sunlight to synthesize "
                "foods from carbon dioxide and water. It generally "
                "involves the green pigment chlorophyll and generates "
                "oxygen as a byproduct."
            ),
        },
        {
            "question": "Name three colors.",
            "response": (
                "Red, blue, and green are three primary colors "
                "of light."
            ),
        },
        {
            "question": "What is 2+2?",
            "response": "Four.",
        },
        {
            "question": "Tell me about dogs.",
            "response": (
                "Dogs are domesticated mammals, not natural wild "
                "animals. They were originally bred from wolves. "
                "They have been bred by humans for a long time, "
                "and were the first animals ever to be domesticated."
            ),
        },
        {
            "question": "Say hello.",
            "response": "Hi there! Nice to meet you!",
        },
    ])

    config = EvaluationConfig(
        evaluations=["count_verbs"],
        response_column="response",
        question_column="question",
        output_path=output_path,
    )

    result_dataset, result = run_evaluation(config, dataset=dataset)

    print(f"  Samples evaluated: {result.num_samples}")
    print(f"  Evaluations run: {result.evaluations_run}")
    print()

    print("  Per-record results:")
    for i, row in enumerate(result_dataset):
        metrics = row["evaluation_metrics"]
        resp = row["response"]
        preview = resp[:50] + "..." if len(resp) > 50 else resp
        print(
            f"    [{i}] count={metrics['count_verbs.count']}, "
            f"density={metrics['count_verbs.density']} | {preview}"
        )
    print()

    print("  Aggregates:")
    for key, value in sorted(result.aggregates.items()):
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")
    print()

    if result.output_path:
        print(f"  Saved to: {result.output_path}")


def main():
    """Run all CountVerbs evaluation smoke tests."""
    parser = argparse.ArgumentParser(
        description="CountVerbs evaluation smoke test.",
    )
    parser.add_argument(
        "--output-path", type=str, default=None,
        help="Optional output JSONL path.",
    )
    parser.add_argument(
        "--log-level", type=str, default="WARNING",
        help="Log level.",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    output = Path(args.output_path) if args.output_path else None

    test_single_item()
    test_batch()
    test_run_evaluation(output_path=output)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
