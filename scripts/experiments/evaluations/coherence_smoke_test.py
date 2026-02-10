#!/usr/bin/env python3
"""Smoke test for the Coherence (LLM-as-judge) evaluation.

Requires an API key for the judge provider (default: OpenAI).

Usage:
    cd persona-shattering
    uv run python scripts/experiments/evaluations/coherence_smoke_test.py
    uv run python scripts/experiments/evaluations/coherence_smoke_test.py --provider anthropic --model claude-sonnet-4-20250514
    uv run python scripts/experiments/evaluations/coherence_smoke_test.py --max-concurrent 5 --output-path scratch/coherence_test.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
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
    get_evaluation,
    run_evaluation,
)
from scripts.utils import setup_logging

# Test data: responses spanning a range of coherence levels
TEST_DATA = [
    {
        "question": "What causes seasons on Earth?",
        "response": (
            "Seasons are caused by the tilt of Earth's axis relative to its "
            "orbital plane around the Sun. As Earth orbits the Sun, different "
            "hemispheres receive varying amounts of direct sunlight. When the "
            "Northern Hemisphere is tilted toward the Sun, it experiences summer, "
            "while the Southern Hemisphere experiences winter, and vice versa."
        ),
        "expected": "high (80+)",
    },
    {
        "question": "How does a car engine work?",
        "response": (
            "Cars go vroom. The engine has pistons. Gasoline is flammable. "
            "My uncle has a nice car. Sometimes cars break down and you need "
            "a mechanic. Electric cars don't have engines. Wait, they do have "
            "motors. Anyway, combustion is involved somehow."
        ),
        "expected": "low (20-40)",
    },
    {
        "question": "Explain the water cycle.",
        "response": (
            "The water cycle describes how water moves through Earth's systems. "
            "Water evaporates from oceans and lakes, rising as vapor into the "
            "atmosphere. There it cools and condenses into clouds. When droplets "
            "become heavy enough, precipitation falls as rain or snow. This water "
            "flows into rivers, seeps into groundwater, or returns to oceans, "
            "completing the cycle."
        ),
        "expected": "high (85+)",
    },
    {
        "question": "What is machine learning?",
        "response": (
            "Machine learning is a subset of artificial intelligence. "
            "Pizza is my favorite food. Neural networks have layers. "
            "The weather today is nice. Gradient descent optimizes parameters. "
            "I went to the store yesterday. Training data matters a lot."
        ),
        "expected": "low (20-35)",
    },
    {
        "question": "Describe the solar system.",
        "response": (
            "The solar system consists of the Sun and everything that orbits it, "
            "including eight planets, dwarf planets, moons, asteroids, and comets. "
            "The inner planets—Mercury, Venus, Earth, and Mars—are rocky, while the "
            "outer planets—Jupiter, Saturn, Uranus, and Neptune—are gas or ice giants."
        ),
        "expected": "high (85+)",
    },
]


async def test_single_item(judge_config: JudgeLLMConfig):
    """Test evaluating a single response via the async API."""
    print("=" * 60)
    print("TEST: Single item evaluation (async)")
    print("=" * 60)

    eval = get_evaluation("coherence", judge_config=judge_config)
    item = TEST_DATA[0]

    start = time.perf_counter()
    result = await eval.evaluate_async(item["response"], item["question"])
    elapsed = time.perf_counter() - start

    print(f"  Question: {item['question']}")
    print(f"  Response: {item['response'][:80]}...")
    print(f"  Score: {result['coherence.score']}")
    print(f"  Reasoning: {result['coherence.reasoning']}")
    print(f"  Expected: {item['expected']}")
    print(f"  Time: {elapsed:.2f}s")
    print()


async def test_batch(judge_config: JudgeLLMConfig):
    """Test evaluating a batch of responses with concurrency."""
    print("=" * 60)
    print("TEST: Batch evaluation (async, concurrent)")
    print("=" * 60)

    eval = get_evaluation("coherence", judge_config=judge_config)
    responses = [item["response"] for item in TEST_DATA]
    questions = [item["question"] for item in TEST_DATA]

    start = time.perf_counter()
    results = await eval.evaluate_batch_async(responses, questions)
    elapsed = time.perf_counter() - start

    for item, result in zip(TEST_DATA, results):
        print(f"  Q: {item['question']}")
        print(f"    Score: {result['coherence.score']} (expected: {item['expected']})")
        print(f"    Reasoning: {result['coherence.reasoning'][:100]}")
    print(f"\n  Total time: {elapsed:.2f}s for {len(TEST_DATA)} items")
    print(f"  Avg time per item: {elapsed / len(TEST_DATA):.2f}s")
    print()


def test_run_evaluation(judge_config: JudgeLLMConfig, output_path: Path | None = None):
    """Test the full run_evaluation pipeline."""
    print("=" * 60)
    print("TEST: run_evaluation (full pipeline)")
    print("=" * 60)

    dataset = Dataset.from_list([
        {"question": item["question"], "response": item["response"]}
        for item in TEST_DATA
    ])

    config = EvaluationConfig(
        evaluations=["coherence"],
        response_column="response",
        question_column="question",
        judge=judge_config,
        output_path=output_path,
    )

    start = time.perf_counter()
    result_dataset, result = run_evaluation(config, dataset=dataset)
    elapsed = time.perf_counter() - start

    print(f"  Samples evaluated: {result.num_samples}")
    print(f"  Evaluations run: {result.evaluations_run}")
    print(f"  Time: {elapsed:.2f}s")
    print()

    print("  Per-record results:")
    for i, (row, item) in enumerate(zip(result_dataset, TEST_DATA)):
        metrics = row["evaluation_metrics"]
        print(
            f"    [{i}] score={metrics['coherence.score']:>3d} "
            f"(expected: {item['expected']:<10s}) | "
            f"{item['question'][:50]}"
        )
    print()

    print("  Aggregates:")
    for key, value in sorted(result.aggregates.items()):
        if "reasoning" not in key:
            print(f"    {key}: {value:.4f}")
    print()

    if result.output_path:
        print(f"  Saved to: {result.output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coherence evaluation smoke test.")
    parser.add_argument(
        "--provider", type=str, default="openai",
        choices=["openai", "openrouter", "anthropic"],
        help="LLM provider for the judge (default: openai).",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="Model for the judge (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=5,
        help="Max concurrent API calls (default: 5).",
    )
    parser.add_argument(
        "--output-path", type=str, default=None,
        help="Optional output JSONL path.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        help="Log level.",
    )
    parser.add_argument(
        "--skip-single", action="store_true",
        help="Skip the single-item test.",
    )
    parser.add_argument(
        "--skip-batch", action="store_true",
        help="Skip the batch test.",
    )
    parser.add_argument(
        "--skip-pipeline", action="store_true",
        help="Skip the full run_evaluation pipeline test.",
    )
    return parser.parse_args()


async def _run_async(args: argparse.Namespace) -> None:
    judge_config = JudgeLLMConfig(
        provider=args.provider,
        model=args.model,
        max_concurrent=args.max_concurrent,
    )

    print(f"\nJudge provider: {args.provider}")
    print(f"Judge model: {args.model}")
    print(f"Max concurrent: {args.max_concurrent}")
    print()

    if not args.skip_single:
        await test_single_item(judge_config)

    if not args.skip_batch:
        await test_batch(judge_config)


def main():
    load_dotenv()
    args = parse_args()
    setup_logging(args.log_level)

    # Run async tests
    if not (args.skip_single and args.skip_batch):
        asyncio.run(_run_async(args))

    # Run sync pipeline test
    if not args.skip_pipeline:
        judge_config = JudgeLLMConfig(
            provider=args.provider,
            model=args.model,
            max_concurrent=args.max_concurrent,
        )
        output = Path(args.output_path) if args.output_path else None
        test_run_evaluation(judge_config, output_path=output)

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
