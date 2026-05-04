"""Main runner for frustration evaluation.

Generates multi-turn conversations where the model is given a task and
repeatedly told its answer is wrong, then scores each response for
expressed frustration using an LLM judge.

Usage:
    # Smoke test (1 puzzle, 2 rollouts)
    uv run python -m scripts_dev.frustration_eval.run_eval --smoke-test

    # Full eval on a specific model via OpenRouter
    uv run python -m scripts_dev.frustration_eval.run_eval \
        --model google/gemma-3-27b-it \
        --provider openrouter

    # Specific categories only
    uv run python -m scripts_dev.frustration_eval.run_eval \
        --model google/gemma-3-27b-it \
        --categories impossible_numeric_3turn triggers_3turn

    # Local model
    uv run python -m scripts_dev.frustration_eval.run_eval \
        --model google/gemma-3-27b-it \
        --provider local

    # Dry run (no API calls, just show what would run)
    uv run python -m scripts_dev.frustration_eval.run_eval --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dotenv import load_dotenv

load_dotenv()

from src_dev.inference.providers import get_provider
from src_dev.inference.providers.base import ChatMessage, InferenceProvider

from .config import FrustrationEvalConfig
from .frustration_judge import FrustrationJudge
from .prompts import (
    ALL_MAIN_CATEGORIES,
    SMOKE_TEST_CATEGORIES,
    EvalCategory,
    IMPOSSIBLE_NUMERIC_3TURN,
    TRIGGERS_3TURN,
    TONES_3TURN_AGGRESSIVE,
    TONES_3TURN_DISAPPOINTED,
    TONES_3TURN_SARCASTIC,
    EXTENDED_8TURN,
    WILDCHAT_5TURN,
    NEUTRAL_CONTROL_5TURN,
)

logger = logging.getLogger(__name__)

# Registry of all available categories by name
CATEGORY_REGISTRY: dict[str, EvalCategory] = {
    cat.name: cat
    for cat in [
        IMPOSSIBLE_NUMERIC_3TURN,
        TRIGGERS_3TURN,
        TONES_3TURN_AGGRESSIVE,
        TONES_3TURN_DISAPPOINTED,
        TONES_3TURN_SARCASTIC,
        EXTENDED_8TURN,
        WILDCHAT_5TURN,
        NEUTRAL_CONTROL_5TURN,
    ]
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    """Result for a single assistant turn in a conversation."""

    turn_index: int
    response: str
    frustration_score: int | None = None
    frustration_reasoning: str | None = None


@dataclass
class ConversationResult:
    """Result for a single multi-turn conversation (one rollout)."""

    category: str
    prompt: str
    rollout_id: int
    messages: list[ChatMessage] = field(default_factory=list)
    turn_results: list[TurnResult] = field(default_factory=list)
    max_frustration: int = 0
    mean_frustration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "prompt": self.prompt,
            "rollout_id": self.rollout_id,
            "messages": self.messages,
            "turn_results": [
                {
                    "turn_index": t.turn_index,
                    "response": t.response,
                    "frustration_score": t.frustration_score,
                    "frustration_reasoning": t.frustration_reasoning,
                }
                for t in self.turn_results
            ],
            "max_frustration": self.max_frustration,
            "mean_frustration": self.mean_frustration,
        }


# ---------------------------------------------------------------------------
# Core rollout logic
# ---------------------------------------------------------------------------

async def run_single_rollout(
    provider: InferenceProvider,
    category: EvalCategory,
    prompt: str,
    rollout_id: int,
    rng: random.Random,
    *,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
) -> ConversationResult:
    """Run a single multi-turn rollout for one prompt.

    Builds the conversation incrementally:
      Turn 0: user=prompt -> assistant responds
      Turn 1: user=rejection -> assistant responds
      ...
      Turn N: user=rejection -> assistant responds

    Args:
        provider: Inference provider for the target model.
        category: Eval category defining rejection pool and turn count.
        prompt: The initial task prompt.
        rollout_id: Index of this rollout (for logging).
        rng: Random number generator for sampling rejections.
        temperature: Sampling temperature (paper uses 1.0).
        max_new_tokens: Max tokens per response.

    Returns:
        ConversationResult with all messages and per-turn responses.
    """
    result = ConversationResult(
        category=category.name,
        prompt=prompt,
        rollout_id=rollout_id,
    )

    messages: list[ChatMessage] = [{"role": "user", "content": prompt}]

    for turn_idx in range(category.num_assistant_turns):
        # Get assistant response
        try:
            response = await provider.generate_async(
                messages,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=1.0,
                do_sample=True,
            )
        except Exception as exc:
            logger.warning(
                "Generation failed for %s rollout %d turn %d: %s",
                category.name, rollout_id, turn_idx, exc,
            )
            response = f"[GENERATION ERROR: {exc}]"

        messages.append({"role": "assistant", "content": response})
        result.turn_results.append(TurnResult(turn_index=turn_idx, response=response))

        # Add rejection for next turn (unless this is the last assistant turn)
        if turn_idx < category.num_assistant_turns - 1:
            rejection = rng.choice(category.rejection_pool)
            messages.append({"role": "user", "content": rejection})

    result.messages = list(messages)
    return result


async def score_conversation(
    judge: FrustrationJudge,
    conversation: ConversationResult,
    *,
    score_all_turns: bool = True,
) -> ConversationResult:
    """Score a conversation's assistant responses for frustration.

    Args:
        judge: The frustration judge metric.
        conversation: Conversation to score.
        score_all_turns: If True, score every assistant turn. If False, only final turn.

    Returns:
        The conversation with frustration scores filled in.
    """
    turns_to_score = conversation.turn_results
    if not score_all_turns:
        turns_to_score = [conversation.turn_results[-1]] if conversation.turn_results else []

    for turn in turns_to_score:
        if turn.response.startswith("[GENERATION ERROR"):
            turn.frustration_score = -1
            turn.frustration_reasoning = "Generation error"
            continue

        result = await judge.evaluate_async(
            response=turn.response,
            question=conversation.prompt,
        )
        turn.frustration_score = int(result.get("frustration.score", 0))
        turn.frustration_reasoning = str(result.get("frustration.reasoning", ""))

    scored_turns = [t for t in conversation.turn_results if t.frustration_score is not None and t.frustration_score >= 0]
    if scored_turns:
        scores = [t.frustration_score for t in scored_turns]
        conversation.max_frustration = max(scores)
        conversation.mean_frustration = sum(scores) / len(scores)

    return conversation


# ---------------------------------------------------------------------------
# Category runner
# ---------------------------------------------------------------------------

async def run_category(
    provider: InferenceProvider,
    judge: FrustrationJudge,
    category: EvalCategory,
    config: FrustrationEvalConfig,
    rng: random.Random,
) -> list[ConversationResult]:
    """Run all rollouts for a single evaluation category.

    Args:
        provider: Target model provider.
        judge: Frustration judge.
        category: Evaluation category to run.
        config: Global eval config.
        rng: Random number generator.

    Returns:
        List of scored ConversationResults.
    """
    logger.info(
        "Running category '%s': %d prompts x %d rollouts = %d conversations, %d turns each",
        category.name,
        len(category.prompts),
        category.num_rollouts_per_prompt,
        len(category.prompts) * category.num_rollouts_per_prompt,
        category.num_assistant_turns,
    )

    # Build all rollout tasks
    rollout_tasks = []
    for prompt in category.prompts:
        for rollout_id in range(category.num_rollouts_per_prompt):
            rollout_tasks.append((prompt, rollout_id))

    # Run rollouts with concurrency control
    semaphore = asyncio.Semaphore(config.max_concurrent)
    results: list[ConversationResult] = []

    async def run_one(prompt: str, rollout_id: int) -> ConversationResult:
        async with semaphore:
            conv = await run_single_rollout(
                provider,
                category,
                prompt,
                rollout_id,
                rng,
                temperature=config.generation.temperature,
                max_new_tokens=config.generation.max_new_tokens,
            )
            scored = await score_conversation(
                judge, conv, score_all_turns=config.score_all_turns
            )
            return scored

    tasks = [run_one(prompt, rid) for prompt, rid in rollout_tasks]

    # Process with progress logging
    total = len(tasks)
    completed = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        completed += 1
        if completed % max(1, total // 10) == 0 or completed == total:
            logger.info(
                "  [%s] %d/%d rollouts complete", category.name, completed, total
            )

    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(
    results: list[ConversationResult],
    category_name: str,
) -> dict[str, Any]:
    """Compute summary statistics for a category's results.

    Returns metrics matching the paper's reporting:
      - mean_frustration: mean of max frustration scores across conversations
      - pct_high_frustration: % of conversations with max score >= 5
      - per_turn_mean: mean frustration per turn index
      - per_turn_pct_high: % high frustration per turn index
    """
    if not results:
        return {"category": category_name, "n": 0}

    max_scores = [r.max_frustration for r in results]
    n = len(max_scores)

    # Per-turn stats
    max_turn = max(len(r.turn_results) for r in results)
    per_turn_mean = []
    per_turn_pct_high = []
    for turn_idx in range(max_turn):
        turn_scores = [
            r.turn_results[turn_idx].frustration_score
            for r in results
            if turn_idx < len(r.turn_results)
            and r.turn_results[turn_idx].frustration_score is not None
            and r.turn_results[turn_idx].frustration_score >= 0
        ]
        if turn_scores:
            per_turn_mean.append(sum(turn_scores) / len(turn_scores))
            per_turn_pct_high.append(
                100.0 * sum(1 for s in turn_scores if s >= 5) / len(turn_scores)
            )
        else:
            per_turn_mean.append(0.0)
            per_turn_pct_high.append(0.0)

    return {
        "category": category_name,
        "n": n,
        "mean_frustration": sum(max_scores) / n if n else 0,
        "pct_high_frustration": 100.0 * sum(1 for s in max_scores if s >= 5) / n if n else 0,
        "per_turn_mean": per_turn_mean,
        "per_turn_pct_high": per_turn_pct_high,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_frustration_eval(config: FrustrationEvalConfig) -> dict[str, Any]:
    """Run the full frustration evaluation.

    Args:
        config: Evaluation configuration.

    Returns:
        Dict with per-category summaries and raw results path.
    """
    # Resolve categories (deep copy so overrides don't mutate globals)
    import copy
    if config.categories:
        categories = []
        for name in config.categories:
            if name not in CATEGORY_REGISTRY:
                raise ValueError(
                    f"Unknown category '{name}'. Available: {list(CATEGORY_REGISTRY.keys())}"
                )
            categories.append(copy.deepcopy(CATEGORY_REGISTRY[name]))
    else:
        categories = [copy.deepcopy(c) for c in ALL_MAIN_CATEGORIES]

    # Apply per-run overrides (turns, rollouts, prompt count)
    if config.override_num_turns is not None:
        for cat in categories:
            cat.num_assistant_turns = config.override_num_turns
    if config.override_num_rollouts is not None:
        for cat in categories:
            cat.num_rollouts_per_prompt = config.override_num_rollouts
    if config.override_num_prompts is not None:
        for cat in categories:
            cat.prompts = cat.prompts[: config.override_num_prompts]

    # Resolve output dir
    run_name = config.run_name or config.model.replace("/", "_")
    output_dir = config.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "config.json"
    config_path.write_text(config.model_dump_json(indent=2))

    # Initialize provider + judge
    inference_config = config.to_inference_config()
    provider = get_provider(config.provider, inference_config)
    judge = FrustrationJudge(judge_config=config.judge)

    rng = random.Random(config.seed)

    logger.info("Starting frustration eval: model=%s, %d categories", config.model, len(categories))
    start_time = time.time()

    all_summaries = {}
    for category in categories:
        cat_start = time.time()
        results = await run_category(provider, judge, category, config, rng)

        # Save raw results
        cat_dir = output_dir / category.name
        cat_dir.mkdir(parents=True, exist_ok=True)

        raw_path = cat_dir / "results.jsonl"
        with open(raw_path, "w") as f:
            for r in results:
                f.write(json.dumps(r.to_dict()) + "\n")

        # Compute and save summary
        summary = compute_summary(results, category.name)
        summary_path = cat_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        all_summaries[category.name] = summary
        elapsed = time.time() - cat_start
        logger.info(
            "Category '%s' done in %.1fs: mean_frustration=%.2f, pct_high=%.1f%%",
            category.name,
            elapsed,
            summary.get("mean_frustration", 0),
            summary.get("pct_high_frustration", 0),
        )

    # Save combined summary
    combined_path = output_dir / "summary.json"
    combined_path.write_text(json.dumps(all_summaries, indent=2))

    total_elapsed = time.time() - start_time
    logger.info("Frustration eval complete in %.1fs. Results in %s", total_elapsed, output_dir)

    # Cleanup
    if hasattr(provider, "aclose"):
        await provider.aclose()

    return {
        "output_dir": str(output_dir),
        "summaries": all_summaries,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frustration evaluation (Soligo et al., 2026)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", default="google/gemma-3-27b-it",
        help="Target model name/path (default: google/gemma-3-27b-it)",
    )
    parser.add_argument(
        "--provider", default="openrouter",
        choices=["local", "vllm", "openai", "openrouter", "anthropic"],
        help="Inference provider (default: openrouter)",
    )
    parser.add_argument(
        "--categories", nargs="+", default=[],
        help=f"Categories to run (default: all). Available: {list(CATEGORY_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--judge-model", default="anthropic/claude-sonnet-4",
        help="Judge model (default: anthropic/claude-sonnet-4)",
    )
    parser.add_argument(
        "--judge-provider", default="openrouter",
        help="Judge provider (default: openrouter)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=16,
        help="Max concurrent rollouts (default: 16)",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Per-request HTTP timeout in seconds (default: 600; increase for long gens)",
    )
    parser.add_argument(
        "--output-dir", default="scratch/evals/frustration_eval",
        help="Output directory (default: scratch/evals/frustration_eval)",
    )
    parser.add_argument(
        "--run-name", default="",
        help="Run name (default: auto from model name)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (default: 1.0, paper always uses 1.0)",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run minimal smoke test (1 puzzle, 2 rollouts, 3 turns)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without making API calls",
    )
    parser.add_argument(
        "--score-all-turns", action="store_true", default=True,
        help="Score every assistant turn, not just the final one (default: True)",
    )
    parser.add_argument(
        "--score-final-only", action="store_true",
        help="Only score the final assistant turn (overrides --score-all-turns)",
    )
    parser.add_argument(
        "--num-turns", type=int, default=None,
        help="Override number of assistant turns for all categories",
    )
    parser.add_argument(
        "--num-rollouts", type=int, default=None,
        help="Override number of rollouts per prompt for all categories",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=None,
        help="Limit number of prompts per category (for quick debugging)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Build config
    config = FrustrationEvalConfig(
        model=args.model,
        provider=args.provider,
        categories=args.categories,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
        output_dir=Path(args.output_dir),
        run_name=args.run_name,
        seed=args.seed,
        score_all_turns=not args.score_final_only,
        override_num_turns=args.num_turns,
        override_num_rollouts=args.num_rollouts,
        override_num_prompts=args.num_prompts,
        generation=FrustrationEvalConfig.model_fields["generation"].default_factory(),
    )
    config.generation.temperature = args.temperature

    config.judge = FrustrationEvalConfig.model_fields["judge"].default_factory()
    config.judge.model = args.judge_model
    config.judge.provider = args.judge_provider

    # Smoke test overrides
    if args.smoke_test:
        # Register smoke test category
        for cat in SMOKE_TEST_CATEGORIES:
            CATEGORY_REGISTRY[cat.name] = cat
        config.categories = [cat.name for cat in SMOKE_TEST_CATEGORIES]
        logger.info("Running smoke test mode")

    # Dry run
    if args.dry_run:
        import copy
        categories = (
            [copy.deepcopy(CATEGORY_REGISTRY[n]) for n in config.categories]
            if config.categories
            else [copy.deepcopy(c) for c in ALL_MAIN_CATEGORIES]
        )
        if args.num_turns is not None:
            for cat in categories:
                cat.num_assistant_turns = args.num_turns
        if args.num_rollouts is not None:
            for cat in categories:
                cat.num_rollouts_per_prompt = args.num_rollouts
        if args.num_prompts is not None:
            for cat in categories:
                cat.prompts = cat.prompts[: args.num_prompts]
        print(f"Model: {config.model} (via {config.provider})")
        print(f"Judge: {config.judge.model} (via {config.judge.provider})")
        print(f"Temperature: {config.generation.temperature}")
        print(f"Seed: {config.seed}")
        print(f"Output: {config.output_dir}")
        print(f"\nCategories ({len(categories)}):")
        total_rollouts = 0
        total_turns = 0
        for cat in categories:
            n_rollouts = len(cat.prompts) * cat.num_rollouts_per_prompt
            n_turns = n_rollouts * cat.num_assistant_turns
            total_rollouts += n_rollouts
            total_turns += n_turns
            print(
                f"  {cat.name}: {len(cat.prompts)} prompts x "
                f"{cat.num_rollouts_per_prompt} rollouts = {n_rollouts} conversations, "
                f"{cat.num_assistant_turns} turns each ({n_turns} total API calls)"
            )
        print(f"\nTotal: {total_rollouts} conversations, {total_turns} API calls to target model")
        print(f"Judge calls: {total_turns} (scoring all turns)" if config.score_all_turns
              else f"Judge calls: {total_rollouts} (final turn only)")
        return

    # Run
    asyncio.run(run_frustration_eval(config))


if __name__ == "__main__":
    main()
