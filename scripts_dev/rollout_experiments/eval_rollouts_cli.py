#!/usr/bin/env python3
"""CLI wrapper for evaluating rollouts with persona metrics.

Usage::

    python -m scripts_dev.rollout_experiments.eval_rollouts_cli \
        --root-dir scratch/monorepo/fine_tuning/.../rollout_sweep_lora_scale \
        --evaluations count_t coherence \
        --incremental
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.conversation_eval import MessageSelector
from src_dev.persona_metrics.eval_rollouts import (
    RolloutEvalConfig,
    evaluate_rollouts,
)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Evaluate rollouts with persona metrics.",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Root directory to search for rollouts/rollouts.jsonl files.",
    )
    parser.add_argument(
        "--evaluations",
        nargs="+",
        required=True,
        help="Evaluator names to run (e.g. count_t coherence).",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="Skip evaluators already present (default: true).",
    )
    parser.add_argument(
        "--no-incremental",
        action="store_false",
        dest="incremental",
        help="Re-run all evaluators even if already present.",
    )
    parser.add_argument(
        "--exclude-seed",
        action="store_true",
        help="Only evaluate rollout messages (exclude seed conversation).",
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        help="Only evaluate messages with these roles (e.g. assistant).",
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        help="Judge LLM provider (e.g. openrouter).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        help="Judge LLM model (e.g. openai/gpt-4o-mini).",
    )

    args = parser.parse_args()

    selector = None
    if args.exclude_seed or args.roles:
        selector = MessageSelector(
            exclude_seed=args.exclude_seed,
            roles=set(args.roles) if args.roles else None,
        )

    judge = None
    if args.judge_provider or args.judge_model:
        judge = JudgeLLMConfig(
            provider=args.judge_provider or "openrouter",
            model=args.judge_model or "openai/gpt-4o-mini",
        )

    config = RolloutEvalConfig(
        root_dir=args.root_dir,
        evaluations=args.evaluations,
        message_selector=selector,
        judge=judge,
        incremental=args.incremental,
    )

    result = evaluate_rollouts(config)
    print(
        f"\nEval complete: {result.num_files_processed} file(s), "
        f"{result.num_messages_evaluated} message(s), "
        f"evals: {result.evaluations_run}"
    )


if __name__ == "__main__":
    main()
