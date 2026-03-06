#!/usr/bin/env python3
"""Generate long-context assistant/user rollouts from seed prompts."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.inference import InferenceConfig, LocalProviderConfig
from scripts.rollout_generation import (
    ContextPolicyConfig,
    FailurePolicyConfig,
    RolloutGenerationConfig,
    UserSimulatorConfig,
    run_rollout_generation,
)


DEFAULT_DATASET_PATH = "datasets/assistant-axis-extraction-questions.jsonl"
DEFAULT_ASSISTANT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_USER_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_NUM_ROLLOUTS_PER_PROMPT = 4
DEFAULT_NUM_ASSISTANT_TURNS = 8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate long-context assistant/user rollouts.")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--num-rollouts-per-prompt", type=int, default=DEFAULT_NUM_ROLLOUTS_PER_PROMPT)
    parser.add_argument("--num-assistant-turns", type=int, default=DEFAULT_NUM_ASSISTANT_TURNS)

    parser.add_argument("--assistant-model", type=str, default=DEFAULT_ASSISTANT_MODEL)
    parser.add_argument("--assistant-temperature", type=float, default=1.0)
    parser.add_argument("--assistant-top-p", type=float, default=0.95)
    parser.add_argument("--assistant-max-new-tokens", type=int, default=2048)
    parser.add_argument("--assistant-batch-size", type=int, default=8)
    parser.add_argument(
        "--assistant-truncate-inputs",
        action="store_true",
        help="Enable tokenizer-side truncation for local assistant prompts (default: disabled).",
    )

    parser.add_argument("--user-provider", type=str, default="openai")
    parser.add_argument("--user-model", type=str, default=DEFAULT_USER_MODEL)
    parser.add_argument("--user-prompt-template", type=str, default="typical_user")
    parser.add_argument(
        "--user-prompt-format",
        choices=["chat_messages", "single_turn_text"],
        default="single_turn_text",
    )
    parser.add_argument("--user-temperature", type=float, default=1.0)
    parser.add_argument("--user-top-p", type=float, default=0.95)
    parser.add_argument("--user-max-new-tokens", type=int, default=20000)
    parser.add_argument("--user-batch-size", type=int, default=16)
    parser.add_argument("--user-max-concurrent", type=int, default=64)

    parser.add_argument("--assistant-max-attempts-per-turn", type=int, default=3)
    parser.add_argument("--user-max-attempts-per-turn", type=int, default=3)

    parser.add_argument("--transcript-variant", type=str, default="rollout_base")
    parser.add_argument("--context-policy", choices=["full_history", "token_budget"], default="full_history")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    load_dotenv()

    run_id = args.run_id or f"rollout-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path("scratch") / "runs" / run_id

    config = RolloutGenerationConfig(
        dataset=DatasetConfig(
            source="local",
            path=args.dataset_path,
            max_samples=args.max_samples,
        ),
        run_dir=run_dir,
        num_assistant_turns=args.num_assistant_turns,
        num_rollouts_per_prompt=args.num_rollouts_per_prompt,
        assistant_inference=InferenceConfig(
            model=args.assistant_model,
            provider="local",
            local=LocalProviderConfig(
                prompt_format="chat",
                truncate_inputs=bool(args.assistant_truncate_inputs),
            ),
            generation=GenerationConfig(
                max_new_tokens=args.assistant_max_new_tokens,
                temperature=args.assistant_temperature,
                top_p=args.assistant_top_p,
                do_sample=True,
                batch_size=args.assistant_batch_size,
                num_responses_per_prompt=1,
            ),
        ),
        user_simulator=UserSimulatorConfig(
            provider=args.user_provider,
            model=args.user_model,
            prompt_template=args.user_prompt_template,
            prompt_format=args.user_prompt_format,
            generation=GenerationConfig(
                max_new_tokens=args.user_max_new_tokens,
                temperature=args.user_temperature,
                top_p=args.user_top_p,
                do_sample=True,
                batch_size=args.user_batch_size,
                num_responses_per_prompt=1,
            ),
            max_concurrent=args.user_max_concurrent,
        ),
        transcript_variant=args.transcript_variant,
        context_policy=ContextPolicyConfig(mode=args.context_policy),
        failure_policy=FailurePolicyConfig(
            assistant_max_attempts_per_turn=args.assistant_max_attempts_per_turn,
            user_max_attempts_per_turn=args.user_max_attempts_per_turn,
        ),
        resume=not args.no_resume,
        overwrite_output=args.overwrite_output,
    )

    dataset, result = run_rollout_generation(config)

    print(f"Run dir: {run_dir}")
    print(f"Conversation export: {result.exports['conversation_training']}")
    print(f"Trace export: {result.exports['conversation_trace']}")
    print(f"Completed conversations: {result.num_completed}/{result.num_conversations}")
    print(
        f"Completed assistant turns: {result.num_assistant_turns_completed} / "
        f"{result.num_conversations * result.num_assistant_turns_target}"
    )
    print(f"Failed conversations: {result.num_failed}")
    print(f"Dataset rows available inline: {len(dataset)}")


if __name__ == "__main__":
    main()
