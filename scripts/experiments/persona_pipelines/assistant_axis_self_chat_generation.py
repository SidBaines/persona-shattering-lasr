#!/usr/bin/env python3
"""Generate Assistant-axis self-chat conversations with OpenRouter."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.inference import InferenceConfig, OpenRouterProviderConfig
from scripts.self_chat_generation import (
    HfUploadConfig,
    SelfChatGenerationConfig,
    run_self_chat_generation,
)


DEFAULT_DATASET_PATH = "datasets/assistant-axis-extraction-questions.jsonl"
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Assistant-axis self-chat rollouts.")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dataset-seed", type=int, default=None)
    parser.add_argument("--num-rollouts-per-prompt", type=int, default=4)
    parser.add_argument("--num-generated-turns", type=int, default=8)
    parser.add_argument("--system-prompt", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-concurrent", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--app-url", type=str, default=None)
    parser.add_argument("--app-name", type=str, default=None)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--no-resume", action="store_true")

    parser.add_argument("--hf-upload", action="store_true")
    parser.add_argument("--hf-repo-id", type=str, default=None)
    parser.add_argument("--hf-path-in-repo", type=str, default="runs")
    parser.add_argument(
        "--hf-commit-message",
        type=str,
        default="Upload self-chat generation run",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    load_dotenv()

    run_id = args.run_id or f"self-chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path("scratch") / "runs" / run_id
    openrouter_config = OpenRouterProviderConfig(
        app_url=args.app_url,
        app_name=args.app_name,
    )
    inference = InferenceConfig(
        model=args.model,
        provider="openrouter",
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            batch_size=args.batch_size,
            num_responses_per_prompt=1,
        ),
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
        openrouter=openrouter_config,
    )

    config = SelfChatGenerationConfig(
        dataset=DatasetConfig(
            source="local",
            path=args.dataset_path,
            max_samples=args.max_samples,
            seed=args.dataset_seed,
        ),
        run_dir=run_dir,
        num_generated_turns=args.num_generated_turns,
        num_rollouts_per_prompt=args.num_rollouts_per_prompt,
        system_prompt=args.system_prompt,
        speaker_a_inference=inference,
        speaker_b_inference=None,
        resume=not args.no_resume,
        overwrite_output=args.overwrite_output,
        hf_upload=HfUploadConfig(
            enabled=args.hf_upload,
            repo_id=args.hf_repo_id,
            path_in_repo=args.hf_path_in_repo,
            commit_message=args.hf_commit_message,
        ),
    )

    dataset, result = run_self_chat_generation(config)
    print(f"Run dir: {run_dir}")
    print(f"Conversation export: {result.exports['conversation_training']}")
    print(f"Trace export: {result.exports['conversation_trace']}")
    print(f"Completed conversations: {result.num_completed}/{result.num_conversations}")
    print(
        f"Completed generated turns: {result.num_generated_turns_completed} / "
        f"{result.num_conversations * result.num_generated_turns_target}"
    )
    print(f"Failed conversations: {result.num_failed}")
    if result.hf_dataset_url:
        print(f"Hugging Face dataset: {result.hf_dataset_url}")
    print(f"Dataset rows available inline: {len(dataset)}")


if __name__ == "__main__":
    main()
