#!/usr/bin/env python3
"""Generate single-turn responses into a canonical run and upload by default."""

from __future__ import annotations

import argparse
from datetime import datetime

from dotenv import load_dotenv

from src_dev.common.config import DatasetConfig, GenerationConfig
from src_dev.inference import (
    AnthropicProviderConfig,
    InferenceConfig,
    LocalProviderConfig,
    OpenAIProviderConfig,
    OpenRouterProviderConfig,
    run_inference,
)
from src_dev.unsupervised_runs import (
    DEFAULT_UNSUPERVISED_HF_REPO_ID,
    ensure_response_run,
    response_run_dir,
    upload_response_run,
)
from src_dev.unsupervised_runs.io import slugify_component


DEFAULT_DATASET_PATH = "datasets/assistant-axis-extraction-questions.jsonl"
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate single-turn response runs.")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--dataset-source",
        choices=["local", "huggingface"],
        default="local",
    )
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dataset-seed", type=int, default=None)

    parser.add_argument(
        "--provider",
        choices=["local", "openai", "openrouter", "anthropic"],
        default="local",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-responses", type=int, default=50)
    parser.add_argument("--max-concurrent", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--local-prompt-format", choices=["auto", "chat", "plain"], default="auto")
    parser.add_argument("--local-chat-system-prompt", type=str, default=None)
    parser.add_argument("--local-truncate-inputs", action="store_true")
    parser.add_argument("--openai-base-url", type=str, default=None)
    parser.add_argument("--openrouter-app-url", type=str, default=None)
    parser.add_argument("--openrouter-app-name", type=str, default=None)
    parser.add_argument("--anthropic-max-tokens", type=int, default=None)

    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--max-attempts-per-sample", type=int, default=3)

    parser.add_argument("--hf-repo-id", type=str, default=DEFAULT_UNSUPERVISED_HF_REPO_ID)
    parser.add_argument("--no-hf-upload", action="store_true")
    return parser.parse_args()


def _default_run_id(model: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"singleturn-{slugify_component(model)}-{timestamp}"


def main() -> None:
    args = _parse_args()
    load_dotenv()

    run_id = args.run_id or _default_run_id(args.model)
    run_dir = response_run_dir(run_id)

    if not run_dir.exists() and not args.overwrite_output:
        ensure_response_run(run_id, repo_id=args.hf_repo_id, required=False)

    if not run_dir.exists() and args.dataset_source == "huggingface" and not args.dataset_name:
        raise ValueError("--dataset-name is required when --dataset-source=huggingface.")
    if not run_dir.exists() and args.dataset_source == "local" and not args.dataset_path:
        raise ValueError("--dataset-path is required when --dataset-source=local.")

    dataset_config = DatasetConfig(
        source=args.dataset_source,
        path=args.dataset_path if args.dataset_source == "local" else None,
        name=args.dataset_name if args.dataset_source == "huggingface" else None,
        split=args.dataset_split,
        max_samples=args.max_samples,
        seed=args.dataset_seed,
    )

    config = InferenceConfig(
        model=args.model,
        provider=args.provider,
        dataset=dataset_config,
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            batch_size=args.batch_size,
            num_responses_per_prompt=args.num_responses,
        ),
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
        local=LocalProviderConfig(
            prompt_format=args.local_prompt_format,
            chat_system_prompt=args.local_chat_system_prompt,
            truncate_inputs=bool(args.local_truncate_inputs),
        ),
        openai=OpenAIProviderConfig(base_url=args.openai_base_url),
        openrouter=OpenRouterProviderConfig(
            app_url=args.openrouter_app_url,
            app_name=args.openrouter_app_name,
        ),
        anthropic=AnthropicProviderConfig(max_tokens=args.anthropic_max_tokens),
        run_dir=run_dir,
        system_prompt=args.system_prompt,
        max_attempts_per_sample=args.max_attempts_per_sample,
        resume=not args.no_resume,
        overwrite_output=args.overwrite_output,
    )

    dataset, result = run_inference(config)
    print(f"Run dir: {run_dir}")
    print(f"Canonical samples: {result.output_path}")
    print(f"Rows: {len(dataset)}")
    print(f"Failed: {result.num_failed}")

    if not args.no_hf_upload:
        hf_url = upload_response_run(run_id, repo_id=args.hf_repo_id)
        print(f"Hugging Face dataset: {hf_url}")


if __name__ == "__main__":
    main()
