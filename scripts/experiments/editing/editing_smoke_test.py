#!/usr/bin/env python3
"""Smoke test for editing that uses inference providers under the hood."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from datasets import Dataset

from scripts.editing import EditingConfig, run_editing
from scripts.editing.config import CodeProviderConfig, QualityConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Smoke test editing via the inference providers.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["anthropic", "openai", "code"],
        default="anthropic",
        help="Provider to use (default: anthropic).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model name (used for anthropic, or default for openai).",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default=None,
        help="Override OpenAI model name (optional).",
    )
    parser.add_argument(
        "--code-editor",
        type=str,
        default=None,
        help="Code editor import path (optional).",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="default_persona_shatter",
        help="Editing prompt template name.",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Optional JSONL path with question/response fields.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional output JSONL path. Defaults to scratch/.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max output tokens for the editing model.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Max concurrent requests (default: 4).",
    )
    parser.add_argument(
        "--disable-quality",
        action="store_true",
        help="Disable quality metrics reporting.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Describe the ocean in one sentence.",
        help="Sample question (used when --input-path is not provided).",
    )
    parser.add_argument(
        "--response",
        type=str,
        default="The ocean is a powerful and mysterious body of water.",
        help="Sample response (used when --input-path is not provided).",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EditingConfig:
    """Build an EditingConfig from CLI args."""
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        run_id = datetime.now().strftime("editing-smoke-%Y%m%d-%H%M%S")
        output_path = Path("scratch") / run_id / "edited.jsonl"

    code_config = (
        CodeProviderConfig(editor=args.code_editor)
        if args.code_editor
        else CodeProviderConfig()
    )
    config = EditingConfig(
        provider=args.provider,
        model=args.model,
        prompt_template=args.prompt_template,
        max_concurrent=args.max_concurrent,
        output_path=output_path,
        quality=QualityConfig(enabled=not args.disable_quality),
        code=code_config,
    )

    config.openai.max_tokens = args.max_tokens
    config.anthropic.max_tokens = args.max_tokens
    if args.openai_model:
        config.openai.model = args.openai_model

    return config


def build_dataset(args: argparse.Namespace) -> Dataset:
    """Build a small in-memory dataset for editing."""
    records = [
        {
            "question": args.question,
            "response": args.response,
        },
        {
            "question": "Write a short note about rainbows.",
            "response": "Rainbows are colorful arcs that appear after rain and sunlight.",
        },
    ]
    return Dataset.from_list(records)


def main() -> None:
    """Run a minimal editing smoke test."""
    load_dotenv()
    args = parse_args()
    config = build_config(args)

    if args.input_path:
        dataset, result = run_editing(config, input_path=Path(args.input_path))
    else:
        dataset = build_dataset(args)
        dataset, result = run_editing(config, dataset=dataset)

    print("\nEditing complete")
    print(f"Provider: {config.provider}")
    if config.provider == "code":
        print(f"Code editor: {config.code.editor}")
    else:
        print(f"Model: {config.model}")
    print(f"Edited samples: {result.num_samples}")
    print(f"Failed samples: {result.num_failed}")
    print(
        "Tokens used: "
        f"{result.total_input_tokens} input, "
        f"{result.total_output_tokens} output"
    )
    if result.output_path:
        print(f"Output saved to: {result.output_path}")

    if len(dataset) > 0:
        first = dataset[0]
        print("\nFirst sample")
        print(f"Question: {first['question']}")
        print(f"Response: {first['response']}")
        print(f"Edited: {first['edited_response']}")


if __name__ == "__main__":
    main()
