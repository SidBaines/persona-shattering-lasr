#!/usr/bin/env python3
"""CLI entry point for the editing stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.editing.config import EditingConfig, QualityConfig
from scripts.editing.run import run_editing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edit model responses using an LLM API.",
    )

    # Provider settings
    parser.add_argument(
        "--provider",
        type=str,
        choices=["anthropic", "openai"],
        default="anthropic",
        help="Editing provider: 'anthropic' or 'openai' (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model to use for editing (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="default_persona_shatter",
        help="Prompt template name (default: default_persona_shatter)",
    )

    # Concurrency
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent API requests (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)",
    )

    # Input / Output
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to input JSONL file (must have 'question' and 'response' columns)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save edited output JSONL file",
    )

    # Quality
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="Disable quality metric evaluation",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = EditingConfig(
        provider=args.provider,
        model=args.model,
        prompt_template=args.prompt_template,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
        quality=QualityConfig(enabled=not args.no_quality),
        output_path=Path(args.output_path) if args.output_path else None,
    )

    run_editing(config, input_path=Path(args.input_path))


if __name__ == "__main__":
    main()
