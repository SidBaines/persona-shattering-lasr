#!/usr/bin/env python3
"""CLI entry point for the editing stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common.persona_metrics import DEFAULT_PERSONA, PERSONA_METRICS
from scripts.editing.config import CodeProviderConfig, EditingConfig, QualityConfig
from scripts.editing.run import run_editing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edit model responses using an LLM API or a code-based editor.",
    )

    # Provider settings
    parser.add_argument(
        "--provider",
        type=str,
        choices=["anthropic", "openai", "code"],
        default="anthropic",
        help="Editing provider: 'anthropic', 'openai', or 'code' (default: anthropic)",
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
    parser.add_argument(
        "--code-editor",
        type=str,
        default=None,
        help="Code editor import path (e.g., scripts.editing.code_editors:reverse_text).",
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
    parser.add_argument(
        "--persona",
        type=str,
        default=DEFAULT_PERSONA,
        choices=sorted(PERSONA_METRICS.keys()),
        help=f"Persona metric for quality evaluation (default: {DEFAULT_PERSONA})",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
        timeout=args.timeout,
        quality=QualityConfig(enabled=not args.no_quality, persona=args.persona),
        output_path=Path(args.output_path) if args.output_path else None,
        code=code_config,
    )

    run_editing(config, input_path=Path(args.input_path))


if __name__ == "__main__":
    main()
