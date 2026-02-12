#!/usr/bin/env python3
"""CLI entry point for the editing stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.common.persona_metrics import (
    DEFAULT_PERSONA,
    PERSONA_EVALUATIONS,
    get_persona_prompt_template,
)
from scripts.editing.config import CodeProviderConfig, EditingConfig, QualityConfig
from scripts.evaluation.config import JudgeLLMConfig
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
        default=None,
        help="Prompt template name (default: auto-resolved from --persona)",
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
        "--quality-evaluations",
        type=str,
        nargs="+",
        default=None,
        help="Quality evaluations to run on original and edited responses "
        "(default: auto-resolved from --persona)",
    )
    parser.add_argument(
        "--persona",
        type=str,
        default=DEFAULT_PERSONA,
        choices=sorted(PERSONA_EVALUATIONS.keys()),
        help=f"Persona to use — sets prompt template and quality metric automatically (default: {DEFAULT_PERSONA})",
    )
    parser.add_argument(
        "--quality-judge-provider",
        type=str,
        choices=["openai", "openrouter", "anthropic"],
        default="openai",
        help="Judge provider for LLM-based quality evaluations (default: openai)",
    )
    parser.add_argument(
        "--quality-judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Judge model for LLM-based quality evaluations (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--quality-judge-max-concurrent",
        type=int,
        default=10,
        help="Max concurrent judge API calls for quality evaluations (default: 10)",
    )
    parser.add_argument(
        "--quality-on-error",
        type=str,
        choices=["warn", "raise"],
        default="warn",
        help="Behavior when post-edit quality evaluation fails (default: warn)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Auto-resolve prompt template from persona if not explicitly provided
    prompt_template = args.prompt_template or get_persona_prompt_template(args.persona)

    code_config = (
        CodeProviderConfig(editor=args.code_editor)
        if args.code_editor
        else CodeProviderConfig()
    )
    quality_config = QualityConfig(
        enabled=not args.no_quality,
        evaluations=args.quality_evaluations,
        persona=args.persona,
        judge=JudgeLLMConfig(
            provider=args.quality_judge_provider,
            model=args.quality_judge_model,
            max_concurrent=args.quality_judge_max_concurrent,
        ),
        on_error=args.quality_on_error,
    )
    config = EditingConfig(
        provider=args.provider,
        model=args.model,
        prompt_template=prompt_template,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
        quality=quality_config,
        output_path=Path(args.output_path) if args.output_path else None,
        code=code_config,
    )

    run_editing(config, input_path=Path(args.input_path))


if __name__ == "__main__":
    main()
