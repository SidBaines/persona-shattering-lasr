#!/usr/bin/env python3
"""Browser-hosted local chat with dynamic LoRA adapter controls.

Examples:
    uv run python scripts/visualisations/local_chat.py \
        --base-model meta-llama/Llama-3.1-8B-Instruct

    uv run python scripts/visualisations/local_chat.py \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --initial-adapter-key o_avoiding \
        --initial-adapter-key neutral_control
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from dotenv import load_dotenv

from scripts.visualisations.local_chat_web import launch_browser_chat
from scripts.visualisations.local_chat_web.prompting import TONE_INSTRUCTIONS

load_dotenv()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for browser local chat."""
    parser = argparse.ArgumentParser(
        description="Browser-based local chat with dynamic PEFT adapter controls."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model reference (hf repo id or local:// path).",
    )
    parser.add_argument(
        "--initial-adapter-key",
        action="append",
        default=[],
        help="Initial adapter key from the curated browser catalog (repeatable).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Torch dtype (e.g., bfloat16, float16, float32).",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for model loading (e.g., auto, cuda, cpu).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Max new tokens per assistant reply.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt.",
    )
    parser.add_argument(
        "--tone",
        type=str,
        choices=sorted(TONE_INSTRUCTIONS.keys()),
        default="balanced",
        help="Assistant tone preset.",
    )
    parser.add_argument(
        "--prompt-format",
        type=str,
        choices=["auto", "chat", "plain"],
        default="auto",
        help="Prompt formatting mode: auto, chat, or plain.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=24,
        help="How many recent turns to include in model prompts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the browser server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind the browser server.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        help="Open the browser automatically on launch.",
    )
    return parser.parse_args(argv)


def _launch_browser_chat(args: argparse.Namespace) -> None:
    launch_browser_chat(args)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for browser local chat."""
    args = parse_args(argv)
    try:
        _launch_browser_chat(args)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
