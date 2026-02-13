#!/usr/bin/env python3
"""San Fran dataset stage 2: API-LLM editing from inference output.

Usage:
    uv run python scripts/experiments/persona_pipelines/san_fran_dataset_llm_editing.py \
        --input-path scratch/<run_id>/inference_output.jsonl

    # Optional tuning for rate limits:
    uv run python scripts/experiments/persona_pipelines/san_fran_dataset_llm_editing.py \
        --input-path scratch/<run_id>/inference_output.jsonl \
        --max-concurrent 6 \
        --retry-max-retries 10 \
        --retry-backoff-factor 1.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from datasets import Dataset
from dotenv import load_dotenv

from scripts.editing import EditingConfig, OpenAIProviderConfig, QualityConfig, run_editing
from scripts.editing.config import RetryConfig
from scripts.utils import read_jsonl


EDITOR_PROVIDER = "openai"
EDITOR_MODEL = "gpt-5-nano-2025-08-07"
EDITOR_PROMPT_TEMPLATE = "sf_guy_casual_grammar"
EDITOR_MAX_TOKENS = 50000
DEFAULT_MAX_CONCURRENT = 20
DEFAULT_RETRY_MAX_RETRIES = 8
DEFAULT_RETRY_BACKOFF_FACTOR = 1.5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run API-LLM editing on San Fran inference output."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to inference output JSONL (must include question/response columns).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional output path (default: sibling file named edited_dataset.jsonl).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help="Maximum concurrent editing API requests.",
    )
    parser.add_argument(
        "--retry-max-retries",
        type=int,
        default=DEFAULT_RETRY_MAX_RETRIES,
        help="Max retry attempts for API calls.",
    )
    parser.add_argument(
        "--retry-backoff-factor",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_FACTOR,
        help="Exponential backoff base delay in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the API-LLM editing stage."""
    args = _parse_args()
    load_dotenv()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    output_path = (
        Path(args.output_path)
        if args.output_path
        else input_path.with_name("edited_dataset.jsonl")
    )

    print(f"\n{'='*60}")
    print("SAN FRAN DATASET - STAGE 2 (LLM EDITING)")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Retry: attempts={args.retry_max_retries}, backoff={args.retry_backoff_factor}s")
    print(f"{'='*60}\n")

    records = read_jsonl(input_path)
    dataset = Dataset.from_list(records)

    editing_config = EditingConfig(
        provider=EDITOR_PROVIDER,
        model=EDITOR_MODEL,
        prompt_template=EDITOR_PROMPT_TEMPLATE,
        openai=OpenAIProviderConfig(max_tokens=EDITOR_MAX_TOKENS),
        max_concurrent=args.max_concurrent,
        quality=QualityConfig(enabled=False),
        retry=RetryConfig(
            max_retries=args.retry_max_retries,
            backoff_factor=args.retry_backoff_factor,
        ),
        output_path=output_path,
    )

    edited_dataset, editing_result = run_editing(editing_config, dataset=dataset)
    print(
        f"\nEdited {editing_result.num_samples} responses "
        f"({editing_result.num_failed} failed)"
    )
    print(f"Saved to: {editing_result.output_path}")
    print(f"Returned dataset rows: {len(edited_dataset)}")


if __name__ == "__main__":
    main()
