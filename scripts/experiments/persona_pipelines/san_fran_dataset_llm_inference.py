#!/usr/bin/env python3
"""San Fran dataset stage 1: local inference only.

Usage:
    uv run python scripts/experiments/persona_pipelines/san_fran_dataset_llm_inference.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.inference import InferenceConfig, run_inference
from scripts.utils import write_jsonl


DATASET_NAME = "vicgalle/alpaca-gpt4"
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# HF_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SAMPLES = 200  # Set to None for full dataset
NUM_RESPONSES_PER_PROMPT = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local inference for San Fran dataset generation."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id (default: auto timestamp).",
    )
    parser.add_argument(
        "--num-responses-per-prompt",
        type=int,
        default=NUM_RESPONSES_PER_PROMPT,
        help="Number of responses to generate per prompt.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the local inference stage."""
    args = _parse_args()
    for cache_dir in (
        "/workspace/.cache",
        "/workspace/.cache/huggingface",
        "/workspace/.cache/huggingface/hub",
        "/workspace/.cache/huggingface/datasets",
        "/workspace/.cache/huggingface/transformers",
    ):
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    load_dotenv()

    run_id = args.run_id or f"san-fran-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    scratch_dir = Path("scratch") / run_id
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("SAN FRAN DATASET - STAGE 1 (INFERENCE)")
    print(f"Run ID: {run_id}")
    print(f"Output: {scratch_dir}")
    print(f"{'='*60}\n")

    inference_config = InferenceConfig(
        model=HF_MODEL,
        provider="local",
        dataset=DatasetConfig(
            source="huggingface",
            name=DATASET_NAME,
            split="train",
            max_samples=MAX_SAMPLES,
        ),
        generation=GenerationConfig(
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            batch_size=16,
            num_responses_per_prompt=args.num_responses_per_prompt,
        ),
        output_path=scratch_dir / "inference_output.jsonl",
    )

    inference_dataset, inference_result = run_inference(inference_config)
    print(f"\nGenerated {inference_result.num_samples} responses")
    print(f"Saved to: {inference_result.output_path}")

    pairs_path = scratch_dir / "question_response_pairs.jsonl"
    write_jsonl(
        [
            {"question": rec["question"], "response": rec["response"]}
            for rec in inference_dataset.to_list()
        ],
        pairs_path,
    )
    print(f"Saved question/response pairs to: {pairs_path}")


if __name__ == "__main__":
    main()
