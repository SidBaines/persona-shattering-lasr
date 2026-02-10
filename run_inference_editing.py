#!/usr/bin/env python3
"""Run inference and editing stages for 5 samples."""

from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from scripts.common.config import (
    DatasetConfig,
    GenerationConfig,
    ModelConfig,
)
from scripts.inference import run_inference, InferenceConfig, LocalProviderConfig
from scripts.editing import run_editing, EditingConfig


def main():
    """Run inference and editing for 5 samples."""
    load_dotenv()

    # Generate unique run ID
    run_id = f"test-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    scratch_dir = Path("scratch") / run_id
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"INFERENCE + EDITING TEST RUN")
    print(f"Run ID: {run_id}")
    print(f"Output: {scratch_dir}")
    print(f"{'='*60}\n")

    # =========================================================================
    # Stage 1: Inference - Generate responses from base model
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 1: INFERENCE")
    print(f"{'='*60}\n")

    inference_config = InferenceConfig(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        provider="local",
        local=LocalProviderConfig(
            dtype="bfloat16",
            device_map="auto",
        ),
        dataset=DatasetConfig(
            source="huggingface",
            name="vicgalle/alpaca-gpt4",
            split="train",
            max_samples=5,  # Only 5 samples as requested
        ),
        generation=GenerationConfig(
            max_new_tokens=500,
            temperature=0.7,
            batch_size=8,
        ),
        output_path=scratch_dir / "inference_output.jsonl",
    )

    inference_dataset, inference_result = run_inference(inference_config)
    print(f"\nGenerated {inference_result.num_samples} responses")
    print(f"Saved to: {inference_result.output_path}")

    # =========================================================================
    # Stage 2: Editing - Use LLM to edit responses
    # =========================================================================
    print(f"\n{'='*60}")
    print("STAGE 2: EDITING")
    print(f"{'='*60}\n")

    editing_config = EditingConfig(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        prompt_template="default_persona_shatter",
        max_concurrent=10,
        output_path=scratch_dir / "edited_dataset.jsonl",
    )

    edited_dataset, editing_result = run_editing(editing_config, dataset=inference_dataset)
    print(f"\nEdited {editing_result.num_samples} responses ({editing_result.num_failed} failed)")
    print(f"Tokens used: {editing_result.total_input_tokens} input, {editing_result.total_output_tokens} output")
    print(f"Saved to: {editing_result.output_path}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"Output directory: {scratch_dir}")
    print(f"Inference output: {inference_result.output_path}")
    print(f"Edited output: {editing_result.output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
