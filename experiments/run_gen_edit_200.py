#!/usr/bin/env python3
"""Generate and edit 200 samples using OpenRouter and Anthropic.

This experiment:
1. Generates 200 responses using OpenRouter
2. Edits those responses using Anthropic Claude
"""

from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

from scripts.inference import run_inference, InferenceConfig, OpenAIProviderConfig
from scripts.editing import run_editing, EditingConfig
from scripts.common.config import DatasetConfig, GenerationConfig


def main():
    """Run generation and editing phases with 200 samples."""
    load_dotenv()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"scratch/run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Starting Generation + Editing Run")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    # ============================================================
    # Phase 1: Generation with OpenRouter
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 1: GENERATION (OpenRouter)")
    print("="*60 + "\n")

    inference_config = InferenceConfig(
        # Use a good OpenRouter model (fast and capable)
        model="anthropic/claude-3.5-sonnet",
        provider="openai",

        # Dataset configuration
        dataset=DatasetConfig(
            source="huggingface",
            name="vicgalle/alpaca-gpt4",
            split="train",
            max_samples=200,
        ),

        # Generation settings
        generation=GenerationConfig(
            max_new_tokens=500,
            temperature=0.7,
            batch_size=1,  # Process one at a time for API
        ),

        # OpenRouter configuration
        openai=OpenAIProviderConfig(
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPEN_ROUTER_API_KEY",
        ),

        # Output path
        output_path=output_dir / "01_generated.jsonl",
    )

    print(f"Model: {inference_config.model}")
    print(f"Dataset: {inference_config.dataset.name}")
    print(f"Samples: {inference_config.dataset.max_samples}")
    print(f"Output: {inference_config.output_path}\n")

    dataset, inference_result = run_inference(inference_config)

    print(f"\n✓ Generated {inference_result.num_samples} responses")
    print(f"  Saved to: {inference_result.output_path}\n")

    # ============================================================
    # Phase 2: Editing with Anthropic
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 2: EDITING (Anthropic Claude)")
    print("="*60 + "\n")

    editing_config = EditingConfig(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        prompt_template="default_persona_shatter",
        max_concurrent=10,
        output_path=output_dir / "02_edited.jsonl",
    )

    print(f"Model: {editing_config.model}")
    print(f"Prompt: {editing_config.prompt_template}")
    print(f"Max concurrent: {editing_config.max_concurrent}")
    print(f"Output: {editing_config.output_path}\n")

    edited_dataset, editing_result = run_editing(editing_config, dataset=dataset)

    print(f"\n✓ Edited {editing_result.num_samples} responses")
    print(f"  Failed: {editing_result.num_failed}")
    print(f"  Input tokens: {editing_result.total_input_tokens:,}")
    print(f"  Output tokens: {editing_result.total_output_tokens:,}")
    print(f"  Saved to: {editing_result.output_path}\n")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - Generation: {inference_result.output_path.name}")
    print(f"  - Editing: {editing_result.output_path.name}")
    print(f"\nNext steps:")
    print(f"  1. Review the edited samples")
    print(f"  2. Run training: uv run python -m scripts.training --help")
    print()


if __name__ == "__main__":
    main()
