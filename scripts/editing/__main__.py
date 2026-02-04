#!/usr/bin/env python3
"""Edit model responses using an LLM API with quality tracking.

Usage:
    cd persona-shattering
    python -m scripts.editing configs/toy_model.yaml [--run-id <run_id>]

This script:
1. Loads inference output from scratch/{run_id}/inference_output.jsonl
2. Calls the LLM API (Anthropic or OpenAI) to edit responses
3. Tracks quality metrics (e.g., O-count reduction)
4. Saves edited responses to scratch/{run_id}/edited_dataset.jsonl
"""

import sys

from dotenv import load_dotenv

from scripts.config import load_config
from scripts.editing.run import run_editing
from scripts.utils import setup_logging


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.editing <config_path> [--run-id <run_id>]")
        sys.exit(1)

    load_dotenv()
    logger = setup_logging()

    config_path = sys.argv[1]

    # Parse optional --run-id argument
    run_id = None
    if "--run-id" in sys.argv:
        idx = sys.argv.index("--run-id")
        if idx + 1 < len(sys.argv):
            run_id = sys.argv[idx + 1]

    config = load_config(config_path)
    if run_id:
        config.run_id = run_id

    logger.info("Running editing with config: %s", config_path)
    logger.info("Provider: %s", config.editing.provider)
    logger.info("Model: %s", config.editing.model)
    logger.info("Max concurrent: %d", config.editing.max_concurrent)
    logger.info("Run ID: %s", config.run_id or "(will be auto-generated)")

    result = run_editing(config)

    # Print some sample outputs
    print("\n" + "=" * 60)
    print("SAMPLE EDITED OUTPUTS")
    print("=" * 60)
    for i, record in enumerate(result.select(range(min(3, len(result))))):
        print(f"\n--- Sample {i+1} ---")
        question = record["question"]
        original = record["response"]
        edited = record["edited_response"]
        if len(question) > 80:
            question = question[:80] + "..."
        if len(original) > 150:
            original = original[:150] + "..."
        if len(edited) > 150:
            edited = edited[:150] + "..."
        print(f"Question: {question}")
        print(f"Original: {original}")
        print(f"Edited:   {edited}")

        # Show quality metrics if available
        metrics = record.get("quality_metrics", {})
        if metrics:
            o_orig = metrics.get("count_o.original", "?")
            o_edit = metrics.get("count_o.edited", "?")
            o_delta = metrics.get("count_o.delta", "?")
            print(f"O-count:  {o_orig} -> {o_edit} (delta: {o_delta})")

    print("\n" + "=" * 60)
    print(f"Total samples edited: {len(result)}")


if __name__ == "__main__":
    main()
