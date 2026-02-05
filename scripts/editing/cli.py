"""CLI entry point for scripts-based editing stage."""

from __future__ import annotations

import click
from dotenv import load_dotenv

from scripts.config import load_config
from scripts.editing.run import run_editing
from scripts.utils import setup_logging


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--run-id", default=None, help="Override run_id from config.")
def main(config_path: str, run_id: str | None) -> None:
    """Run the editing stage using scripts implementations."""
    load_dotenv()
    logger = setup_logging()

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
