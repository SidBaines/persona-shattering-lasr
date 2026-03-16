#!/usr/bin/env python3
"""Prompt iteration script for conscientious editing variants.

Supports a tight iteration loop for refining editing prompt templates:

1. Run inference once (generates original base-model responses, stored in run-dir)
2. Apply multiple editing prompt variants to those same original responses
3. Export compare.jsonl (one row per question, variants as fields) for TUI review
4. Add new prompt versions to scripts/editing/prompts.py, re-run with new --prompts
5. Repeat — inference and existing edits are always skipped automatically

Usage:

    # First run: generate base-model responses + apply v1 prompts
    uv run python scripts/experiments/conscientious_iteration.py \\
        --run-dir scratch/c_iter_001 \\
        --prompts c+v1 c-v1 neutral_paraphrase_control

    # After adding c+v2, c-v2 to prompts.py, iterate on same run-dir:
    uv run python scripts/experiments/conscientious_iteration.py \\
        --run-dir scratch/c_iter_001 \\
        --prompts c+v2 c-v2

    # Review all variants side by side (command is also printed at end of run):
    uv run python scripts/jsonl_tui/cli.py scratch/c_iter_001/compare.jsonl \\
        --variant-fields original neutral_paraphrase_control c+v1 c-v1 c+v2 c-v2

Run-dir structure:
    <run-dir>/
    ├── original_responses.jsonl      # inference output; written once, then immutable
    ├── edits/
    │   ├── c+v1.jsonl                # one file per editing variant
    │   ├── c-v1.jsonl
    │   └── ...
    └── compare.jsonl                 # rebuilt on every run; use for TUI review
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.datasets import load_dataset_from_config
from scripts.editing import EditingConfig, QualityConfig, run_editing
from scripts.inference import InferenceConfig, run_inference
from scripts.utils import read_jsonl, write_jsonl

load_dotenv()

DEFAULT_DATASET = "vicgalle/alpaca-gpt4"
DEFAULT_INFERENCE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_INFERENCE_PROVIDER = "local"
DEFAULT_EDITING_MODEL = "claude-sonnet-4-20250514"
DEFAULT_EDITING_PROVIDER = "anthropic"
DEFAULT_MAX_SAMPLES = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterate on conscientious editing prompt variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Directory for storing run outputs. Created if it doesn't exist.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        required=True,
        metavar="TEMPLATE",
        help=(
            "Editing prompt template names to apply "
            "(e.g. c+v1 c-v1 neutral_paraphrase_control). "
            "Already-existing variants are skipped unless --overwrite-edits is set."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=f"Number of samples to generate via inference (default: {DEFAULT_MAX_SAMPLES}).",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HuggingFace dataset name for inference (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--inference-model",
        default=DEFAULT_INFERENCE_MODEL,
        help=f"Model name for local inference (default: {DEFAULT_INFERENCE_MODEL}).",
    )
    parser.add_argument(
        "--inference-provider",
        default=DEFAULT_INFERENCE_PROVIDER,
        help=f"Inference provider: local, openai, anthropic, openrouter (default: {DEFAULT_INFERENCE_PROVIDER}).",
    )
    parser.add_argument(
        "--editing-model",
        default=DEFAULT_EDITING_MODEL,
        help=f"Model name for editing (default: {DEFAULT_EDITING_MODEL}).",
    )
    parser.add_argument(
        "--editing-provider",
        default=DEFAULT_EDITING_PROVIDER,
        help=f"Editing provider: anthropic, openai (default: {DEFAULT_EDITING_PROVIDER}).",
    )
    parser.add_argument(
        "--overwrite-edits",
        action="store_true",
        help="Re-run editing even if the output file already exists.",
    )
    return parser.parse_args()


def _phase_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}\n")


def run_inference_phase(args: argparse.Namespace, original_path: Path) -> None:
    """Run inference to produce original responses. Skipped if output already exists."""
    if original_path.exists():
        row_count = sum(1 for _ in original_path.open() if _.strip())
        print(f"Skipping inference — already exists ({row_count} rows): {original_path}")
        return

    _phase_header("PHASE 1: INFERENCE")
    print(f"  Model:    {args.inference_model}")
    print(f"  Provider: {args.inference_provider}")
    print(f"  Dataset:  {args.dataset} (max {args.max_samples} samples)")
    print(f"  Output:   {original_path}\n")

    config = InferenceConfig(
        model=args.inference_model,
        provider=args.inference_provider,
        dataset=DatasetConfig(
            source="huggingface",
            name=args.dataset,
            split="train",
            max_samples=args.max_samples,
        ),
        generation=GenerationConfig(
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            batch_size=4,
        ),
        output_path=original_path,
    )
    _, result = run_inference(config)
    print(f"\nGenerated {result.num_samples} responses -> {original_path}")


def run_editing_phase(
    args: argparse.Namespace,
    original_path: Path,
    edits_dir: Path,
) -> None:
    """Run editing for each prompt variant, skipping variants that already have output."""
    _phase_header("PHASE 2: EDITING")
    print(f"  Prompts:  {args.prompts}")
    print(f"  Model:    {args.editing_model}")
    print(f"  Provider: {args.editing_provider}\n")

    edits_dir.mkdir(parents=True, exist_ok=True)

    original_dataset = load_dataset_from_config(
        DatasetConfig(source="local", path=str(original_path))
    )
    print(f"Loaded {len(original_dataset)} original responses from {original_path}\n")

    for prompt in args.prompts:
        edit_path = edits_dir / f"{prompt}.jsonl"
        if edit_path.exists() and not args.overwrite_edits:
            print(f"  Skipping {prompt!r} — already exists: {edit_path}")
            continue

        print(f"  Running {prompt!r} -> {edit_path}")
        config = EditingConfig(
            provider=args.editing_provider,
            model=args.editing_model,
            prompt_template=prompt,
            quality=QualityConfig(enabled=False),
            output_path=edit_path,
        )
        _, result = run_editing(config, dataset=original_dataset)
        print(f"    Done: {result.num_samples} edited, {result.num_failed} failed")


def build_compare_jsonl(
    original_path: Path,
    edits_dir: Path,
    compare_path: Path,
) -> list[str]:
    """Build compare.jsonl from original responses and all available edit files.

    Format: one record per question with variant texts as fields. Always rebuilt
    from scratch so that newly added variants appear automatically.

    Returns:
        Sorted list of variant names included (excluding 'original').
    """
    _phase_header("PHASE 3: BUILDING COMPARE DATASET")

    # Load original responses, keyed by question (preserve insertion order)
    originals: dict[str, str] = {}
    for row in read_jsonl(original_path):
        q = row.get("question", "")
        if q and q not in originals:
            originals[q] = row.get("response", "")

    # Collect all edit files (alphabetical for deterministic field ordering)
    edit_files = sorted(edits_dir.glob("*.jsonl"))
    variant_names = [f.stem for f in edit_files]

    edits_by_variant: dict[str, dict[str, str]] = {}
    for edit_file in edit_files:
        variant = edit_file.stem
        variant_map: dict[str, str] = {}
        for row in read_jsonl(edit_file):
            q = row.get("question", "")
            if q:
                variant_map[q] = row.get("edited_response", "")
        edits_by_variant[variant] = variant_map

    # One record per question with all variants as fields
    records = []
    for question, original_text in originals.items():
        record: dict[str, str] = {
            "question": question,
            "original": original_text,
        }
        for variant in variant_names:
            record[variant] = edits_by_variant[variant].get(question, "")
        records.append(record)

    write_jsonl(records, compare_path)
    print(f"  Written {len(records)} records to {compare_path}")
    print(f"  Fields: original" + (f", {', '.join(variant_names)}" if variant_names else ""))
    return variant_names


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    original_path = run_dir / "original_responses.jsonl"
    edits_dir = run_dir / "edits"
    compare_path = run_dir / "compare.jsonl"

    run_inference_phase(args, original_path)
    run_editing_phase(args, original_path, edits_dir)
    variant_names = build_compare_jsonl(original_path, edits_dir, compare_path)

    all_fields = ["original"] + variant_names
    fields_str = " ".join(all_fields)
    print(f"\n{'=' * 60}")
    print("DONE — review with the TUI:")
    print(f"  uv run python scripts/jsonl_tui/cli.py {compare_path} \\")
    print(f"      --variant-fields {fields_str}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
