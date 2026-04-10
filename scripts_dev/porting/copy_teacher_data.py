"""Copy teacher distillation data from a source model's monorepo path to a target model.

This enables training a new student model (e.g. gemma-3-4b-it) using the same
teacher responses (chosen) that were generated for a larger model's run, avoiding
expensive re-generation of teacher data via OpenRouter.

The pipeline will then detect that teacher responses exist but the target model's
student column is missing, and only run the student (rejected) generation pass.

Usage
-----
    uv run python scripts_dev/porting/copy_teacher_data.py \
        --source-model gemma-3-27b-it \
        --target-model gemma-3-4b-it \
        --trait conscientiousness \
        --direction suppressor \
        --version 2 \
        --constitution conscientiousness_low_v2

    # Dry run (show what would be copied without uploading):
    uv run python scripts_dev/porting/copy_teacher_data.py \
        --source-model gemma-3-27b-it \
        --target-model gemma-3-4b-it \
        --trait conscientiousness \
        --direction suppressor \
        --version 2 \
        --constitution conscientiousness_low_v2 \
        --dry-run
"""

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src_dev.utils.hf_hub import (
    check_exists_in_dataset_repo,
    download_from_dataset_repo,
    upload_file_to_dataset_repo,
)

HF_REPO = "persona-shattering-lasr/monorepo"


def _monorepo_prefix(model: str, trait: str, direction: str, version: str) -> str:
    return f"fine_tuning/{model}/ocean/{trait}/{direction}/v{version}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy teacher distillation data between monorepo model paths.",
    )
    parser.add_argument("--source-model", required=True, help="Source model short name (e.g. gemma-3-27b-it)")
    parser.add_argument("--target-model", required=True, help="Target model short name (e.g. gemma-3-4b-it)")
    parser.add_argument("--trait", required=True, help="OCEAN trait (e.g. conscientiousness)")
    parser.add_argument("--direction", required=True, help="amplifier or suppressor")
    parser.add_argument("--version", required=True, help="Version number (e.g. 2)")
    parser.add_argument("--constitution", required=True, help="Constitution name without .json (e.g. conscientiousness_low_v2)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without uploading")
    args = parser.parse_args()

    src_prefix = _monorepo_prefix(args.source_model, args.trait, args.direction, args.version)
    dst_prefix = _monorepo_prefix(args.target_model, args.trait, args.direction, args.version)
    distillation_rel = f"data/distillation/{args.constitution}.jsonl"

    src_path = f"{src_prefix}/{distillation_rel}"
    dst_path = f"{dst_prefix}/{distillation_rel}"

    print(f"Source: {HF_REPO} :: {src_path}")
    print(f"Target: {HF_REPO} :: {dst_path}")

    # Check if target already exists
    if check_exists_in_dataset_repo(repo_id=HF_REPO, path_in_repo=dst_path):
        print(f"\nTarget already exists on HF — skipping upload.")
        print("If you want to overwrite, delete the target file first.")
        return

    # Download source
    print(f"\nDownloading source distillation data...")
    local_cache = Path(f"scratch/tmp_porting_{args.source_model}_to_{args.target_model}")
    download_from_dataset_repo(
        repo_id=HF_REPO,
        path_in_repo=f"{src_prefix}/data/distillation",
        local_dir=local_cache,
    )
    local_file = local_cache / src_prefix / distillation_rel
    if not local_file.exists():
        raise FileNotFoundError(
            f"Distillation file not found at {local_file}. "
            f"Check that {src_path} exists on the monorepo."
        )

    # Validate
    df = pd.read_json(local_file, orient="records", lines=True)
    assert "response" in df.columns, f"Missing 'response' (teacher) column: {list(df.columns)}"
    assert "prompt" in df.columns, f"Missing 'prompt' column: {list(df.columns)}"
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    if args.target_model in df.columns:
        print(f"\n  Note: target model '{args.target_model}' column already exists in source data.")
        print("  The pipeline will detect this and skip student generation.")

    if args.dry_run:
        print(f"\n[DRY RUN] Would upload {local_file} to {dst_path}")
        return

    # Upload
    print(f"\nUploading to {HF_REPO} :: {dst_path} ...")
    upload_file_to_dataset_repo(
        local_path=local_file,
        repo_id=HF_REPO,
        path_in_repo=dst_path,
        commit_message=(
            f"Port teacher distillation data from {args.source_model} to "
            f"{args.target_model} ({args.trait}/{args.direction}/v{args.version}, "
            f"{len(df)} rows)"
        ),
    )
    print("  Done.")

    # Cleanup
    import shutil
    shutil.rmtree(local_cache, ignore_errors=True)

    print(f"\nTeacher data is now at: {HF_REPO} :: {dst_path}")
    print(
        f"\nNext step — run the pipeline:\n"
        f"  bash scripts_dev/oct_pipeline/run_ocean_persona_e2e.sh \\\n"
        f"      --constitution scripts_dev/oct_pipeline/ocean/{args.constitution}.json \\\n"
        f"      --trait {args.trait} --direction {args.direction} --version {args.version} \\\n"
        f"      --model {args.target_model} \\\n"
        f"      --teacher z-ai/glm-4.5-air \\\n"
        f"      --student-max-num-seqs 256 --student-max-num-batched-tokens 65536"
    )


if __name__ == "__main__":
    main()
