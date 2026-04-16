"""Copy teacher distillation data between monorepo paths.

Supports two porting patterns:

1. **Cross-model** (same version, different student model): reuse teacher
   (chosen) responses to avoid expensive re-generation when training a new
   student model.  The pipeline will detect that the teacher column is present
   but the target model's student column is missing, and only run the student
   (rejected) generation pass.

2. **Cross-version** (same model, different version): reuse the full
   distillation JSONL — both teacher and student responses — when the
   constitutions and models are identical across versions (e.g. vanton4 →
   vanton4_rank1, which differs only in LoRA rank).

Usage
-----
    # Cross-model: copy teacher data from gemma-27b to gemma-4b
    uv run python scripts_dev/porting/copy_teacher_data.py \
        --source-model gemma-3-27b-it \
        --target-model gemma-3-4b-it \
        --trait conscientiousness \
        --direction suppressor \
        --version 2 \
        --constitution conscientiousness_low_v2

    # Cross-version: reuse vanton4 distillation data for vanton4_rank1
    uv run python scripts_dev/porting/copy_teacher_data.py \
        --source-model llama-3.1-8b-it \
        --target-model llama-3.1-8b-it \
        --source-version anton4 \
        --target-version anton4_rank1 \
        --trait conscientiousness \
        --direction suppressor \
        --constitution conscientiousness_suppressing_full_vanton4

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
        description="Copy teacher distillation data between monorepo paths.",
    )
    parser.add_argument("--source-model", required=True, help="Source model short name (e.g. gemma-3-27b-it)")
    parser.add_argument("--target-model", required=True, help="Target model short name (e.g. gemma-3-4b-it)")
    parser.add_argument("--trait", required=True, help="OCEAN trait (e.g. conscientiousness)")
    parser.add_argument("--direction", required=True, help="amplifier or suppressor")
    # --version sets both source and target version; use --source-version /
    # --target-version to port across versions with the same model.
    parser.add_argument("--version", default=None, help="Version string for both source and target (e.g. 2 or anton4)")
    parser.add_argument("--source-version", default=None, help="Source version (overrides --version for source)")
    parser.add_argument("--target-version", default=None, help="Target version (overrides --version for target)")
    parser.add_argument("--constitution", required=True, help="Constitution name without .json (e.g. conscientiousness_low_v2)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without uploading")
    args = parser.parse_args()

    source_version = args.source_version or args.version
    target_version = args.target_version or args.version
    if source_version is None or target_version is None:
        parser.error("Provide --version (applies to both) or --source-version / --target-version separately.")

    src_prefix = _monorepo_prefix(args.source_model, args.trait, args.direction, source_version)
    dst_prefix = _monorepo_prefix(args.target_model, args.trait, args.direction, target_version)
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
    local_cache = Path(f"scratch/tmp_porting_{args.source_model}_v{source_version}_to_{args.target_model}_v{target_version}")
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
            f"Port distillation data {args.source_model}/v{source_version} → "
            f"{args.target_model}/v{target_version} "
            f"({args.trait}/{args.direction}, {len(df)} rows)"
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
        f"      --trait {args.trait} --direction {args.direction} --version {target_version} \\\n"
        f"      --model {args.target_model} \\\n"
        f"      --teacher z-ai/glm-4.5-air \\\n"
        f"      --student-max-num-seqs 256 --student-max-num-batched-tokens 65536"
    )


if __name__ == "__main__":
    main()
