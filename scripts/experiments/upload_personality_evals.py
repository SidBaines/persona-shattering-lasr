"""Upload personality eval results (logs + figures) to HuggingFace Hub.

Canonical dataset repo: ``persona-shattering-lasr/personality_evals``

Expected local layout (produced by the personality eval suite):

    scratch/evals/personality/{run_name}/
    ├── suite_config.json
    ├── {model_name}/{eval_name}/run_info.json
    ├── {model_name}/{eval_name}/native/inspect_logs/*.json
    └── figures/
        ├── bfi_sweep_lines.png
        └── ...

After upload, the HF repo contains:

    eval_logs/{run_name}/          ← full run directory
    figures/{run_name}/            ← figures only

Usage:

    uv run python -m scripts.experiments.upload_personality_evals \\
        --run-dir scratch/evals/personality/20240101_120000_sarcasm

    # Upload to a different repo
    uv run python -m scripts.experiments.upload_personality_evals \\
        --run-dir scratch/evals/personality/20240101_120000_sarcasm \\
        --repo-id my-org/my-evals-repo

    # Dry-run: print what would be uploaded without actually uploading
    uv run python -m scripts.experiments.upload_personality_evals \\
        --run-dir scratch/evals/personality/20240101_120000_sarcasm \\
        --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DATASET_REPO_ID = "persona-shattering-lasr/personality_evals"


def upload_run(
    run_dir: Path,
    repo_id: str,
    dry_run: bool = False,
) -> None:
    """Upload a single eval run directory to HuggingFace Hub.

    Args:
        run_dir: Local run directory (contains suite_config.json, model dirs, figures/).
        repo_id: HuggingFace dataset repo ID.
        dry_run: If True, print what would be uploaded without uploading.
    """
    from scripts.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo

    run_name = run_dir.name
    figures_dir = run_dir / "figures"

    if not run_dir.exists():
        print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Run:        {run_name}")
    print(f"Local dir:  {run_dir}")
    print(f"Repo:       {repo_id}")
    print()

    if dry_run:
        print("[dry-run] Would upload:")
        print(f"  {run_dir}/ → eval_logs/{run_name}/")
        if figures_dir.exists():
            print(f"  {figures_dir}/ → figures/{run_name}/")
        print("\nNo files were uploaded (--dry-run).")
        return

    login_from_env()

    print(f"Uploading eval logs → eval_logs/{run_name}/ ...")
    url = upload_folder_to_dataset_repo(
        local_dir=run_dir,
        repo_id=repo_id,
        path_in_repo=f"eval_logs/{run_name}",
        commit_message=f"Upload eval logs: {run_name}",
        # Exclude figures from the logs upload; they get their own path below.
        ignore_patterns=["figures/**"],
    )
    print(f"  ✓ {url}/tree/main/eval_logs/{run_name}")

    if figures_dir.exists():
        print(f"Uploading figures → figures/{run_name}/ ...")
        upload_folder_to_dataset_repo(
            local_dir=figures_dir,
            repo_id=repo_id,
            path_in_repo=f"figures/{run_name}",
            commit_message=f"Upload figures: {run_name}",
        )
        print(f"  ✓ {url}/tree/main/figures/{run_name}")
    else:
        print(f"  (no figures directory found at {figures_dir}, skipping)")

    print(f"\n✅ Upload complete → {url}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload personality eval results to HuggingFace Hub"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Local eval run directory (e.g. scratch/evals/personality/20240101_sarcasm)",
    )
    parser.add_argument(
        "--repo-id",
        default=DATASET_REPO_ID,
        help=f"HuggingFace dataset repo ID (default: {DATASET_REPO_ID})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without actually uploading",
    )
    args = parser.parse_args()

    upload_run(
        run_dir=args.run_dir.resolve(),
        repo_id=args.repo_id,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
