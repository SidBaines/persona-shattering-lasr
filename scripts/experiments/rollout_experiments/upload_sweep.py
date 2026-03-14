"""Upload rollout sweep results to HuggingFace Hub.

Canonical dataset repo: ``persona-shattering-lasr/t_frequency_lora_sweeps``

Expected local layout (produced by the sweep runner):

    scratch/runs/{sweep_id}/{run_name}/
    ├── sweep_config.json
    ├── scale_{s:+.2f}/{condition}/
    │   ├── run_info.json
    │   ├── rollouts.jsonl
    │   ├── rollouts_evaluated.jsonl
    │   └── ...
    └── sweep_plot.png   (optional)

After upload, the HF repo contains:

    runs/{run_name}/     ← full run directory

Usage::

    python -m scripts.experiments.rollout_experiments.upload_sweep \\
        --run-dir scratch/runs/t_frequency_lora_sweep/20260310_184854_t_avoiding

    # Upload to a different repo
    python -m scripts.experiments.rollout_experiments.upload_sweep \\
        --run-dir scratch/runs/t_frequency_lora_sweep/20260310_184854_t_avoiding \\
        --repo-id my-org/my-sweeps-repo

    # Dry-run: print what would be uploaded without actually uploading
    python -m scripts.experiments.rollout_experiments.upload_sweep \\
        --run-dir scratch/runs/t_frequency_lora_sweep/20260310_184854_t_avoiding \\
        --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATASET_REPO_ID = "persona-shattering-lasr/t_frequency_lora_sweeps"

# Files/dirs to exclude from uploads (large intermediate files not needed for analysis).
_IGNORE_PATTERNS = [
    "exports/**",
    "events/**",
    "datasets/**",
    "per_message_metrics.jsonl",
]


def upload_run(
    run_dir: Path,
    repo_id: str,
    dry_run: bool = False,
) -> None:
    """Upload a single sweep run directory to HuggingFace Hub.

    Args:
        run_dir: Local run directory (contains sweep_config.json and scale_* subdirs).
        repo_id: HuggingFace dataset repo ID.
        dry_run: If True, print what would be uploaded without uploading.
    """
    from dotenv import load_dotenv

    from scripts.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo

    load_dotenv()

    run_name = run_dir.name
    path_in_repo = f"runs/{run_name}"

    if not run_dir.exists():
        print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Run:        {run_name}")
    print(f"Local dir:  {run_dir}")
    print(f"Repo:       {repo_id}")
    print(f"Path:       {path_in_repo}")
    print()

    if dry_run:
        print("[dry-run] Would upload:")
        print(f"  {run_dir}/ → {path_in_repo}/")
        print(f"  Ignoring: {_IGNORE_PATTERNS}")
        print("\nNo files were uploaded (--dry-run).")
        return

    login_from_env()

    print(f"Uploading → {path_in_repo}/ ...")
    url = upload_folder_to_dataset_repo(
        local_dir=run_dir,
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        commit_message=f"Upload sweep run: {run_name}",
        ignore_patterns=_IGNORE_PATTERNS,
    )
    print(f"\nUploaded to {url}/tree/main/{path_in_repo}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload rollout sweep results to HuggingFace Hub"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Local sweep run directory (e.g. scratch/runs/t_frequency_lora_sweep/20260310_184854_t_avoiding)",
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
