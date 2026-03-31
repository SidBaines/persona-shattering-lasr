"""Upload an OCT run directory to a custom path in a HF dataset repo.

This is intentionally simple: it mirrors a local run directory as-is so the
remote copy stays self-contained and matches the on-disk layout exactly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src_dev.utils.hf_hub import (
    login_from_env,
    upload_folder_to_dataset_repo,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Local OCT run directory to upload.",
    )
    parser.add_argument(
        "--path-in-repo",
        required=True,
        help="Destination subdirectory inside the dataset repo.",
    )
    parser.add_argument(
        "--repo-id",
        default="persona-shattering-lasr/monorepo",
        help="HF dataset repo id.",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Optional explicit commit message.",
    )
    parser.add_argument(
        "--include-distilled-model",
        action="store_true",
        help="Upload models/distilled/ as well. By default it is excluded.",
    )
    return parser.parse_args()


def main() -> None:
    """Upload the run directory to the requested HF dataset path."""
    args = parse_args()
    login_from_env()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    commit_message = args.commit_message or f"Upload OCT run {run_dir.name}"
    ignore_patterns = None
    if not args.include_distilled_model:
        ignore_patterns = [
            "models/distilled",
            "models/distilled/**",
        ]
    url = upload_folder_to_dataset_repo(
        local_dir=run_dir,
        repo_id=args.repo_id,
        path_in_repo=args.path_in_repo,
        commit_message=commit_message,
        ignore_patterns=ignore_patterns,
    )
    print(url)
    print(f"Uploaded {run_dir} -> {args.repo_id}/{args.path_in_repo}")


if __name__ == "__main__":
    main()
