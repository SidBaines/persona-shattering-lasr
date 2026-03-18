#!/usr/bin/env python3
"""Import a canonical response run from another dataset repo into unsupervised-runs."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from scripts.datasets import validate_run
from scripts.unsupervised_runs import (
    DEFAULT_UNSUPERVISED_HF_REPO_ID,
    response_run_dir,
    upload_response_run,
)
from scripts.unsupervised_runs.io import hydrate_dataset_subtree


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import a canonical response run from a source HF repo.")
    parser.add_argument("--response-run-id", type=str, required=True)
    parser.add_argument("--source-repo-id", type=str, required=True)
    parser.add_argument("--source-path-in-repo", type=str, required=True)
    parser.add_argument("--target-repo-id", type=str, default=DEFAULT_UNSUPERVISED_HF_REPO_ID)
    parser.add_argument("--no-hf-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    load_dotenv()

    run_dir = response_run_dir(args.response_run_id)
    hydrated = hydrate_dataset_subtree(
        repo_id=args.source_repo_id,
        path_in_repo=args.source_path_in_repo,
        local_dir=run_dir,
        required=True,
    )
    if not hydrated:
        raise RuntimeError("Failed to hydrate source run.")

    validate_run(run_dir)
    print(f"Imported canonical run into: {run_dir}")

    if not args.no_hf_upload:
        hf_url = upload_response_run(args.response_run_id, repo_id=args.target_repo_id)
        print(f"Uploaded imported run to: {hf_url}")


if __name__ == "__main__":
    main()
