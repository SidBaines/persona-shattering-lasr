"""One-off HF monorepo path migrations.

Two migrations:
  1. c_minus_v2 MMLU results: wrong model-slug path → canonical path
  2. c_minus_v3 adapter + evals: typo 'conscientious/' dir → canonical path

Run once with a valid HF token that has write access to persona-shattering-lasr/monorepo.

Usage:
    HF_TOKEN=<your_token> uv run python -m scripts_dev.porting.migrate_hf_paths
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile

REPO = "persona-shattering-lasr/monorepo"

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN not set")

api = HfApi(token=token)

# Verify auth
try:
    who = api.whoami()
    print(f"Authenticated as: {who['name']}")
except Exception as e:
    raise SystemExit(f"Auth failed: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def copy_tree(src_prefix: str, dst_prefix: str, dry_run: bool = False) -> list[str]:
    """Copy all files under src_prefix to dst_prefix. Returns list of errors."""
    try:
        all_entries = list(api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=src_prefix, recursive=True))
    except Exception as e:
        print(f"  Source not found (already migrated?): {e}")
        return []
    files = [f for f in all_entries if isinstance(f, RepoFile)]
    print(f"  Copying {len(files)} files: {src_prefix!r} → {dst_prefix!r}")
    if dry_run:
        for f in files[:5]:
            print(f"  [dry] {f.path} → {dst_prefix + f.path[len(src_prefix):]}")
        if len(files) > 5:
            print(f"  [dry] ... and {len(files) - 5} more")
        return []

    # Download all files into a temp dir preserving relative structure, then
    # upload the whole folder in one commit (much faster than per-file uploads).
    errors = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        print(f"  Downloading to temp dir...")
        for i, f in enumerate(files):
            rel = f.path[len(src_prefix):].lstrip("/")
            try:
                local = api.hf_hub_download(repo_id=REPO, filename=f.path, repo_type="dataset")
                dest = tmp_path / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(Path(local).read_bytes())
            except Exception as e:
                errors.append((f.path, str(e)))
                print(f"  ERROR downloading {f.path}: {e}")
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(files)} downloaded...")

        if errors:
            print(f"  {len(errors)} download errors — skipping upload")
            return errors

        print(f"  Uploading folder to {dst_prefix!r}...")
        api.upload_folder(
            folder_path=tmp,
            path_in_repo=dst_prefix,
            repo_id=REPO,
            repo_type="dataset",
        )
    print(f"  Upload complete.")
    return errors


def delete_tree(prefix: str, dry_run: bool = False) -> list[str]:
    """Delete all files under prefix. Returns list of errors."""
    try:
        all_entries = list(api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=prefix, recursive=True))
    except Exception as e:
        print(f"  Source not found (already deleted?): {e}")
        return []
    files = [f for f in all_entries if isinstance(f, RepoFile)]
    print(f"  Deleting {len(files)} files under {prefix!r}")
    errors = []
    to_delete = [f.path for f in files]
    if dry_run:
        for p in to_delete[:5]:
            print(f"  [dry] delete {p}")
        if len(to_delete) > 5:
            print(f"  [dry] ... and {len(to_delete) - 5} more")
        return errors
    try:
        api.delete_files(
            repo_id=REPO,
            repo_type="dataset",
            delete_patterns=to_delete,
        )
    except Exception as e:
        # Fall back to individual deletes
        print(f"  Batch delete failed ({e}), falling back to individual deletes...")
        for path in to_delete:
            try:
                api.delete_file(path_in_repo=path, repo_id=REPO, repo_type="dataset")
            except Exception as e2:
                errors.append((path, str(e2)))
    return errors


# ---------------------------------------------------------------------------
# Migration 1: c_minus_v2 MMLU — wrong model slug → canonical
# ---------------------------------------------------------------------------

MMLU_SRC = "fine_tuning/llama-3.1-8B-Instruct/ocean/conscientiousness/evals/mcq/mmlu/c_minus_v2"
MMLU_DST = "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v2/evals/mcq/mmlu/c_minus_v2"

# ---------------------------------------------------------------------------
# Migration 2: c_minus_v3 — typo 'conscientious/' → canonical path
# ---------------------------------------------------------------------------

V3_SRC = "fine_tuning/llama-3.1-8b-it/ocean/conscientious/suppressor-v3-llama-3.1-8b-instruct"
V3_DST = "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v3"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

DRY_RUN = "--dry-run" in sys.argv

if DRY_RUN:
    print("=== DRY RUN (pass no args to execute) ===\n")

print("=== Migration 1: c_minus_v2 MMLU ===")
errs = copy_tree(MMLU_SRC, MMLU_DST, dry_run=DRY_RUN)
if errs:
    print(f"  {len(errs)} copy errors — aborting delete for this migration")
else:
    delete_tree(MMLU_SRC, dry_run=DRY_RUN)
    print("  Migration 1 done.\n")

print("=== Migration 2: c_minus_v3 conscientious→conscientiousness ===")
errs = copy_tree(V3_SRC, V3_DST, dry_run=DRY_RUN)
if errs:
    print(f"  {len(errs)} copy errors — aborting delete for this migration")
else:
    delete_tree(V3_SRC, dry_run=DRY_RUN)
    print("  Migration 2 done.\n")

print("All migrations complete.")
