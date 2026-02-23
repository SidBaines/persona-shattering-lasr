"""Helpers for uploading artifacts to Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, login


def login_from_env(token_env: str = "HF_TOKEN") -> None:
    """Authenticate to Hugging Face Hub using a token from env vars."""
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(
            f"Missing {token_env}. Set it in your environment to upload to Hugging Face Hub."
        )
    login(token=token, add_to_git_credential=False)


def upload_file_to_dataset_repo(
    *,
    local_path: Path,
    repo_id: str,
    path_in_repo: str,
    commit_message: str,
) -> str:
    """Upload a local file to a dataset repo on Hugging Face Hub."""
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def upload_folder_to_model_repo(
    *,
    local_dir: Path,
    repo_id: str,
    path_in_repo: str,
    commit_message: str,
) -> str:
    """Upload a local folder (e.g. LoRA adapter) to a model repo on Hugging Face Hub."""
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"
