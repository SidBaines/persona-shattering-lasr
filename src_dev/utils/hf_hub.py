"""Helpers for uploading and downloading artifacts to/from Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import requests
from huggingface_hub import HfApi, snapshot_download

try:
    from huggingface_hub.utils import set_client_factory
except ImportError:
    set_client_factory = None

try:
    from huggingface_hub import configure_http_backend
except ImportError:
    configure_http_backend = None

# Extended timeouts (seconds) to avoid ReadTimeout on slow connections during the
# final commit step, which can block for a long time on large uploads.
_TIMEOUT = httpx.Timeout(connect=10, read=300, write=300, pool=10)


def _configure_timeout() -> None:
    """Install an extended-timeout httpx client for huggingface_hub."""
    if set_client_factory is not None:
        set_client_factory(lambda: httpx.Client(timeout=_TIMEOUT))
        return

    if configure_http_backend is not None:
        timeout = _TIMEOUT

        class _TimeoutSession(requests.Session):
            def request(self, method, url, **kwargs):  # type: ignore[override]
                kwargs.setdefault(
                    "timeout",
                    (
                        timeout.connect,
                        timeout.read,
                    ),
                )
                return super().request(method, url, **kwargs)

        configure_http_backend(lambda: _TimeoutSession())


def _get_token(token_env: str = "HF_TOKEN") -> str:
    """Return HF token from env, raising if missing."""
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(
            f"Missing {token_env}. Set it in your environment to upload to Hugging Face Hub."
        )
    return token


def login_from_env(token_env: str = "HF_TOKEN") -> None:
    """Ensure the HF token is available for huggingface_hub API calls.

    Avoids calling ``login()`` (which triggers a ``whoami`` API call and can
    hit HF's strict rate limit on that endpoint).  Instead we set the token
    env var so that ``HfApi()`` without an explicit token picks it up
    automatically from the environment.
    """
    token = _get_token(token_env)
    # huggingface_hub checks HF_TOKEN (and the legacy HUGGING_FACE_HUB_TOKEN)
    # before falling back to the on-disk cache written by login().  Setting it
    # here ensures HfApi() / snapshot_download() work without any network call.
    os.environ.setdefault("HF_TOKEN", token)


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

    _configure_timeout()
    api = HfApi(token=_get_token())
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def upload_folder_to_dataset_repo(
    *,
    local_dir: Path,
    repo_id: str,
    path_in_repo: str,
    commit_message: str,
    ignore_patterns: list[str] | None = None,
    allow_patterns: list[str] | None = None,
    delete_patterns: list[str] | None = None,
) -> str:
    """Upload a local folder to a dataset repo on Hugging Face Hub.

    Args:
        local_dir: Local directory to upload.
        repo_id: HuggingFace dataset repo ID (``org/name``).
        path_in_repo: Destination path within the repo.
        commit_message: Commit message for the upload.
        ignore_patterns: Optional glob patterns to exclude (forwarded to
            ``HfApi.upload_folder``).
        allow_patterns: Optional glob patterns to include (forwarded to
            ``HfApi.upload_folder``). Only matching files are uploaded.
        delete_patterns: Optional glob patterns for files to delete from the
            remote repo in the same commit.  Files that are also being uploaded
            are NOT deleted (the HF API ignores them in that case).

    Returns:
        URL of the uploaded dataset repo.
    """
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    _configure_timeout()
    api = HfApi(token=_get_token())
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        ignore_patterns=ignore_patterns,
        allow_patterns=allow_patterns,
        delete_patterns=delete_patterns,
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

    _configure_timeout()
    api = HfApi(token=_get_token())
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"


def download_from_dataset_repo(
    *,
    repo_id: str,
    path_in_repo: str,
    local_dir: Path,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Download files from a dataset repo on Hugging Face Hub.

    Uses ``snapshot_download`` with ``allow_patterns`` scoped under
    ``path_in_repo`` so only the requested files are fetched.

    Args:
        repo_id: HuggingFace dataset repo ID (``org/name``).
        path_in_repo: Prefix path within the repo to download from.
        local_dir: Local directory to download into. The repo structure
            under ``path_in_repo`` is replicated here.
        allow_patterns: Glob patterns *relative to path_in_repo* for files
            to download.  E.g. ``["rollouts/rollouts.jsonl"]``.
            If ``None``, all files under ``path_in_repo`` are downloaded.

    Returns:
        The local_dir path.
    """
    _configure_timeout()
    token = _get_token()

    # Prefix patterns with the repo-internal path so snapshot_download
    # matches the full repo-relative paths.
    prefixed: list[str] | None = None
    if allow_patterns is not None:
        prefixed = [f"{path_in_repo}/{p}" for p in allow_patterns]

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=prefixed,
        token=token,
    )
    return local_dir
