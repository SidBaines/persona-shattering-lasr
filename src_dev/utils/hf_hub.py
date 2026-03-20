"""Helpers for uploading and retrieving artifacts on Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from huggingface_hub import HfApi, hf_hub_download, login, snapshot_download

try:
    # huggingface_hub >= 0.23
    from huggingface_hub import configure_http_backend as _configure_http_backend  # type: ignore[attr-defined]
    _HAS_CONFIGURE_HTTP = True
except ImportError:
    _HAS_CONFIGURE_HTTP = False
    try:
        from huggingface_hub.utils import set_client_factory as _set_client_factory  # type: ignore[import-untyped]
    except ImportError:
        _set_client_factory = None  # type: ignore[assignment]

# Extended timeouts (seconds) to avoid ReadTimeout on slow connections during the
# final commit step, which can block for a long time on large uploads.
_TIMEOUT = httpx.Timeout(connect=10, read=300, write=300, pool=10)


def _configure_timeout() -> None:
    """Install an extended-timeout httpx client for huggingface_hub."""
    if _HAS_CONFIGURE_HTTP:
        _configure_http_backend(lambda: httpx.Client(timeout=_TIMEOUT))
    elif _set_client_factory is not None:
        _set_client_factory(lambda: httpx.Client(timeout=_TIMEOUT))


def _get_token(token_env: str = "HF_TOKEN") -> str:
    """Return HF token from env, raising if missing."""
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(
            f"Missing {token_env}. Set it in your environment to upload to Hugging Face Hub."
        )
    return token


def login_from_env(token_env: str = "HF_TOKEN") -> None:
    """Authenticate to Hugging Face Hub using a token from env vars."""
    login(token=_get_token(token_env), add_to_git_credential=False)


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
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    except Exception:
        pass  # repo already exists
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
) -> str:
    """Upload a local folder to a dataset repo on Hugging Face Hub.

    Args:
        local_dir: Local directory to upload.
        repo_id: HuggingFace dataset repo ID (``org/name``).
        path_in_repo: Destination path within the repo.
        commit_message: Commit message for the upload.
        ignore_patterns: Optional glob patterns to exclude (forwarded to
            ``HfApi.upload_folder``).

    Returns:
        URL of the uploaded dataset repo.
    """
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    _configure_timeout()
    api = HfApi(token=_get_token())
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    except Exception:
        pass  # repo already exists
    api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        ignore_patterns=ignore_patterns,
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


def dataset_repo_subpath_exists(
    *,
    repo_id: str,
    path_in_repo: str,
) -> bool:
    """Return whether a file or directory path exists in a dataset repo."""
    _configure_timeout()
    api = HfApi(token=_get_token())
    normalized = path_in_repo.strip("/").rstrip("/")
    if not normalized:
        return True
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception:
        return False
    prefix = f"{normalized}/"
    return any(path == normalized or path.startswith(prefix) for path in files)


def download_file_from_dataset_repo(
    *,
    repo_id: str,
    path_in_repo: str,
    local_dir: Path,
) -> Path:
    """Download a single file from a dataset repo into a local directory."""
    _configure_timeout()
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=path_in_repo,
        local_dir=str(local_dir),
        token=_get_token(),
    )
    return Path(downloaded)


def download_dataset_subpath(
    *,
    repo_id: str,
    path_in_repo: str,
    local_dir: Path,
) -> Path:
    """Download one dataset-repo subpath into a local directory."""
    _configure_timeout()
    local_dir.mkdir(parents=True, exist_ok=True)
    normalized = path_in_repo.strip("/").rstrip("/")
    if not normalized:
        raise ValueError("path_in_repo must be non-empty")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[normalized, f"{normalized}/**"],
        local_dir=str(local_dir),
        token=_get_token(),
    )
    return local_dir / normalized
