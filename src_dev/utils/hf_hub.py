"""Helpers for uploading and downloading artifacts to/from Hugging Face Hub."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import requests
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import configure_http_backend
import huggingface_hub.utils._http as _hf_http

# ---------------------------------------------------------------------------
# Shim: huggingface_hub 0.36.x passes allow_redirects= to its HTTP session,
# but httpx (>= 0.20) renamed that param to follow_redirects.  Patch
# http_backoff to translate the kwarg before it hits session.request().
# ---------------------------------------------------------------------------
_orig_http_backoff = _hf_http.http_backoff


def _patched_http_backoff(method, url, *, max_retries=5, base_wait_time=1,
                          max_wait_time=8, retry_on_exceptions=None,
                          retry_on_status_codes=(500, 502, 503, 504),
                          **kwargs):
    """Wrap http_backoff to translate allow_redirects→follow_redirects for httpx sessions."""
    import requests as _requests
    session = _hf_http.get_session()
    if not isinstance(session, _requests.Session):
        _params = inspect.signature(session.request).parameters
        if "allow_redirects" in kwargs and "allow_redirects" not in _params:
            if "follow_redirects" in _params:
                kwargs["follow_redirects"] = kwargs.pop("allow_redirects")
            else:
                kwargs.pop("allow_redirects")
        # httpx defaults follow_redirects=False; match requests' default of True.
        if "follow_redirects" in _params and "follow_redirects" not in kwargs:
            kwargs["follow_redirects"] = True
        # httpx does not accept proxies per-request; drop it silently.
        if "proxies" in kwargs and "proxies" not in _params:
            kwargs.pop("proxies")
    _kwargs = dict(
        max_retries=max_retries,
        base_wait_time=base_wait_time,
        max_wait_time=max_wait_time,
        retry_on_status_codes=retry_on_status_codes,
    )
    if retry_on_exceptions is not None:
        _kwargs["retry_on_exceptions"] = retry_on_exceptions
    return _orig_http_backoff(method, url, **_kwargs, **kwargs)


_hf_http.http_backoff = _patched_http_backoff
# file_download.py uses a direct `from .utils._http import http_backoff` binding,
# so we must also patch it there.
import huggingface_hub.file_download as _hf_file_download
_hf_file_download.http_backoff = _patched_http_backoff

# Extended timeouts (seconds) to avoid ReadTimeout on slow connections during the
# final commit step, which can block for a long time on large uploads.
_TIMEOUT = 300
_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 300


def _backend_factory() -> requests.Session:
    session = requests.Session()
    _original_request = session.request

    def _request_with_timeout(method: str, url: str, **kwargs):  # type: ignore[override]
        kwargs.setdefault("timeout", _TIMEOUT)
        return _original_request(method, url, **kwargs)

    session.request = _request_with_timeout  # type: ignore[method-assign]
    return session


def _configure_timeout() -> None:
    """Install an extended-timeout requests session for huggingface_hub."""

    def _backend_factory() -> requests.Session:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        # Patch send to always use our timeouts (huggingface_hub respects the session)
        _orig_send = session.send

        def _send_with_timeout(request, **kwargs):
            kwargs.setdefault("timeout", (_CONNECT_TIMEOUT, _READ_TIMEOUT))
            return _orig_send(request, **kwargs)

        session.send = _send_with_timeout  # type: ignore[method-assign]
        return session

    configure_http_backend(backend_factory=_backend_factory)


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


def check_exists_in_dataset_repo(
    *,
    repo_id: str,
    path_in_repo: str,
) -> bool:
    """Check if a path exists in a HF dataset repo without downloading.

    Args:
        repo_id: HuggingFace dataset repo ID (``org/name``).
        path_in_repo: Path within the repo to check for.

    Returns:
        True if any files exist under that path, False otherwise.
    """
    try:
        _configure_timeout()
        api = HfApi(token=_get_token())
        files = list(api.list_repo_tree(
            repo_id=repo_id, repo_type="dataset", path_in_repo=path_in_repo,
        ))
        return len(files) > 0
    except Exception:
        return False


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

    # Always scope to path_in_repo to avoid resolving the entire repo.
    # If caller passes specific patterns they are prefixed; otherwise we use
    # a wildcard scoped to the subfolder.
    if allow_patterns is not None:
        prefixed: list[str] = [f"{path_in_repo}/{p}" for p in allow_patterns]
    else:
        prefixed = [f"{path_in_repo}/**", f"{path_in_repo}/*"]

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=prefixed,
        token=token,
    )
    return local_dir
