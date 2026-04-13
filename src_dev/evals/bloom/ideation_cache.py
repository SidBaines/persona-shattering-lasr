"""Trait-scoped, manually-versioned ideation cache for bloom sweeps.

Layout on HF::

    evals/bloom_ideation/{trait}/v{N}/{ideation_fp}/
      understanding.json
      ideation.json
      ideation_meta.json

Unlike the per-cell rollout+judgment cache, ideation is target-agnostic: one
ideation_fp covers every ``(model, adapter, combo)`` eval for that trait +
version. Bumping ``SCENARIO_VERSION`` yields a fresh ``v{N+1}/`` subtree
(old versions remain on HF for comparison) and forces every downstream
rollout cell to be recomputed because ``rollout_cell_fp`` hashes both the
ideation fp AND the version.

Does NOT use :class:`CanonicalCell` — ideation has a different shape
(trait-scoped, version-bucketed) than a LoRA scale cell.
"""

from __future__ import annotations

from pathlib import Path

from src_dev.utils.hf_hub import (
    check_exists_in_dataset_repo,
    download_path_to_dir,
    upload_folder_to_dataset_repo,
)

UNDERSTANDING_FILENAME = "understanding.json"
IDEATION_FILENAME = "ideation.json"
IDEATION_META_FILENAME = "ideation_meta.json"

UPLOAD_ALLOW_PATTERNS = [
    UNDERSTANDING_FILENAME,
    IDEATION_FILENAME,
    IDEATION_META_FILENAME,
]


def ideation_hf_dir(
    *,
    eval_name: str,
    trait: str,
    ideation_version: int,
    ideation_fp: str,
) -> str:
    """Canonical HF path (relative to the repo root) for an ideation cache entry.

    ``eval_name`` is typically ``"bloom"`` — the trait scoping happens inside
    via ``{eval_name}_ideation/{trait}/``. Returned as a forward-slash string.
    """
    return (
        f"evals/{eval_name}_ideation/{trait}/v{int(ideation_version)}/{ideation_fp}"
    )


def ideation_local_dir(
    scratch_root: Path,
    *,
    eval_name: str,
    trait: str,
    ideation_version: int,
    ideation_fp: str,
) -> Path:
    """Canonical local scratch dir mirroring :func:`ideation_hf_dir`."""
    return Path(scratch_root) / ideation_hf_dir(
        eval_name=eval_name,
        trait=trait,
        ideation_version=ideation_version,
        ideation_fp=ideation_fp,
    )


def ideation_complete_on_disk(local_dir: Path) -> bool:
    """True when both understanding and ideation JSONs are present."""
    return (local_dir / UNDERSTANDING_FILENAME).exists() and (
        local_dir / IDEATION_FILENAME
    ).exists()


def hydrate_ideation(
    *,
    scratch_root: Path,
    eval_name: str,
    trait: str,
    ideation_version: int,
    ideation_fp: str,
    repo_id: str,
    skip_download: bool = False,
) -> tuple[Path, bool]:
    """Pull cached understanding + ideation for one trait/version/fp.

    Returns ``(local_dir, complete)`` where ``complete`` is True iff both
    ``understanding.json`` and ``ideation.json`` are present on disk after
    the (optional) download step.
    """
    local_dir = ideation_local_dir(
        scratch_root,
        eval_name=eval_name,
        trait=trait,
        ideation_version=ideation_version,
        ideation_fp=ideation_fp,
    )
    local_dir.mkdir(parents=True, exist_ok=True)

    if skip_download:
        return local_dir, ideation_complete_on_disk(local_dir)

    hf_dir = ideation_hf_dir(
        eval_name=eval_name,
        trait=trait,
        ideation_version=ideation_version,
        ideation_fp=ideation_fp,
    )
    if check_exists_in_dataset_repo(repo_id=repo_id, path_in_repo=hf_dir):
        download_path_to_dir(
            repo_id=repo_id,
            path_in_repo=hf_dir,
            target_dir=local_dir,
        )

    return local_dir, ideation_complete_on_disk(local_dir)


def upload_ideation(
    *,
    local_dir: Path,
    eval_name: str,
    trait: str,
    ideation_version: int,
    ideation_fp: str,
    repo_id: str,
    commit_message: str,
    allow_patterns: list[str] | None = None,
) -> str:
    """Upload the ideation local dir to its canonical HF path."""
    hf_dir = ideation_hf_dir(
        eval_name=eval_name,
        trait=trait,
        ideation_version=ideation_version,
        ideation_fp=ideation_fp,
    )
    return upload_folder_to_dataset_repo(
        local_dir=local_dir,
        repo_id=repo_id,
        path_in_repo=hf_dir,
        commit_message=commit_message,
        allow_patterns=allow_patterns
        if allow_patterns is not None
        else list(UPLOAD_ALLOW_PATTERNS),
    )
