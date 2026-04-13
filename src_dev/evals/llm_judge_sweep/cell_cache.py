"""Per-cell HF hydrate/upload helpers for the llm-judge scale sweep.

The sweep runner is cell-oriented: each cell is atomic, independently
rehydratable, and its artifacts live at a canonical HF path (see
:mod:`cell_identity`). This module mediates IO between the local scratch
copy of a cell and its HF home.

Shape of a cell directory (both local and on HF)::

    <cell_dir>/
      rollouts/
        rollouts.jsonl
        rollout_info.json
        experiment_metadata.json
      judge_runs/
        {rater_id}/
          {metric_name}.jsonl
      cell_info.json
      manifest.json

Everything under ``judge_runs/`` is per-(rater, metric) — a cell can be
hydrated with rollouts but missing a judge metric, in which case the runner
just re-runs the missing judge against the cached rollouts. The hydration
status dataclass surfaces exactly which artifacts are present locally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from src_dev.evals.llm_judge_sweep.cell_identity import CanonicalCell
from src_dev.utils.hf_hub import (
    check_exists_in_dataset_repo,
    download_path_to_dir,
    upload_folder_to_dataset_repo,
)

ROLLOUTS_RELPATH = "rollouts/rollouts.jsonl"
ROLLOUT_INFO_RELPATH = "rollouts/rollout_info.json"
CELL_INFO_RELPATH = "cell_info.json"


@dataclass
class CellArtifactStatus:
    """What's materialised at a cell's local dir right now."""

    has_rollouts: bool = False
    present_judge_metrics: set[tuple[str, str]] = field(default_factory=set)

    def missing_judge_metrics(
        self, required: Iterable[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        return [rm for rm in required if rm not in self.present_judge_metrics]

    @property
    def fully_materialised(self) -> bool:
        """True when rollouts exist and at least one judge output exists.

        Whether *all* required judge outputs exist is context-dependent — the
        caller checks :meth:`missing_judge_metrics` with its own requirements.
        """
        return self.has_rollouts and bool(self.present_judge_metrics)


def cell_status_on_disk(
    cell_dir: Path,
    *,
    required_judge_metrics: Sequence[tuple[str, str]] = (),
) -> CellArtifactStatus:
    """Inspect a local cell dir and report what's present.

    ``required_judge_metrics`` controls which ``(rater_id, metric_name)``
    pairs are *checked*. Extra files on disk outside this set are ignored;
    missing ones just don't appear in the returned status.
    """
    status = CellArtifactStatus()
    status.has_rollouts = (cell_dir / ROLLOUTS_RELPATH).exists()
    for rater_id, metric in required_judge_metrics:
        if (cell_dir / "judge_runs" / rater_id / f"{metric}.jsonl").exists():
            status.present_judge_metrics.add((rater_id, metric))
    return status


def hydrate_cell(
    cell: CanonicalCell,
    *,
    scratch_root: Path,
    model_slug: str,
    eval_name: str,
    fingerprint: str,
    repo_id: str,
    required_judge_metrics: Sequence[tuple[str, str]] = (),
    skip_download: bool = False,
) -> tuple[Path, CellArtifactStatus]:
    """Pull any of the cell's artifacts that exist on HF into local scratch.

    Returns ``(local_dir, status)``. ``status`` reflects what's locally
    present *after* the download attempt — so the runner can decide what
    still needs to be computed. It's safe to call on cells that aren't yet
    on HF: the existence check short-circuits and no error is raised.

    Why do the existence check first: ``download_path_to_dir`` under a
    missing prefix can raise or silently succeed with zero files depending
    on the HF client version; the explicit check keeps behaviour
    predictable.

    Args:
        skip_download: When True, only inspect local disk (use for
            ``--no-upload``-style flows where HF is disabled).
    """
    local_dir = cell.local_dir(
        scratch_root=scratch_root,
        model_slug=model_slug,
        eval_name=eval_name,
        fingerprint=fingerprint,
    )
    local_dir.mkdir(parents=True, exist_ok=True)

    if skip_download:
        return local_dir, cell_status_on_disk(
            local_dir, required_judge_metrics=required_judge_metrics
        )

    hf_dir = cell.hf_dir(model_slug, eval_name, fingerprint)
    if check_exists_in_dataset_repo(repo_id=repo_id, path_in_repo=hf_dir):
        download_path_to_dir(
            repo_id=repo_id,
            path_in_repo=hf_dir,
            target_dir=local_dir,
        )

    return local_dir, cell_status_on_disk(
        local_dir, required_judge_metrics=required_judge_metrics
    )


def upload_cell(
    cell: CanonicalCell,
    *,
    local_dir: Path,
    model_slug: str,
    eval_name: str,
    fingerprint: str,
    repo_id: str,
    commit_message: str,
    allow_patterns: list[str] | None = None,
) -> str:
    """Upload the cell's local dir to its canonical HF path.

    ``allow_patterns`` (relative to ``local_dir``) narrows the upload to
    specific files — useful when only new judge outputs need pushing and
    rollouts were hydrated from HF (no need to re-upload).
    """
    hf_dir = cell.hf_dir(model_slug, eval_name, fingerprint)
    return upload_folder_to_dataset_repo(
        local_dir=local_dir,
        repo_id=repo_id,
        path_in_repo=hf_dir,
        commit_message=commit_message,
        allow_patterns=allow_patterns,
    )
