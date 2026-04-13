"""Bloom cell artifact layout + hydrate/upload helpers.

Cell shape (both local and on HF)::

    <cell_dir>/
      rollouts/
        rollout.json
        transcript_*.json
      judge_runs/
        {judge_model}/
          {quality}.json       # per-quality split of bloom's judgment.json
      ideation_ref.json         # {trait, version, ideation_fp, hf_path}
      cell_info.json

``has_rollouts`` is true iff ``rollouts/rollout.json`` is present. A judge
quality pair ``(judge_model, quality)`` counts as present when the
corresponding file exists. Missing pairs can be re-judged against cached
rollouts without re-running rollouts themselves.

Cross-repo IO is delegated to :mod:`src_dev.evals.cell_sweep.cache`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from src_dev.evals.cell_sweep.cache import hydrate_cell_dir, upload_cell_dir
from src_dev.evals.cell_sweep.cell_identity import CanonicalCell

ROLLOUTS_DIR = "rollouts"
ROLLOUT_JSON_RELPATH = f"{ROLLOUTS_DIR}/rollout.json"
TRANSCRIPTS_GLOB = "transcript_*.json"
JUDGE_RUNS_RELDIR = "judge_runs"
IDEATION_REF_RELPATH = "ideation_ref.json"
CELL_INFO_RELPATH = "cell_info.json"

UPLOAD_ALLOW_PATTERNS = [
    "rollouts/**",
    "judge_runs/**",
    "ideation_ref.json",
    "cell_info.json",
]


@dataclass
class CellArtifactStatus:
    """What's materialised at a bloom cell's local dir right now.

    ``has_rollouts`` is True when ``rollouts/rollout.json`` exists (bloom's
    rollout stage marker). ``present_judge_qualities`` is the set of
    ``(judge_model, quality)`` pairs whose per-quality JSON file exists on
    disk — extra files outside the ``required_judge_qualities`` set passed
    to :func:`cell_status_on_disk` are not surfaced.
    """

    has_rollouts: bool = False
    present_judge_qualities: set[tuple[str, str]] = field(default_factory=set)

    def missing_judge_qualities(
        self, required: Iterable[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        return [rq for rq in required if rq not in self.present_judge_qualities]


def _judge_quality_path(cell_dir: Path, judge_model: str, quality: str) -> Path:
    return cell_dir / JUDGE_RUNS_RELDIR / judge_model / f"{quality}.json"


def cell_status_on_disk(
    cell_dir: Path,
    *,
    required_judge_qualities: Sequence[tuple[str, str]] = (),
) -> CellArtifactStatus:
    """Inspect a local bloom cell dir and report which artifacts are present."""
    status = CellArtifactStatus()
    status.has_rollouts = (cell_dir / ROLLOUT_JSON_RELPATH).exists()
    for judge_model, quality in required_judge_qualities:
        if _judge_quality_path(cell_dir, judge_model, quality).exists():
            status.present_judge_qualities.add((judge_model, quality))
    return status


def hydrate_cell(
    cell: CanonicalCell,
    *,
    scratch_root: Path,
    model_slug: str,
    eval_name: str,
    fingerprint: str,
    repo_id: str,
    required_judge_qualities: Sequence[tuple[str, str]] = (),
    skip_download: bool = False,
) -> tuple[Path, CellArtifactStatus]:
    """Pull cell artifacts from HF and return ``(local_dir, status)``."""
    local_dir = hydrate_cell_dir(
        cell,
        scratch_root=scratch_root,
        model_slug=model_slug,
        eval_name=eval_name,
        fingerprint=fingerprint,
        repo_id=repo_id,
        skip_download=skip_download,
    )
    return local_dir, cell_status_on_disk(
        local_dir, required_judge_qualities=required_judge_qualities
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
    """Upload the cell's local dir to its canonical HF path."""
    return upload_cell_dir(
        cell,
        local_dir=local_dir,
        model_slug=model_slug,
        eval_name=eval_name,
        fingerprint=fingerprint,
        repo_id=repo_id,
        commit_message=commit_message,
        allow_patterns=allow_patterns
        if allow_patterns is not None
        else list(UPLOAD_ALLOW_PATTERNS),
    )
