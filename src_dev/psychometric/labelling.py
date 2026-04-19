"""Factor-label cache IO + (later) LLM labelling entry points.

This module currently exposes the label-cache loading helpers used by the
factor-extremes HTML exporter. The LLM labelling functions
(``label_factors_llm``, ``label_factors_claude_cli``, prompt builders, etc.)
will be extracted here in a later refactor pass and coexist with them.

Cache layout (set by the labelling stage):
    {questionnaire_dir}/labeling/
        item_labels_{label}.json               — loading-based summary (cheap)
        llm_labels_{label}_{timestamp}.json    — LLM-generated (full); newest wins
        llm_labels_{label}.json                — legacy single-file location (pre-cache-rotation)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_llm_labels_from_path(path: Path) -> list[dict]:
    """Load a label cache file, treating empty/invalid payloads as unavailable."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.warning("Ignoring malformed LLM label cache at %s: expected list", path)
        return []

    labels = [entry for entry in data if isinstance(entry, dict)]
    if not labels:
        return []

    return labels


def llm_labels_have_axis_names(labels: list[dict]) -> bool:
    """Return True when every factor label includes a non-empty axis name."""
    if not labels:
        return False
    return all(str(label.get("axis_name", "")).strip() for label in labels)


def load_latest_nonempty_llm_labels(
    labeling_dir: Path,
    label: str,
    *,
    require_axis_names: bool = False,
) -> list[dict]:
    """Return the newest non-empty LLM label cache for an analysis label.

    Searches ``llm_labels_{label}_*.json`` plus the legacy
    ``llm_labels_{label}.json`` location. Files without ``axis_name`` fields
    are skipped when ``require_axis_names=True`` (used by the factor-extremes
    HTML exporter, which needs axis names for its factor chips).
    """
    candidate_paths = set(labeling_dir.glob(f"llm_labels_{label}_*.json"))
    legacy_path = labeling_dir / f"llm_labels_{label}.json"
    if legacy_path.exists():
        candidate_paths.add(legacy_path)

    for path in sorted(candidate_paths, key=lambda p: p.stat().st_mtime, reverse=True):
        labels = load_llm_labels_from_path(path)
        if require_axis_names and labels and not llm_labels_have_axis_names(labels):
            logger.warning(
                "Ignoring old-schema LLM label cache without axis_name fields: %s",
                path,
            )
            continue
        if labels:
            return labels

    return []
