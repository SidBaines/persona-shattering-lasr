"""Split bloom's judgment.json into per-quality files for the cell-sweep layout.

Bloom writes one ``judgment.json`` per rollout with a ``judgments`` array,
each entry containing multiple numeric quality fields (``behavior_presence``
for the OCEAN / behavior score, plus additional qualities like
``coherence``). The cell-sweep layout puts each quality in its own file
under ``judge_runs/{judge_model}/{quality}.json`` so re-judging with a new
judge (or extending to a new quality) is an additive operation and lets the
runner skip qualities that are already cached.

``behavior_presence`` is stored as the raw 1-9 bloom score — the
``score - 5 → OCEAN value`` offset is applied only by downstream aggregation
code (matching legacy ``_load_judgment_scores`` in ``scripts_dev/evals/bloom/runner.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def split_judgment_into_qualities(
    judgment_json_path: Path,
    *,
    out_dir: Path,
    judge_model: str,
    behavior_name: str,
    ideation_fp: str,
    rollout_cell_fp: str,
    quality_keys: Sequence[str],
    judge_temperature: float | None = None,
) -> list[Path]:
    """Split one bloom judgment.json into per-quality files.

    Args:
        judgment_json_path: Path to bloom's ``judgment.json`` (as produced by
            a ``bloom judgment ...`` invocation).
        out_dir: The cell dir. Output files land at
            ``out_dir/judge_runs/{judge_model}/{quality}.json``.
        judge_model: Identifier of the judge model (used both as the
            subdirectory name and recorded inside each file).
        behavior_name: Trait name (e.g. ``"conscientiousness"``).
        ideation_fp: Fingerprint of the ideation cache entry this judgment
            was computed against.
        rollout_cell_fp: Fingerprint of the rollout-cell this judgment runs
            on top of.
        quality_keys: Quality fields to extract (e.g.
            ``["behavior_presence", "coherence"]``). Unknown keys produce an
            empty per-scenario list.

    Returns:
        The list of per-quality JSON paths that were written.
    """
    data = json.loads(Path(judgment_json_path).read_text())
    judgments: list[dict[str, Any]] = list(data.get("judgments", []))

    judge_dir = out_dir / "judge_runs" / judge_model
    judge_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for quality in quality_keys:
        scores: list[float] = []
        per_scenario: list[dict[str, Any]] = []
        for j in judgments:
            raw = j.get(quality)
            scenario_entry: dict[str, Any] = {
                "score": float(raw) if isinstance(raw, (int, float)) else None,
            }
            for passthrough in ("scenario_id", "scenario_index", "variation_id", "rep"):
                if passthrough in j:
                    scenario_entry[passthrough] = j[passthrough]
            per_scenario.append(scenario_entry)
            if isinstance(raw, (int, float)):
                scores.append(float(raw))

        payload: dict[str, Any] = {
            "judge_model": judge_model,
            "quality": quality,
            "behavior_name": behavior_name,
            "ideation_fp": ideation_fp,
            "rollout_cell_fp": rollout_cell_fp,
            "judge_temperature": judge_temperature,
            "n": len(scores),
            "scores": scores,
            "per_scenario": per_scenario,
        }

        out_path = judge_dir / f"{quality}.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        written.append(out_path)

    return written
