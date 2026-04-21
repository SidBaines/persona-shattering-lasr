"""Combine per-pair questionnaire outputs into a single multi-preset matrix.

Extracted from the ``psychometric_rollout_fa.py`` orchestrator so
analysis scripts that work on external-rollout runs can reuse the same
row-concatenation + column-union logic without pulling in the full
orchestrator.

Inputs per pair (``(rollout_key, questionnaire_key)``):

    - ``response_matrix``: (n_personas, n_items) float array
    - ``metadata``: list of per-persona dicts (must include ``sample_id``)
    - ``items``: list of per-column dicts (must include ``item_id``,
      optionally ``block``, ``dimension``, etc.)

Columns are grouped by an underlying *version* — presets that share a
version pool into one column block in the output. Row alignment within
a rollout uses sample_id intersection across all its paired
questionnaires (so every surviving row has a response to every version).

The output matches what ``run_factor_analysis`` expects and what the
orchestrator's inline ``_combine_per_pair_outputs`` produced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_pair_outputs(
    questionnaire_dir: Path,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Load ``(response_matrix, metadata, items)`` from a per-pair dir.

    Expects the canonical Stage-2 output layout
    ``<questionnaire_dir>/questionnaire/{response_matrix.npy,
    metadata.jsonl, items.json}``.
    """
    q_dir = questionnaire_dir / "questionnaire"
    matrix_path = q_dir / "response_matrix.npy"
    meta_path = q_dir / "metadata.jsonl"
    items_path = q_dir / "items.json"
    for p in (matrix_path, meta_path, items_path):
        if not p.exists():
            raise FileNotFoundError(
                f"[Combine] Missing {p}. Run Stage 1+2 for this pair first."
            )
    matrix = np.load(matrix_path)
    with meta_path.open("r", encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f if line.strip()]
    with items_path.open("r", encoding="utf-8") as f:
        items = json.load(f)
    return matrix, metadata, items


def combine_per_pair_outputs(
    pair_data: dict[tuple[str, str], tuple[np.ndarray, list[dict], list[dict]]],
    pair_version: dict[tuple[str, str], str],
    *,
    out_dir: Path | None = None,
    provenance_extra: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Combine per-pair response matrices into one multi-preset matrix.

    Args:
        pair_data: Maps ``(rollout_key, q_key)`` → ``(matrix, metadata, items)``.
        pair_version: Maps ``(rollout_key, q_key)`` → questionnaire version
            string. Presets sharing a version pool into one column block.
        out_dir: If given, writes ``response_matrix.npy``, ``metadata.jsonl``,
            ``items.json``, and ``provenance.json`` under ``out_dir``.
            Intended to be the combined-run's ``questionnaire/`` dir so
            downstream FA stages find it at the usual path.
        provenance_extra: Optional dict merged into ``provenance.json``
            (e.g. ``rollout_run_ids`` per preset).

    Returns:
        ``(combined_matrix, combined_metadata, combined_items)``.
        Combined metadata rows carry ``rollout_preset_key`` and
        ``version_sources`` (preset → q_key map) fields.
    """
    rollout_keys: list[str] = []
    seen_r: set[str] = set()
    for r, _ in pair_data:
        if r not in seen_r:
            seen_r.add(r)
            rollout_keys.append(r)

    versions: list[str] = []
    seen_v: set[str] = set()
    for p in pair_data:
        v = pair_version[p]
        if v not in seen_v:
            seen_v.add(v)
            versions.append(v)

    # Each (rollout, version) must have exactly one pair.
    pair_for: dict[tuple[str, str], str] = {}
    for (r_key, q_key), v in pair_version.items():
        key = (r_key, v)
        if key in pair_for:
            raise RuntimeError(
                f"[Combine] rollout {r_key!r} has multiple pairs for version "
                f"{v!r}: {pair_for[key]!r} and {q_key!r}."
            )
        pair_for[key] = q_key
    for r_key in rollout_keys:
        for v in versions:
            if (r_key, v) not in pair_for:
                raise RuntimeError(
                    f"[Combine] rollout {r_key!r} has no pair for version "
                    f"{v!r}. Every rollout must cover every selected version."
                )

    # Intersect sample_ids across each rollout's paired questionnaires.
    per_rollout_sids: dict[str, list[str]] = {}
    for r_key in rollout_keys:
        rollout_q_keys = [pair_for[(r_key, v)] for v in versions]
        sids_per_q: list[set[str]] = []
        for q_key in rollout_q_keys:
            _, meta, _ = pair_data[(r_key, q_key)]
            sids_per_q.append({m["sample_id"] for m in meta if m.get("sample_id")})
        common = set.intersection(*sids_per_q) if sids_per_q else set()
        _, first_meta, _ = pair_data[(r_key, rollout_q_keys[0])]
        ordered = [m["sample_id"] for m in first_meta if m.get("sample_id") in common]
        if not ordered:
            raise RuntimeError(
                f"[Combine] No shared sample_ids for rollout {r_key!r} across "
                f"questionnaires {rollout_q_keys!r}."
            )
        per_rollout_sids[r_key] = ordered
        n_dropped = len(first_meta) - len(ordered)
        if n_dropped:
            print(
                f"[Combine] rollout={r_key!r}: kept {len(ordered)} rows, "
                f"dropped {n_dropped} without responses in every paired questionnaire"
            )

    # Build per-version namespaced column blocks.
    q_items_combined: list[dict] = []
    v_col_counts: dict[str, int] = {}
    for v in versions:
        source_r = rollout_keys[0]
        source_q = pair_for[(source_r, v)]
        _, _, items = pair_data[(source_r, source_q)]
        v_col_counts[v] = len(items)
        src_item_ids = [it.get("item_id", it.get("id")) for it in items]
        for r_key in rollout_keys[1:]:
            _, _, other_items = pair_data[(r_key, pair_for[(r_key, v)])]
            other_ids = [it.get("item_id", it.get("id")) for it in other_items]
            if other_ids != src_item_ids:
                raise RuntimeError(
                    f"[Combine] item-order mismatch for version {v!r} between "
                    f"rollouts {source_r!r} and {r_key!r}."
                )
        for it in items:
            namespaced = dict(it)
            orig_item_id = it.get("item_id", it.get("id"))
            orig_col_id = it.get("col_id", orig_item_id)
            namespaced["item_id"] = f"{v}/{orig_item_id}"
            namespaced["col_id"] = f"{v}/{orig_col_id}"
            if "id" in it:
                namespaced["id"] = f"{v}/{it['id']}"
            namespaced["questionnaire_version"] = v
            q_items_combined.append(namespaced)

    # Assemble the combined matrix block-by-block.
    n_total_rows = sum(len(per_rollout_sids[r]) for r in rollout_keys)
    n_total_cols = sum(v_col_counts[v] for v in versions)
    combined = np.full((n_total_rows, n_total_cols), np.nan, dtype=float)

    row_offset = 0
    combined_metadata: list[dict] = []
    for r_key in rollout_keys:
        sids = per_rollout_sids[r_key]
        sid_to_row = {sid: i for i, sid in enumerate(sids)}
        first_v = versions[0]
        _, base_meta, _ = pair_data[(r_key, pair_for[(r_key, first_v)])]
        base_by_sid = {m["sample_id"]: m for m in base_meta if m.get("sample_id")}
        for sid in sids:
            row = dict(base_by_sid[sid])
            row["rollout_preset_key"] = r_key
            row["version_sources"] = {v: pair_for[(r_key, v)] for v in versions}
            combined_metadata.append(row)

        col_offset = 0
        for v in versions:
            q_key = pair_for[(r_key, v)]
            matrix, meta, _ = pair_data[(r_key, q_key)]
            n_cols = v_col_counts[v]
            if matrix.shape[1] != n_cols:
                raise RuntimeError(
                    f"[Combine] column count mismatch for version {v!r} in "
                    f"rollout {r_key!r} / preset {q_key!r}."
                )
            sid_to_src_row = {
                m["sample_id"]: i for i, m in enumerate(meta) if m.get("sample_id")
            }
            for sid, dst_idx in sid_to_row.items():
                src_idx = sid_to_src_row.get(sid)
                if src_idx is None:
                    raise RuntimeError(
                        f"[Combine] sample_id {sid!r} missing from "
                        f"(rollout={r_key!r}, preset={q_key!r}) — intersection bug?"
                    )
                combined[row_offset + dst_idx, col_offset:col_offset + n_cols] = matrix[src_idx]
            col_offset += n_cols
        row_offset += len(sids)

    n_nan = int(np.isnan(combined).sum())
    if n_nan:
        frac = n_nan / combined.size
        print(
            f"[Combine] {n_nan} NaN cells in combined matrix "
            f"({frac:.2%} of {combined.size} — will be imputed in Stage 3)"
        )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "response_matrix.npy", combined)
        with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
            for row in combined_metadata:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (out_dir / "items.json").open("w", encoding="utf-8") as f:
            json.dump(q_items_combined, f, ensure_ascii=False, indent=2)

        provenance: dict[str, Any] = {
            "rollouts": rollout_keys,
            "versions": versions,
            "pairs": [
                {
                    "rollout_preset_key": r_key,
                    "questionnaire_preset_key": q_key,
                    "questionnaire_version": pair_version[(r_key, q_key)],
                }
                for (r_key, q_key) in pair_data.keys()
            ],
        }
        if provenance_extra:
            provenance.update(provenance_extra)
        with (out_dir.parent / "provenance.json").open("w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2, ensure_ascii=False)
        print(
            f"[Combine] Wrote combined ({combined.shape[0]} rows × "
            f"{combined.shape[1]} cols) to {out_dir}"
        )

    return combined, combined_metadata, q_items_combined
