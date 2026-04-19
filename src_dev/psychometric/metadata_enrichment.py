"""Archetype + scenario-id lookups for response-matrix metadata.

During rollout generation, the seed message is tagged
``[scenario: <id> | archetype: <name>]``. This module re-extracts that tag
from the canonical-samples JSONL so downstream plotting / variance-decomp
validation can annotate each persona row with ``archetype`` and
``scenario_id`` without re-running the rollout pipeline.

A tiny module-level cache avoids re-reading the JSONL when the same lookup is
consumed by multiple stages in one process.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_SCENARIO_ARCHETYPE_PAT = re.compile(
    r"\[scenario:\s*([^|\]]+?)\s*\|\s*archetype:\s*([^\]]+?)\s*\]"
)

_DEFAULT_LOOKUP_CACHE: dict[tuple[str, ...], dict[str, tuple[str, str]]] = {}


def load_archetype_scenario_lookup(
    rollout_dirs: list[Path] | tuple[Path, ...],
    *,
    cache: dict[str, tuple[str, str]] | None = None,
) -> dict[str, tuple[str, str]]:
    """Return ``sample_id -> (archetype, scenario_id)`` from one or more
    rollout canonical datasets.

    In multi-rollout mode, unions the lookups across every supplied rollout
    directory so metadata rows from either source resolve correctly.
    ``sample_id`` values are content-hashed, so collisions across rollout
    sets are possible but extremely unlikely; if any do occur we keep the
    first observation (sample content is identical by construction, so
    either entry is fine).

    Returns ``{}`` if no canonical dataset is found — callers handle the
    empty case.

    If ``cache`` is supplied, entries are merged into it in-place and the
    cache is returned. This lets callers share a cache across stages. If
    ``cache`` is ``None``, a module-level cache keyed on ``rollout_dirs`` is
    used so repeated calls with the same directories don't re-read the
    JSONL.
    """
    if cache is None:
        key = tuple(str(Path(d)) for d in rollout_dirs)
        cached = _DEFAULT_LOOKUP_CACHE.get(key)
        if cached is not None:
            return cached
        lookup: dict[str, tuple[str, str]] = {}
    else:
        key = None
        lookup = cache
    for rollout_dir in rollout_dirs:
        path = Path(rollout_dir) / "datasets" / "canonical_samples.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                sid = rec.get("sample_id")
                messages = rec.get("input", {}).get("messages") or []
                if not sid or not messages:
                    continue
                m = _SCENARIO_ARCHETYPE_PAT.search(messages[0].get("content", "") or "")
                if m and sid not in lookup:
                    lookup[sid] = (m.group(2), m.group(1))
    if key is not None:
        _DEFAULT_LOOKUP_CACHE[key] = lookup
    return lookup


def enrich_meta_with_archetype_scenario(
    meta: list[dict],
    lookup: dict[str, tuple[str, str]],
    *,
    verbose: bool = True,
    context_tag: str = "[Variance decomp]",
) -> list[dict]:
    """Return a shallow copy of ``meta`` with archetype + scenario_id added
    per row where resolvable via sample_id.

    Rows that can't be resolved are left without those keys — downstream
    code (e.g. ``prompt_effects``) warns on missing keys.
    """
    if not lookup:
        if verbose:
            print(
                f"  {context_tag} canonical_samples.jsonl not found — "
                "archetype/scenario_id will be NaN."
            )
        return meta

    n_hit = 0
    enriched: list[dict] = []
    for row in meta:
        sid = row.get("sample_id")
        hit = lookup.get(sid) if sid else None
        if hit is not None:
            arch, sc_id = hit
            enriched.append({**row, "archetype": arch, "scenario_id": sc_id})
            n_hit += 1
        else:
            enriched.append(dict(row))
    if verbose and n_hit < len(meta):
        print(
            f"  {context_tag} enriched {n_hit}/{len(meta)} rows with "
            "archetype/scenario_id (rest lack a resolvable sample_id)."
        )
    return enriched
