#!/usr/bin/env python3
"""Helpers for the `label-fa-factors` skill.

Three subcommands:

    resolve  <path>                        → JSON: resolved paths + analysis key, or choices if ambiguous
    describe <path> [--top-n N]            → Markdown per-factor context (loadings + item text)
    write    <path> --labels <labels.json> → Validate labels JSON and write to cache path

`<path>` may be any of:
  * A .npz file produced by `save_factor_analysis`.
  * A rotation save-dir (contains `factor_extremes.html` / `plots/`).
  * A `raw/` / `residualized/` / `per_block/<block>/<resid>/` dir that holds
    one or more `fa_*.npz` files.
  * A higher-level `factor_analysis/` dir — in which case the resolver lists
    every rotation found beneath it and the caller must disambiguate.

The script intentionally re-implements the tiny amount of item-description
logic it needs (mirroring `_describe_column_for_labeller` in the parent
script) so it can run without importing the heavy FA script.

### How rich item context is loaded (important for trait_mcq / fc / fc_pair / vignette)

The ``items.json`` that `psychometric_rollout_fa.py` drops next to the
response matrix is not guaranteed to hold raw questionnaire items — in
multi-pair combine mode it is the **column_defs list**, which has no
``options`` / ``answer_mapping`` / ``stem`` / ``scenario`` / ``option_a`` /
``option_b`` fields. Without those, the labeller would only see the item
stem and be blind to what the actual answer choices mean.

To avoid that, ``describe`` also loads the raw questionnaire JSONs from
``datasets/psychometric_questionnaires/{version}.json`` (found by walking
up from the npz) for every ``questionnaire_version`` referenced in the
column_defs, and merges those rich items into the lookup table.

If a raw source is missing, ``describe`` prints a prominent warning
header listing which versions / fields are unavailable, and each item
that cannot be rendered richly is tagged with a ``⚠ RICH CONTEXT
MISSING`` line naming the fields that could not be recovered. **Never
label a factor from a degraded description without checking those
warnings first** — you might be labelling trait_mcq items with no
visibility into the options.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_LABEL_FIELDS = (
    "factor_index",
    "axis_name",
    "summary",
    "description",
    "positive_pole",
    "negative_pole",
)


def _find_fa_dir(path: Path) -> Path:
    """Walk up until we hit a directory named ``factor_analysis``."""
    cur = path if path.is_dir() else path.parent
    for p in [cur, *cur.parents]:
        if p.name == "factor_analysis":
            return p
    raise SystemExit(
        f"Could not locate a 'factor_analysis' ancestor for {path}."
    )


def _find_questionnaire_dir(path: Path) -> Path:
    """Walk up until we hit a sibling 'questionnaire' dir with items.json."""
    cur = path if path.is_dir() else path.parent
    for p in [cur, *cur.parents]:
        q = p / "questionnaire"
        if q.is_dir() and (q / "items.json").exists():
            return q
    raise SystemExit(
        f"Could not find a sibling 'questionnaire' dir (with items.json) "
        f"walking up from {path}."
    )


def _discover_npz(path: Path) -> list[Path]:
    """Return all fa_*.npz files reachable from ``path``."""
    if path.is_file() and path.suffix == ".npz":
        return [path]
    if not path.is_dir():
        raise SystemExit(f"Not a file or directory: {path}")
    return sorted(path.rglob("fa_*.npz"))


def _derive_key(npz_path: Path, fa_dir: Path) -> str:
    """Reconstruct the analysis key used by the parent script.

    Keys observed in the script:
      * ``{resid}_{rotation}``                         (pool)
      * ``block_{block}_{resid}_{rotation}``           (per-block)
      * ``trait_oriented_{rotation}``                  (trait-oriented pass)
    """
    with open(npz_path.with_suffix(".json")) as f:
        cfg = json.load(f)["config"]
    resid = "residualized" if cfg.get("residualized") else "raw"
    rotation = cfg["rotation"]

    rel = npz_path.relative_to(fa_dir).parts
    if "per_block" in rel:
        block = cfg.get("block") or rel[rel.index("per_block") + 1]
        return f"block_{block}_{resid}_{rotation}"
    # Trait-oriented pass lives in a parallel `factor_analysis_trait_oriented`
    # tree — detect by the parent fa_dir name.
    if fa_dir.name == "factor_analysis_trait_oriented":
        return f"trait_oriented_{rotation}"
    return f"{resid}_{rotation}"


def _save_dir_for(npz_path: Path, key: str, fa_dir: Path) -> Path:
    """Best-effort guess at the HTML/plots save_dir matching this npz."""
    # Per-block layout uses an `_artifacts` subdir sibling to the npz.
    if key.startswith("block_"):
        return npz_path.with_name(npz_path.stem + "_artifacts")
    # Pool layout: top-level `<resid>_<rotation>/` sibling of `<resid>/`.
    return fa_dir / key


def _labeling_dir(questionnaire_dir: Path) -> Path:
    # FA pipeline convention: labeling/ sits as a sibling of questionnaire/
    # under the run root (matches ``cfg.ctx.effective_questionnaire_dir /
    # "labeling"`` in src_dev/psychometric/stages/{labeling,factor_analysis}.py).
    d = questionnaire_dir.parent / "labeling"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_one(npz_path: Path) -> dict:
    fa_dir = _find_fa_dir(npz_path)
    questionnaire_dir = _find_questionnaire_dir(npz_path)
    key = _derive_key(npz_path, fa_dir)
    save_dir = _save_dir_for(npz_path, key, fa_dir)
    labeling_dir = _labeling_dir(questionnaire_dir)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return {
        "npz": str(npz_path),
        "config_json": str(npz_path.with_suffix(".json")),
        "item_labels_json": str(npz_path.parent / f"{npz_path.stem}_item_labels.json"),
        "questionnaire_dir": str(questionnaire_dir),
        "items_json": str(questionnaire_dir / "items.json"),
        "save_dir": str(save_dir),
        "labeling_dir": str(labeling_dir),
        "analysis_key": key,
        "output_path": str(labeling_dir / f"llm_labels_{key}_manual_{ts}.json"),
    }


def cmd_resolve(args: argparse.Namespace) -> None:
    path = Path(args.path).resolve()
    candidates = _discover_npz(path)
    if not candidates:
        raise SystemExit(f"No fa_*.npz files found under {path}.")
    if len(candidates) > 1:
        print(json.dumps({
            "ambiguous": True,
            "message": (
                f"{len(candidates)} rotations found. Pass a specific .npz "
                "or a more specific rotation dir."
            ),
            "choices": [
                {
                    "npz": str(p),
                    "analysis_key": _derive_key(p, _find_fa_dir(p)),
                }
                for p in candidates
            ],
        }, indent=2))
        return
    print(json.dumps(_resolve_one(candidates[0]), indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Rich-item loading
# ─────────────────────────────────────────────────────────────────────────────

# Encodings where the column sign IS trait-interpretable (see
# `src_dev/psychometric/labelling.py` and `questionnaire_io.py` for context).
_TRAIT_INTERPRETABLE_ENCODINGS = {"trait_score_0-1", "trait_aligned_0-1"}


def _find_raw_questionnaires_dir(start: Path) -> Path | None:
    """Walk up from ``start`` until we find a ``datasets/psychometric_questionnaires`` dir."""
    for p in [start, *start.parents]:
        candidate = p / "datasets" / "psychometric_questionnaires"
        if candidate.is_dir():
            return candidate
    return None


def _load_raw_questionnaire_items(path: Path) -> list[dict]:
    """Return raw items from a questionnaire JSON with their rich fields intact.

    Mirrors the subset of `src_dev.psychometric.questionnaire_io.load_questionnaire`
    needed for rich rendering. Preserves:
      - trait_mcq:      options, answer_mapping, question
      - likert:         text, reverse_keyed
      - fc_pair:        stem, options, high_option
      - forced_choice:  option_a, option_b
      - vignette:       scenario, options, title

    Each item retains its raw ``id`` (bare, not version-namespaced) and is
    given a ``type`` field so downstream renderers can branch on it.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    items: list[dict] = []

    for it in data.get("block_4_trait_mcq", {}).get("items", []):
        items.append({
            "id": str(it["id"]),
            "type": "trait_mcq",
            "question": it["question"],
            "text": it["question"],
            "options": it["options"],
            "answer_mapping": it["answer_mapping"],
            "primary_dimension": it.get("primary_dimension"),
        })
    for it in data.get("block_3_likert", {}).get("items", []):
        items.append({
            "id": str(it["id"]),
            "type": "likert",
            "text": it["text"],
            "reverse_keyed": it.get("reverse_keyed", it.get("rev", False)),
            "primary_dimension": it.get("primary_dimension", it.get("dim")),
        })
    for it in data.get("items", []):  # legacy flat Likert (v1/v2)
        items.append({
            "id": str(it["id"]),
            "type": "likert",
            "text": it["text"],
            "reverse_keyed": it.get("reverse_keyed", it.get("rev", False)),
            "primary_dimension": it.get("primary_dimension", it.get("dim")),
        })
    for it in data.get("block_fc_pairs", {}).get("items", []):
        items.append({
            "id": str(it["id"]),
            "type": "fc_pair",
            "stem": it["stem"],
            "options": it["options"],
            "high_option": it["high_option"],
            "axis": it.get("axis"),
        })
    for it in data.get("block_1_forced_choice", {}).get("pairs", []):
        items.append({
            "id": str(it["id"]),
            "type": "forced_choice",
            "option_a": it["option_a"],
            "option_b": it["option_b"],
        })
    for it in data.get("block_2_vignettes", {}).get("scenarios", []):
        items.append({
            "id": str(it["id"]),
            "type": "vignette",
            "title": it.get("title"),
            "scenario": it["scenario"],
            "options": it["options"],
            "primary_dimensions": it.get("primary_dimensions"),
        })
    return items


_RICH_FIELD_NAMES = ("options", "answer_mapping", "stem", "scenario", "option_a", "option_b")


def _build_items_lookup(
    resolved: dict, column_defs: list[dict]
) -> tuple[dict[str, dict], list[str]]:
    """Build ``{id → raw_item}``, preferring rich raw questionnaire sources.

    Returns ``(items_by_id, warnings)``. ``warnings`` is a list of
    human-readable strings explaining any source that was missing or
    skipped — the caller MUST surface these to the labeller so they know
    when rendering is degraded (e.g. trait_mcq items shown with no
    options).

    Lookup order:
      1. ``resolved["items_json"]`` — used only if at least one entry has
         one of the rich fields ``{options, answer_mapping, stem, scenario,
         option_a, option_b}``. In multi-pair FA runs this file is the
         column_defs list (no rich fields) — we fall through to raw JSONs.
      2. For every ``questionnaire_version`` referenced by the
         ``column_defs``, we load
         ``<project_root>/datasets/psychometric_questionnaires/{version}.json``
         and register every item by both its bare id and its
         ``{version}/{id}`` namespace.
    """
    items_by_id: dict[str, dict] = {}
    warnings: list[str] = []

    def _register(item: dict, version: str | None = None) -> None:
        raw_id = str(item.get("id") or item.get("item_id") or item.get("col_id") or "")
        if not raw_id:
            return
        items_by_id.setdefault(raw_id, item)
        bare = raw_id.split("/", 1)[-1]
        items_by_id.setdefault(bare, item)
        if version:
            items_by_id.setdefault(f"{version}/{bare}", item)

    items_path = Path(resolved["items_json"])
    local_items: list[dict] = []
    try:
        with open(items_path, encoding="utf-8") as f:
            local_items = json.load(f)
    except FileNotFoundError:
        warnings.append(f"items.json not found at {items_path}.")

    rich_local = any(
        any(k in it for k in _RICH_FIELD_NAMES) for it in local_items
    )
    if rich_local:
        for it in local_items:
            _register(it)
    elif local_items:
        warnings.append(
            f"items.json at {items_path} contains only column_defs (no "
            f"{_RICH_FIELD_NAMES} fields). Falling back to raw questionnaire "
            "JSONs for rich context."
        )

    # Prefer an explicit ``questionnaire_version`` field on the column_def,
    # falling back to the prefix before the first "/" in col_id / item_id
    # (which is how multi-pair combine mode namespaces columns: see
    # `psychometric_rollout_fa.py` lines ~1572-1581). The _item_labels.json
    # sidecar next to the npz strips the version field, so without this
    # fallback we would never discover which raw questionnaires to load.
    def _version_of(cd: dict) -> str | None:
        v = cd.get("questionnaire_version")
        if v:
            return str(v)
        for k in ("col_id", "item_id"):
            raw = cd.get(k)
            if raw and "/" in str(raw):
                return str(raw).split("/", 1)[0]
        return None

    needed_versions = sorted({v for cd in column_defs if (v := _version_of(cd))})

    raw_dir = _find_raw_questionnaires_dir(Path(resolved["npz"]))
    if raw_dir is None:
        if needed_versions and not rich_local:
            warnings.append(
                "Could not locate 'datasets/psychometric_questionnaires/' by "
                f"walking up from {resolved['npz']}. Items from versions "
                f"{needed_versions} will render without options / "
                "answer_mapping / stem / scenario — describe output will tag "
                "each such item with a ⚠ RICH CONTEXT MISSING line."
            )
    else:
        missing_versions: list[str] = []
        for v in needed_versions:
            # Two naming conventions are used under
            # ``datasets/psychometric_questionnaires``:
            #   - ``{version}.json`` (trait_ocean_v1.json, …)
            #   - ``psychometric_questionnaire_{version}.json`` (v5.json → …_v5.json)
            candidates = [
                raw_dir / f"{v}.json",
                raw_dir / f"psychometric_questionnaire_{v}.json",
            ]
            raw_path = next((p for p in candidates if p.exists()), None)
            if raw_path is None:
                missing_versions.append(v)
                continue
            try:
                raw_items = _load_raw_questionnaire_items(raw_path)
            except (KeyError, json.JSONDecodeError) as e:
                warnings.append(
                    f"Could not parse raw questionnaire {raw_path}: {e}. "
                    f"Items from version {v!r} will render without options / "
                    "answer_mapping / stem / scenario."
                )
                continue
            for it in raw_items:
                _register(it, version=v)
        if missing_versions:
            warnings.append(
                f"Raw questionnaire JSON missing under {raw_dir} for versions "
                f"{missing_versions}. Items from those versions will render "
                "without options / answer_mapping / stem / scenario."
            )

    return items_by_id, warnings


# ─────────────────────────────────────────────────────────────────────────────
# Per-factor description
# ─────────────────────────────────────────────────────────────────────────────


def _missing_rich_note(missing_fields: str) -> str:
    """A prominent one-line warning placed inside a degraded item description."""
    return (
        f"  ⚠ RICH CONTEXT MISSING: raw item not found (or lacks "
        f"{missing_fields}). Only the stem is shown — you cannot see "
        f"the actual answer choices for this item."
    )


def _lookup_item(col_def: dict, items_by_id: dict[str, dict]) -> dict | None:
    """Look up a column's raw item across the possible id conventions."""
    candidates = [
        col_def.get("item_id"),
        col_def.get("col_id"),
    ]
    for c in candidates:
        if not c:
            continue
        if c in items_by_id:
            return items_by_id[c]
        bare = str(c).split("/", 1)[-1]
        if bare in items_by_id:
            return items_by_id[bare]
    return None


def _describe_item(col_def: dict, loading: float, items_by_id: dict[str, dict]) -> str:
    """Build a block-aware description for one column's top-loading item.

    For non-Likert blocks this REQUIRES ``items_by_id`` to contain the raw
    questionnaire item (with fields like ``options``, ``answer_mapping``,
    ``stem``, ``scenario``, ``option_a``/``option_b``). When the raw item
    is missing — usually because the caller only had column_defs and no
    raw questionnaire source — the description degrades to a stem-only
    form with a ``⚠ RICH CONTEXT MISSING`` tag so the labeller can see
    the degradation explicitly rather than silently labelling from the
    question stem alone.

    Callers should build ``items_by_id`` via ``_build_items_lookup`` and
    surface its warnings to the user before emitting descriptions.
    """
    block = col_def["block"]
    sign = "+" if loading > 0 else "−"
    item_id = str(col_def.get("item_id") or col_def.get("col_id") or "?")
    item = _lookup_item(col_def, items_by_id)

    if block == "fc":
        if item and "option_a" in item and "option_b" in item:
            return (
                f'[FC, loading={loading:+.3f}]\n'
                f'  A (+1): "{item["option_a"]["text"]}"\n'
                f'  B (−1): "{item["option_b"]["text"]}"\n'
                f'  → {sign} loading: high-factor personas choose '
                f'{"A" if loading > 0 else "B"}.'
            )
        return (
            f'[FC, loading={loading:+.3f}] item_id={item_id}\n'
            f'  Text: "{col_def.get("text", "?")}"\n'
            + _missing_rich_note("option_a/option_b")
        )

    if block == "fc_pair":
        plus_letter = col_def.get("high_option", "A")
        minus_letter = "B" if plus_letter == "A" else "A"
        if item and "stem" in item and "options" in item:
            option_by_label = {o["label"]: o["text"] for o in item["options"]}
            picked = plus_letter if loading > 0 else minus_letter
            return (
                f'[fc_pair, loading={loading:+.3f}]\n'
                f'  Stem: "{item["stem"]}"\n'
                f'  A: "{option_by_label.get("A", "?")}"\n'
                f'  B: "{option_by_label.get("B", "?")}"\n'
                f'  (+1 = option {plus_letter})\n'
                f'  → {sign} loading: high-factor personas pick option {picked}.'
            )
        return (
            f'[fc_pair, loading={loading:+.3f}] item_id={item_id}\n'
            f'  Text: "{col_def.get("text", "?")}"\n'
            f'  (+1 = option {plus_letter})\n'
            + _missing_rich_note("stem/options")
        )

    if block == "vignette":
        dim = col_def.get("dimension", "?")
        if item and "scenario" in item and "options" in item:
            lines = [
                f"[Vignette → {dim}, loading={loading:+.3f}]",
                f'  Scenario: "{item["scenario"]}"',
                f"  Options ({dim} score in parens):",
            ]
            for opt in item["options"]:
                score = opt.get("scoring", {}).get(dim, 0)
                lines.append(f'    {opt["label"]} ({dim}={score:+d}): "{opt["text"]}"')
            lines.append(
                f"  → {sign} loading: high-factor personas pick "
                f"{'higher' if loading > 0 else 'lower'}-{dim} options."
            )
            return "\n".join(lines)
        return (
            f'[Vignette → {dim}, loading={loading:+.3f}] item_id={item_id}\n'
            f'  Text: "{col_def.get("text", "?")}"\n'
            + _missing_rich_note("scenario/options")
        )

    if block == "likert":
        reverse = col_def.get("reverse_keyed", False)
        agree_more = (loading > 0) != reverse
        return (
            f'[Likert, loading={loading:+.3f}] "{col_def["text"]}"\n'
            f'  Scale: 1=strongly disagree … 5=strongly agree\n'
            f'  → {sign} loading: high-factor personas '
            f'{"agree more" if agree_more else "disagree more"}.'
        )

    if block == "trait_mcq":
        dim = col_def.get("dimension", "?")
        encoding = col_def.get("encoding", "letter_1-4")
        lines = [
            f"[TRAIT MCQ → {dim}, encoding={encoding}, loading={loading:+.3f}]",
            f'  Question: "{col_def["text"]}"',
        ]
        if item and "options" in item and "answer_mapping" in item:
            answer_mapping = item["answer_mapping"]
            options_raw = item["options"]
            if isinstance(options_raw, list):
                options = {str(o.get("label", "")): str(o.get("text", "")) for o in options_raw}
            else:
                options = {str(k): str(v) for k, v in options_raw.items()}
            high = [f'    {l}: "{options.get(l, "?")}"' for l, v in answer_mapping.items() if int(v) == 1]
            low = [f'    {l}: "{options.get(l, "?")}"' for l, v in answer_mapping.items() if int(v) == 0]
            if high:
                lines += [f"  High-{dim} options:"] + high
            if low:
                lines += [f"  Low-{dim} options:"] + low
        else:
            lines.append(f"  item_id={item_id}")
            lines.append(_missing_rich_note("options/answer_mapping"))
        if encoding in _TRAIT_INTERPRETABLE_ENCODINGS:
            lines.append(
                f"  → {sign} loading: high-factor personas pick "
                f"{'more high-' if loading > 0 else 'more low-'}{dim} options."
            )
        else:
            lines.append(
                "  → sign is NOT trait-interpretable (letter rank shuffled "
                "per item)."
            )
        return "\n".join(lines)

    return (
        f'[{block}, loading={loading:+.3f}] item_id={item_id} '
        f'"{col_def.get("text", "?")}" (no block-aware renderer; '
        "rich context unavailable)."
    )


def cmd_describe(args: argparse.Namespace) -> None:
    path = Path(args.path).resolve()
    candidates = _discover_npz(path)
    if len(candidates) != 1:
        raise SystemExit(
            f"describe requires exactly one rotation; got {len(candidates)}. "
            "Run `resolve` first to pick one."
        )
    npz_path = candidates[0]
    resolved = _resolve_one(npz_path)

    data = np.load(npz_path)
    loadings = data["loadings"]
    communalities = data["communalities"]
    ss = data["ss_loadings"]
    pvar = data["proportion_variance"]
    fcorr = data["factor_correlation_matrix"] if "factor_correlation_matrix" in data.files else None
    n_items, n_factors = loadings.shape

    with open(resolved["item_labels_json"]) as f:
        column_defs = json.load(f)

    # Build a rich lookup preferring raw questionnaire sources over the
    # (possibly column_defs-only) items.json next to the npz. The returned
    # warnings explicitly describe any degradation so the labeller can
    # see when trait_mcq / fc / fc_pair / vignette items are being shown
    # without their options.
    items_by_id, rich_warnings = _build_items_lookup(resolved, column_defs)

    top_n = args.top_n
    out: list[str] = []
    out.append(f"# Factor context — {resolved['analysis_key']}")
    out.append("")
    out.append(f"- n_items: {n_items}, n_factors: {n_factors}")
    out.append(f"- npz: `{resolved['npz']}`")
    out.append(f"- save_dir: `{resolved['save_dir']}`")
    out.append(f"- write labels to: `{resolved['output_path']}`")
    blocks_present = sorted({c["block"] for c in column_defs})
    out.append(f"- blocks present: {blocks_present}")
    if rich_warnings:
        out.append("")
        out.append("> ⚠ **Rich-context warnings** — read these before labelling:")
        for w in rich_warnings:
            out.append(f"> - {w}")
    out.append("")

    for fi in range(n_factors):
        col = loadings[:, fi]
        order = np.argsort(col)
        top_pos = [i for i in order[-top_n:][::-1] if col[i] > 0]
        top_neg = [i for i in order[:top_n] if col[i] < 0]

        out.append("---")
        out.append("")
        out.append(f"## Factor {fi}")
        out.append(
            f"- SS loading: {float(ss[fi]):.3f}  |  "
            f"prop. variance: {float(pvar[fi]):.3%}"
        )
        if fcorr is not None:
            neighbours = sorted(
                [(fj, float(fcorr[fi, fj])) for fj in range(n_factors) if fj != fi],
                key=lambda x: -abs(x[1]),
            )[:3]
            if neighbours:
                out.append(
                    "- strongest factor correlations: "
                    + ", ".join(f"f{fj}={r:+.2f}" for fj, r in neighbours)
                )
        out.append("")

        out.append(f"### Positive pole ({len(top_pos)} items)")
        for i in top_pos:
            comm = float(communalities[i])
            cross_js = [fj for fj in range(n_factors) if fj != fi]
            max_fj = max(cross_js, key=lambda fj: abs(loadings[i, fj]))
            max_cross = float(loadings[i, max_fj])
            out.append(
                f"- col_id=`{column_defs[i]['col_id']}` "
                f"comm={comm:.2f} max-cross=f{max_fj}:{max_cross:+.2f}"
            )
            out.append(
                "  ```\n  "
                + _describe_item(column_defs[i], float(col[i]), items_by_id).replace("\n", "\n  ")
                + "\n  ```"
            )
        out.append("")

        out.append(f"### Negative pole ({len(top_neg)} items)")
        for i in top_neg:
            comm = float(communalities[i])
            cross_js = [fj for fj in range(n_factors) if fj != fi]
            max_fj = max(cross_js, key=lambda fj: abs(loadings[i, fj]))
            max_cross = float(loadings[i, max_fj])
            out.append(
                f"- col_id=`{column_defs[i]['col_id']}` "
                f"comm={comm:.2f} max-cross=f{max_fj}:{max_cross:+.2f}"
            )
            out.append(
                "  ```\n  "
                + _describe_item(column_defs[i], float(col[i]), items_by_id).replace("\n", "\n  ")
                + "\n  ```"
            )
        out.append("")

    print("\n".join(out))


# ─────────────────────────────────────────────────────────────────────────────
# Write labels
# ─────────────────────────────────────────────────────────────────────────────


def _validate_labels(payload: Any, n_factors: int) -> list[dict]:
    if isinstance(payload, dict) and "factors" in payload:
        payload = payload["factors"]
    if not isinstance(payload, list):
        raise SystemExit("Labels payload must be a list (or {'factors': [...]}).")
    seen = set()
    for entry in payload:
        if not isinstance(entry, dict):
            raise SystemExit(f"Each label must be a dict; got {type(entry).__name__}.")
        missing = [f for f in REQUIRED_LABEL_FIELDS if f not in entry]
        if missing:
            raise SystemExit(f"Label missing fields {missing}: {entry}")
        fi = entry["factor_index"]
        if not isinstance(fi, int):
            raise SystemExit(f"factor_index must be int, got {fi!r}.")
        if fi in seen:
            raise SystemExit(f"Duplicate factor_index {fi}.")
        seen.add(fi)
        axis = str(entry.get("axis_name", "")).strip()
        if not axis:
            raise SystemExit(f"axis_name must be non-empty for factor {fi}.")
    if len(payload) != n_factors:
        raise SystemExit(
            f"Expected {n_factors} labels, got {len(payload)}."
        )
    missing_idx = set(range(n_factors)) - seen
    if missing_idx:
        raise SystemExit(f"Missing labels for factor indices: {sorted(missing_idx)}")
    return sorted(payload, key=lambda e: e["factor_index"])


def cmd_write(args: argparse.Namespace) -> None:
    path = Path(args.path).resolve()
    candidates = _discover_npz(path)
    if len(candidates) != 1:
        raise SystemExit(
            f"write requires exactly one rotation; got {len(candidates)}."
        )
    resolved = _resolve_one(candidates[0])

    data = np.load(candidates[0])
    n_factors = int(data["loadings"].shape[1])

    with open(args.labels) as f:
        payload = json.load(f)
    labels = _validate_labels(payload, n_factors)

    out_path = Path(args.output) if args.output else Path(resolved["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "written": str(out_path),
        "analysis_key": resolved["analysis_key"],
        "n_factors": n_factors,
    }, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("resolve", help="Resolve paths and analysis key.")
    r.add_argument("path")
    r.set_defaults(func=cmd_resolve)

    d = sub.add_parser("describe", help="Emit per-factor markdown context.")
    d.add_argument("path")
    d.add_argument("--top-n", type=int, default=10)
    d.set_defaults(func=cmd_describe)

    w = sub.add_parser("write", help="Validate and write labels JSON.")
    w.add_argument("path")
    w.add_argument("--labels", required=True, help="Path to JSON file with the labels payload.")
    w.add_argument("--output", default=None, help="Override output path.")
    w.set_defaults(func=cmd_write)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
