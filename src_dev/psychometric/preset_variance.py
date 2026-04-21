"""Per-preset variance decomposition of FA factor scores.

For a combined-run FA (multiple rollout presets in one matrix), this
reports how much of each factor's total variance is explained by the
``rollout_preset_key`` grouping. High η² means the factor is largely
tracking *which dataset/model a persona came from* rather than
within-population persona variation — usually a sign that the largest
factor in a mixed run is an artefact of the mixing.

Output (per rotation):

    - CSV with columns: factor, proportion_variance, eta2_preset,
      dominated_by_preset_split
    - Optional bar plot showing "factor variance (% of total)" vs
      "% of that factor's variance explained by preset split".

The η² computation uses the same ``prompt_effects`` helper that the
validation stage uses for archetype/scenario fields — this module just
calls it with ``group_field="rollout_preset_key"`` and writes a
dedicated report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src_dev.factor_analysis.interpretation import prompt_effects
from src_dev.psychometric.preprocessing import preprocess_response_matrix


def _align_metadata_to_scores(
    response_matrix: np.ndarray,
    metadata: list[dict],
    column_defs: list[dict],
    *,
    min_item_variance: float,
    high_variance_persona_drop_pct: float,
) -> list[dict]:
    """Replay the FA preprocessing's row-filtering to align metadata to scores.

    ``preprocess_response_matrix`` drops personas with any NaN values (and
    optionally drops high-variance rows). FA scores have the post-filter
    length; we rerun the preprocessing just to recover that row mask.
    """
    _cleaned, meta_filtered, _cols, _groups = preprocess_response_matrix(
        response_matrix, metadata, column_defs,
        min_item_variance=min_item_variance,
        high_variance_persona_drop_pct=high_variance_persona_drop_pct,
        do_residualize=False,
    )
    return meta_filtered


def report_preset_variance(
    fa_dir: Path,
    response_matrix: np.ndarray,
    metadata: list[dict],
    column_defs: list[dict],
    *,
    min_item_variance: float,
    high_variance_persona_drop_pct: float,
    group_field: str = "rollout_preset_key",
    out_dir: Path | None = None,
) -> dict[str, list[dict]]:
    """Compute per-factor η²-by-preset for every rotation in ``fa_dir``.

    Args:
        fa_dir: Root directory containing per-rotation subdirs
            (e.g. ``raw``, ``raw_oblimin``, ``raw_varimax``). Each subdir
            holds one or more ``fa_<k>_<method>_<rotation>.npz`` files.
        response_matrix / metadata / column_defs: the same inputs Stage 3
            consumes; used to replay preprocessing and align metadata to
            the FA scores' row count.
        min_item_variance / high_variance_persona_drop_pct: Stage 3 config
            (must match what produced the FA artifacts).
        group_field: Metadata key to group on. Default
            ``rollout_preset_key`` (set automatically by the combine step).
        out_dir: Where to write ``preset_variance_<rotation>.csv`` +
            a summary ``preset_variance_summary.json``. Defaults to
            ``fa_dir``.

    Returns:
        Mapping of rotation-subdir name → list of per-factor dicts.
    """
    out_dir = out_dir or fa_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_aligned = _align_metadata_to_scores(
        response_matrix, metadata, column_defs,
        min_item_variance=min_item_variance,
        high_variance_persona_drop_pct=high_variance_persona_drop_pct,
    )
    n_presets_seen = len({
        str(r.get(group_field)) for r in meta_aligned
        if r.get(group_field) is not None
    })
    print(
        f"[Preset-variance] Grouping variable: {group_field!r}  "
        f"({n_presets_seen} preset(s) present in {len(meta_aligned)} aligned rows)"
    )
    if n_presets_seen < 2:
        print(
            f"[Preset-variance] Only {n_presets_seen} preset present — "
            "variance decomposition is trivial; skipping."
        )
        return {}

    # FA artifacts are laid out as
    #   fa_dir/<scope>/fa_<k>_<method>_<rotation>.npz
    # where <scope> is typically "raw" (or "raw_residualized" etc.) and the
    # rotation name is the last underscore-segment of the filename stem.
    # Collect all npz files across scopes and label each result by
    # "<scope>/<rotation>" so we don't collapse multiple rotations into one
    # rotation-dir heuristic.
    per_rotation: dict[str, list[dict]] = {}
    npz_paths = sorted(fa_dir.rglob("fa_*_*_*.npz"))
    for npz in npz_paths:
        stem = npz.stem                      # e.g. "fa_7_principal_oblimin"
        scope = npz.parent.name               # e.g. "raw"
        rotation = stem.split("_")[-1]        # "oblimin" / "varimax"
        rot_key = f"{scope}/{rotation}"
        data = np.load(npz, allow_pickle=True)
        scores = data["scores"]
        proportion_variance = data["proportion_variance"]

        if scores.shape[0] != len(meta_aligned):
            print(
                f"[Preset-variance] WARN: rotation={rot_key} scores rows "
                f"{scores.shape[0]} ≠ aligned meta rows {len(meta_aligned)}; "
                "skipping this rotation."
            )
            continue

        eta2 = prompt_effects(scores, meta_aligned, group_field=group_field)
        rows = []
        for f_idx in range(scores.shape[1]):
            rows.append({
                "factor": f_idx + 1,
                "proportion_variance": float(proportion_variance[f_idx]),
                "eta2_preset": float(eta2[f_idx]),
                "dominated_by_preset": bool(eta2[f_idx] > 0.5),
            })
        per_rotation[rot_key] = rows

        safe_rot = rot_key.replace("/", "__")
        _write_csv(out_dir / f"preset_variance_{safe_rot}.csv", rows)
        print(f"\n[Preset-variance] rotation={rot_key}  (npz={npz.name})")
        print(
            f"  {'factor':>6}  {'prop_var':>9}  {'eta²':>7}  note"
        )
        for r in sorted(rows, key=lambda r: -r["proportion_variance"]):
            marker = "  ← dominated by preset split" if r["dominated_by_preset"] else ""
            print(
                f"  {r['factor']:>6d}  {r['proportion_variance']:>9.4f}  "
                f"{r['eta2_preset']:>7.3f}{marker}"
            )

    summary_path = out_dir / "preset_variance_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "group_field": group_field,
                "n_presets": n_presets_seen,
                "n_aligned_rows": len(meta_aligned),
                "rotations": per_rotation,
            },
            f,
            indent=2,
        )
    print(f"\n[Preset-variance] Wrote summary → {summary_path}")
    return per_rotation


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    with path.open("w", encoding="utf-8", newline="") as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
