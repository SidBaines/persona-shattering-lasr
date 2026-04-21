"""Response-matrix preprocessing for factor analysis.

Responsibilities:

* Drop persona rows with any missing values (parse failures) — mean-
  imputation would attenuate correlations and bias factor loadings.
* Optionally drop top-percentile-variance personas (incoherent / garbage
  rollouts).
* Drop low-variance columns using a **per-block relative** threshold:
  each column's variance is divided by its block's median non-zero
  variance, yielding a scale-invariant "relative variance" that can be
  thresholded uniformly across blocks with different raw variance scales
  (Likert 1-5 ~ 0.5–2, fc_pair ±1 ~ 0.5–1, trait_mcq 0/1 ~ 0.1–0.25).
* Optionally residualize by per-prompt-group means (skipped when groups
  are size-1 since subtracting a single-element mean zeros everything).
* Export per-column variance diagnostics.

The statistical residualization primitive lives in
``src_dev.factor_analysis.preprocessing.residualize`` and is wrapped here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src_dev.factor_analysis.preprocessing import residualize


def preprocess_response_matrix(
    response_matrix: np.ndarray,
    metadata: list[dict],
    column_defs: list[dict],
    *,
    min_item_variance: float = 0.1,
    high_variance_persona_drop_pct: float = 0.0,
    do_residualize: bool = False,
    residualize_group_field: str | None = None,
    variance_export_path: Path | None = None,
) -> tuple[np.ndarray, list[dict], list[dict], np.ndarray | None]:
    """Preprocess the response matrix for factor analysis.

    Args:
        response_matrix: Raw persona × item matrix.
        metadata: Per-row metadata (same length as ``response_matrix``).
        column_defs: Per-column definitions (same length as item axis).
        min_item_variance: Per-block relative-variance floor. A column is
            dropped if its variance is below this fraction of its block's
            median non-zero variance. 0 disables filtering.
        high_variance_persona_drop_pct: Drop personas whose across-item
            response variance is in the top N percentile. 0 disables.
        do_residualize: If True, subtract per-group means via
            ``residualize`` (skipped when groups are size-1).
        residualize_group_field: Metadata field used as the grouping
            variable when ``do_residualize=True``. Defaults to
            ``"input_group_id"`` (the standard A/B generated-rollout
            grouping). For multi-preset external-rollout runs, set to
            ``"rollout_preset_key"`` to subtract per-model means and
            strip out between-model variance before FA.
        variance_export_path: If provided, write a JSONL file at this path
            with one row per column ranked by pre-filter variance (computed
            after row-level filtering).

    Returns:
        ``(cleaned matrix, filtered metadata, filtered column_defs, group_ids_or_None)``.
    """
    effective_group_field = residualize_group_field or "input_group_id"
    K, M = response_matrix.shape

    # Drop rows with any missing values (parse failures).
    missing_per_row = np.sum(np.isnan(response_matrix), axis=1)
    row_mask = missing_per_row == 0
    data = response_matrix[row_mask].copy()
    meta_filtered = [m for m, keep in zip(metadata, row_mask) if keep]
    n_dropped = K - data.shape[0]
    print(f"  Kept {data.shape[0]}/{K} rows (dropped {n_dropped} with any missing values)")
    if n_dropped > 0:
        missing_counts = missing_per_row[~row_mask]
        print(
            f"  Dropped row missing-value counts: "
            f"median={np.median(missing_counts):.0f}, "
            f"max={np.max(missing_counts):.0f}"
        )

    # Drop high-variance personas (potential incoherent/garbage rollouts).
    # Computed on complete rows before column filtering so the threshold is
    # stable regardless of which columns survive the low-variance filter.
    if high_variance_persona_drop_pct > 0:
        row_vars = np.var(data, axis=1)
        threshold = np.percentile(row_vars, 100 - high_variance_persona_drop_pct)
        keep = row_vars <= threshold
        n_before = data.shape[0]
        data = data[keep]
        meta_filtered = [m for m, k in zip(meta_filtered, keep) if k]
        n_hi_var_dropped = n_before - data.shape[0]
        print(
            f"  Dropped {n_hi_var_dropped}/{n_before} high-variance personas "
            f"(top {high_variance_persona_drop_pct}%, var > {threshold:.3f})"
        )

    # Drop low-variance columns using a per-block relative threshold.
    col_var = np.var(data, axis=0)
    col_blocks = np.array([str(c.get("block", "")) for c in column_defs])
    col_var_rel = np.zeros_like(col_var, dtype=np.float64)
    block_scales: dict[str, float] = {}
    for block in np.unique(col_blocks):
        block_mask = col_blocks == block
        block_vars = col_var[block_mask]
        pos = block_vars[block_vars > 0]
        median_var = float(np.median(pos)) if pos.size > 0 else 0.0
        block_scales[block] = median_var
        if median_var > 0:
            col_var_rel[block_mask] = col_var[block_mask] / median_var
        else:
            # No column in this block has positive variance — all zero-relative,
            # so all will be dropped by any positive threshold.
            col_var_rel[block_mask] = 0.0
    col_mask = col_var_rel >= min_item_variance

    if variance_export_path is not None:
        variance_export_path.parent.mkdir(parents=True, exist_ok=True)
        ranked = sorted(
            zip(column_defs, col_var, col_var_rel, col_mask),
            key=lambda r: float(r[2]),
            reverse=True,
        )
        with variance_export_path.open("w", encoding="utf-8") as f:
            for col_def, var, var_rel, keep in ranked:
                row = {
                    "variance": float(var),
                    "variance_relative_to_block_median": float(var_rel),
                    "block_median_variance": float(
                        block_scales.get(str(col_def.get("block", "")), 0.0)
                    ),
                    "kept": bool(keep),
                    "block": col_def.get("block"),
                    "item_id": col_def.get("item_id"),
                    "col_id": col_def.get("col_id"),
                    "dimension": col_def.get("dimension"),
                    "question": col_def.get("text"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  Wrote ranked item variances to {variance_export_path}")

    data = data[:, col_mask]
    cols_filtered = [col for col, keep in zip(column_defs, col_mask) if keep]
    dropped_cols = int(np.sum(~col_mask))
    if dropped_cols > 0:
        drops_by_block: dict[str, int] = {}
        kept_by_block: dict[str, int] = {}
        for b, keep in zip(col_blocks, col_mask):
            if keep:
                kept_by_block[b] = kept_by_block.get(b, 0) + 1
            else:
                drops_by_block[b] = drops_by_block.get(b, 0) + 1
        print(
            f"  Dropped {dropped_cols}/{M} columns (relative variance < {min_item_variance} "
            f"of per-block median)"
        )
        for b in sorted(set(drops_by_block) | set(kept_by_block)):
            print(
                f"    block={b!r}: kept {kept_by_block.get(b, 0)}, "
                f"dropped {drops_by_block.get(b, 0)}, "
                f"median_var={block_scales.get(b, 0.0):.4f}"
            )
    print(f"  Final matrix shape: {data.shape}")

    group_ids = None
    if do_residualize:
        # Check that residualization is meaningful (need >1 sample per group
        # on average, otherwise subtracting group means zeros everything out)
        group_counts: dict[str, int] = {}
        for m in meta_filtered:
            gid = m.get(effective_group_field, m["sample_id"])
            group_counts[gid] = group_counts.get(gid, 0) + 1
        max_group_size = max(group_counts.values()) if group_counts else 0

        if max_group_size <= 1:
            print(
                f"  Skipping residualization: all groups (by "
                f"{effective_group_field!r}) have size 1"
            )
        else:
            data, _group_means, group_inv = residualize(
                data, meta_filtered, group_field=effective_group_field,
            )
            group_ids = group_inv
            n_groups = len(group_counts)
            print(
                f"  Residualized across {n_groups} groups "
                f"(field={effective_group_field!r})"
            )

            # Re-filter columns created with near-zero variance by residualization.
            # Uses a small absolute epsilon here — residualization can zero a
            # column outright (if it was constant within every group) and we
            # need to drop those before factoring; a relative threshold is
            # less meaningful post-residualization since the scale reference
            # is gone.
            col_var_post = np.var(data, axis=0)
            col_mask_post = col_var_post >= 1e-8
            if not col_mask_post.all():
                dropped_post = int(np.sum(~col_mask_post))
                data = data[:, col_mask_post]
                cols_filtered = [c for c, keep in zip(cols_filtered, col_mask_post) if keep]
                print(f"  Dropped {dropped_post} zero-variance columns after residualization")

    return data, meta_filtered, cols_filtered, group_ids
