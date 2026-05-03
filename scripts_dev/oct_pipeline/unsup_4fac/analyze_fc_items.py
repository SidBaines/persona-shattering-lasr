"""Per-item analysis of the F0 forced-choice questionnaire.

For each of the 32 FC items, computes:
  - Pearson r between the item's per-persona response (in [-1, +1] after
    averaging the two orderings) and per-persona FA F0 score across the
    200 baseline personas.
  - Mean and variance of item responses (flags saturated items).
  - "Pole-aligned shift" of the item's mean from 0.

Then reports:
  - Top items (high positive r with F0): the load-bearing FC items.
  - Bottom items (negative r): potentially mis-keyed or measuring something
    different from F0.
  - Saturated items: both options' baseline pick rate is < 5% / > 95%
    so the item can't distinguish anyone.
  - Cronbach's α for the full questionnaire and for the trimmed top-N
    subsets.
  - Trimmed FC↔FA F0 correlation: how much improvement we'd get by dropping
    the worst N items.

By default reads from the local validate_fc_persona output dir; falls
back to pulling the per-ordering response matrices + metadata from
monorepo if the local files are missing.

Usage::

    uv run python scripts_dev/oct_pipeline/unsup_4fac/analyze_fc_items.py

    # Or override the label / questionnaire to inspect a different run:
    uv run python scripts_dev/oct_pipeline/unsup_4fac/analyze_fc_items.py \\
        --label conviction_amp_dpo_v6_fc_persona
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src_dev.utils.hf_hub import download_path_to_dir

MONOREPO_REPO = "persona-shattering-lasr/monorepo"
LOCAL_VALIDATE_ROOT = Path("scratch/factor_inspect/validate_fc_persona")
DEFAULT_QUESTIONNAIRE = Path("datasets/psychometric_questionnaires/f0_forced_choice_v1.json")


# ── HF rehydrate helpers ────────────────────────────────────────────────────


def hf_path_for_label(label: str) -> str:
    """Mirror of validate_lora_fc_persona.hf_path_for_label, baseline-only.

    For non-baseline labels we don't actually know the LoRA prefix without
    parsing the adapter URL. The user should pass --hf-prefix to override.
    """
    if label == "baseline_fc_persona":
        return f"evals/factor_validate_fc_persona/{label}"
    # Best-effort; the caller can override.
    return ""


def ensure_local_dir(local_dir: Path, hf_subpath_root: str, ord_label: str) -> bool:
    """Ensure local_dir/questionnaire_<ord_label>/ has the response_matrix
    + metadata. Try local first, fall back to HF rehydrate. Returns True on
    success."""
    ord_dir = local_dir / f"questionnaire_{ord_label}"
    response_matrix_path = ord_dir / "response_matrix.npy"
    if response_matrix_path.exists() and (ord_dir / "metadata.jsonl").exists():
        return True
    if not hf_subpath_root:
        return False
    hf_subpath = f"{hf_subpath_root}/questionnaire_{ord_label}"
    print(f"[items] rehydrating from HF: {hf_subpath}")
    try:
        download_path_to_dir(
            repo_id=MONOREPO_REPO,
            path_in_repo=hf_subpath,
            target_dir=ord_dir,
        )
    except Exception as exc:
        print(f"[items] HF rehydrate failed: {type(exc).__name__}: {exc}")
        return False
    return response_matrix_path.exists()


# ── Load FC + FA data ──────────────────────────────────────────────────────


def load_fc_matrix(local_dir: Path, hf_subpath_root: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Average the two orderings to get an (n_personas, n_items) matrix in [-1, +1].

    Returns (M_avg, sample_ids, item_ids).
    """
    matrices: list[np.ndarray] = []
    sample_ids_per: list[list[str]] = []
    item_ids_per: list[list[str]] = []

    for ord_label in ("high_as_A", "high_as_B"):
        ok = ensure_local_dir(local_dir, hf_subpath_root, ord_label)
        if not ok:
            raise FileNotFoundError(f"missing data for ordering {ord_label} in {local_dir}")
        ord_dir = local_dir / f"questionnaire_{ord_label}"
        M = np.load(ord_dir / "response_matrix.npy").astype(float)
        meta = [
            json.loads(line) for line in
            (ord_dir / "metadata.jsonl").read_text().splitlines() if line.strip()
        ]
        items = json.loads((ord_dir / "items.json").read_text())
        sids = [m["sample_id"] for m in meta]
        iids = [it["item_id"] for it in items]
        matrices.append(M)
        sample_ids_per.append(sids)
        item_ids_per.append(iids)

    if sample_ids_per[0] != sample_ids_per[1] or item_ids_per[0] != item_ids_per[1]:
        raise RuntimeError("ordering-1 and ordering-2 disagree on personas or items")

    M_avg = np.nanmean(np.stack(matrices, axis=0), axis=0)
    return M_avg, sample_ids_per[0], item_ids_per[0]


def load_fa_f0(local_dir: Path, label: str, sample_ids: list[str]) -> np.ndarray:
    """Pull per-persona FA F0 scores from the cached scores npz."""
    npz_path = local_dir / f"{label}_scores.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing {npz_path}; can't get FA F0 scores")
    z = np.load(npz_path, allow_pickle=True)
    if "fa_f0_per_persona" not in z.files:
        raise RuntimeError(f"{npz_path} has no 'fa_f0_per_persona' field")
    cached_sids = list(z["sample_ids"])
    fa_f0 = np.asarray(z["fa_f0_per_persona"])
    if cached_sids != sample_ids:
        # Realign.
        sid_to_row = {s: r for r, s in enumerate(cached_sids)}
        rows = [sid_to_row[s] for s in sample_ids if s in sid_to_row]
        if len(rows) != len(sample_ids):
            raise RuntimeError(
                f"persona mismatch: {len(rows)} cached vs {len(sample_ids)} target"
            )
        fa_f0 = fa_f0[rows]
    return fa_f0


# ── Load item metadata for the FC questionnaire ────────────────────────────


def load_item_facets(questionnaire_path: Path) -> dict[str, str]:
    fc_qsts = json.loads(questionnaire_path.read_text())
    return {it["id"]: it["facet"] for it in fc_qsts["block_2_forced_choice"]["items"]}


def load_item_texts(questionnaire_path: Path) -> dict[str, dict]:
    fc_qsts = json.loads(questionnaire_path.read_text())
    return {it["id"]: it for it in fc_qsts["block_2_forced_choice"]["items"]}


# ── Statistics ─────────────────────────────────────────────────────────────


def per_item_correlations(
    M_avg: np.ndarray,         # (n_personas, n_items), in [-1, +1]
    fa_f0: np.ndarray,         # (n_personas,)
) -> np.ndarray:
    """Per-item Pearson r with FA F0 across personas. NaN if undefined (no var)."""
    n_items = M_avg.shape[1]
    rs = np.full(n_items, np.nan)
    for j in range(n_items):
        col = M_avg[:, j]
        valid = np.isfinite(col) & np.isfinite(fa_f0)
        if valid.sum() < 5:
            continue
        if np.std(col[valid]) < 1e-9 or np.std(fa_f0[valid]) < 1e-9:
            continue  # saturated
        rs[j] = float(np.corrcoef(col[valid], fa_f0[valid])[0, 1])
    return rs


def per_item_means(M_avg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-item mean and std across personas."""
    means = np.nanmean(M_avg, axis=0)
    stds = np.nanstd(M_avg, axis=0)
    return means, stds


def cronbach_alpha(M_avg: np.ndarray) -> float:
    """Standard Cronbach's α on the persona × item matrix.

    α = (k / (k-1)) * (1 - sum(var_per_item) / var_total_score)

    Uses pole-aligned scores in [-1, +1]. NaNs treated as personas with that
    item missing (excluded item-by-item via nanvar).
    """
    n_items = M_avg.shape[1]
    if n_items < 2:
        return float("nan")
    item_var = np.nanvar(M_avg, axis=0, ddof=1)
    total = np.nansum(M_avg, axis=1)
    total_var = float(np.nanvar(total, ddof=1))
    if total_var < 1e-12:
        return float("nan")
    alpha = (n_items / (n_items - 1)) * (1 - float(np.nansum(item_var)) / total_var)
    return alpha


def trimmed_fc_correlation(
    M_avg: np.ndarray,
    fa_f0: np.ndarray,
    keep_idx: list[int],
) -> float:
    """Mean across kept items per persona, then Pearson r with FA F0."""
    if not keep_idx:
        return float("nan")
    sub = M_avg[:, keep_idx]
    fc_score = np.nanmean(sub, axis=1)
    valid = np.isfinite(fc_score) & np.isfinite(fa_f0)
    if valid.sum() < 5:
        return float("nan")
    return float(np.corrcoef(fc_score[valid], fa_f0[valid])[0, 1])


# ── Reporting ──────────────────────────────────────────────────────────────


def print_table_sorted_by_r(
    items: list[dict],
    rs: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    facets: dict[str, str],
    item_texts: dict[str, dict],
    label: str,
):
    print()
    print("=" * 110)
    print(f"  Per-item correlations with FA F0 across baseline personas — sorted by r")
    print(f"  ({label})")
    print("=" * 110)
    print(f"  {'r':>6s}  {'mean':>6s}  {'std':>5s}  {'facet':<22s}  id           short text")
    print("  " + "-" * 100)

    rows = []
    for j, it_id in enumerate(items):
        rows.append({
            "rank": None,
            "id": it_id,
            "facet": facets.get(it_id, "?"),
            "r": float(rs[j]) if np.isfinite(rs[j]) else float("nan"),
            "mean": float(means[j]),
            "std": float(stds[j]),
            "high_pole_text": item_texts.get(it_id, {}).get("high_pole_text", ""),
            "low_pole_text": item_texts.get(it_id, {}).get("low_pole_text", ""),
        })
    # Sort by r descending; NaN r at the bottom.
    rows.sort(key=lambda r: (-1e9 if not np.isfinite(r["r"]) else -r["r"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        rstr = f"{r['r']:+.3f}" if np.isfinite(r["r"]) else " nan "
        short = (r["high_pole_text"] or "")[:60].replace("\n", " ")
        print(
            f"  {rstr:>6s}  {r['mean']:+.3f}  {r['std']:.3f}  "
            f"{r['facet']:<22s}  {r['id']:<11s}  {short}"
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline_fc_persona")
    parser.add_argument("--hf-prefix", default=None,
                        help="HF subpath root (e.g. evals/factor_validate_fc_persona/<label>). "
                             "Default derives from --label for baseline; for LoRA labels pass it explicitly.")
    parser.add_argument("--questionnaire", default=str(DEFAULT_QUESTIONNAIRE))
    parser.add_argument("--saturate-threshold", type=float, default=0.95,
                        help="An item is 'saturated' if |mean| > this on baseline (1.0 = fully one-sided).")
    parser.add_argument("--out-json", default=None,
                        help="Write the per-item table + summary to this JSON path. "
                             "Default: scratch/factor_inspect/validate_fc_persona/<label>/<label>_item_analysis.json")
    args = parser.parse_args()

    local_dir = LOCAL_VALIDATE_ROOT / args.label
    local_dir.mkdir(parents=True, exist_ok=True)
    hf_subpath_root = args.hf_prefix or hf_path_for_label(args.label)
    if hf_subpath_root:
        print(f"[items] HF subpath root: {hf_subpath_root}")
    else:
        print(f"[items] no HF subpath root; will use local files only")

    # 1. Load FC per-item matrix.
    M_avg, sample_ids, item_ids = load_fc_matrix(local_dir, hf_subpath_root)
    print(f"[items] FC M_avg shape: {M_avg.shape}  (n_personas, n_items)")

    # 2. Load FA F0 baseline scores.
    fa_f0 = load_fa_f0(local_dir, args.label, sample_ids)
    print(f"[items] FA F0 vector length: {len(fa_f0)}")

    # 3. Per-item statistics.
    rs = per_item_correlations(M_avg, fa_f0)
    means, stds = per_item_means(M_avg)
    facets = load_item_facets(Path(args.questionnaire))
    item_texts = load_item_texts(Path(args.questionnaire))
    rows = print_table_sorted_by_r(item_ids, rs, means, stds, facets, item_texts, args.label)

    # 4. Saturation check.
    saturated = [i for i in range(len(item_ids)) if abs(means[i]) > args.saturate_threshold]
    print()
    print(f"Saturated items (|mean| > {args.saturate_threshold}): {len(saturated)}")
    for j in saturated:
        print(f"  [{item_ids[j]}] mean={means[j]:+.3f}  facet={facets.get(item_ids[j], '?')}")

    # 5. Cronbach's α — full and after trimming.
    alpha_full = cronbach_alpha(M_avg)
    print()
    print(f"Cronbach's α (all 32 items, baseline): {alpha_full:+.3f}")

    # Trim to items with r > thresholds and report new α + FC↔FA correlation.
    print()
    print("Effect of trimming on questionnaire-level statistics:")
    print(f"  {'criterion':<35s}  {'n_kept':>7s}  {'alpha':>7s}  {'FC↔FA r':>9s}")
    for thresh in [0.0, 0.05, 0.10, 0.15, 0.20]:
        keep = [j for j in range(len(item_ids)) if np.isfinite(rs[j]) and rs[j] >= thresh]
        a = cronbach_alpha(M_avg[:, keep]) if len(keep) >= 2 else float("nan")
        rr = trimmed_fc_correlation(M_avg, fa_f0, keep)
        print(f"  r >= {thresh:.2f}                          {len(keep):>7d}  {a:+.3f}  {rr:+.3f}")

    # Drop saturated.
    keep_unsat = [j for j in range(len(item_ids)) if abs(means[j]) <= args.saturate_threshold]
    a = cronbach_alpha(M_avg[:, keep_unsat]) if len(keep_unsat) >= 2 else float("nan")
    rr = trimmed_fc_correlation(M_avg, fa_f0, keep_unsat)
    print(f"  drop saturated only             {len(keep_unsat):>7d}  {a:+.3f}  {rr:+.3f}")

    # Drop saturated AND r < 0.10
    keep_combo = [
        j for j in range(len(item_ids))
        if abs(means[j]) <= args.saturate_threshold
        and np.isfinite(rs[j]) and rs[j] >= 0.10
    ]
    a = cronbach_alpha(M_avg[:, keep_combo]) if len(keep_combo) >= 2 else float("nan")
    rr = trimmed_fc_correlation(M_avg, fa_f0, keep_combo)
    print(f"  drop saturated + r < 0.10        {len(keep_combo):>7d}  {a:+.3f}  {rr:+.3f}")

    # 6. Write JSON output.
    out_path = Path(args.out_json) if args.out_json else local_dir / f"{args.label}_item_analysis.json"
    out = {
        "label": args.label,
        "n_personas": int(M_avg.shape[0]),
        "n_items": int(M_avg.shape[1]),
        "alpha_full": alpha_full,
        "alpha_after_drop_saturated": cronbach_alpha(M_avg[:, keep_unsat]) if len(keep_unsat) >= 2 else None,
        "alpha_after_drop_saturated_and_lowr": cronbach_alpha(M_avg[:, keep_combo]) if len(keep_combo) >= 2 else None,
        "fc_fa_r_full": float(trimmed_fc_correlation(M_avg, fa_f0, list(range(len(item_ids))))),
        "fc_fa_r_after_drop_saturated": float(trimmed_fc_correlation(M_avg, fa_f0, keep_unsat)),
        "fc_fa_r_after_drop_saturated_and_lowr": float(trimmed_fc_correlation(M_avg, fa_f0, keep_combo)),
        "saturated_item_ids": [item_ids[j] for j in saturated],
        "kept_item_ids_after_combo_trim": [item_ids[j] for j in keep_combo],
        "per_item": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print()
    print(f"Wrote per-item analysis JSON: {out_path}")


if __name__ == "__main__":
    main()
