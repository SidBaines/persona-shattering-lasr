"""Post-hoc per-block breakdown of F0 dominant-item shifts for v6 paired-DPO.

The standard validate_lora output groups the 66 F0-dominant items together,
but F0 is a mixture of two semantically distinct sub-axes that we want to
see separately:

  * Sub-axis A (Likert, ~37 items): self-reports of engaged-epistemic
    behaviour (verify / show working / hedge / charitable-pushback / etc),
    PLUS a small number of reverse-keyed yielding items that are partly
    acquiescence-contaminated (over-correct, thank-and-update, concede).
  * Sub-axis B (trait_mcq, ~29 items): the conscientious-introvert-stable
    action-recommendation pattern — pick the dutiful/structured option on
    Openness/Conscientiousness/Extraversion/Neuroticism MCQs. **The
    Engaged-with-Stakes facet of v6 explicitly targets this sub-axis.**

If the per-block split shows that the trait_mcq items moved in the trained
direction (toward HIGH for amp, toward LOW for sup) while the Likerts
moved against it, then v6's Engaged-with-Stakes facet *is* contributing
signal but the score is being dragged the wrong way by Sub-axis A's
acquiescence contamination. If the trait_mcq items also went the wrong
way, the new facet didn't get purchase from the constitution.

Reads cached files only — no GPU, no vLLM, runs in seconds.

Run::

    uv run python scripts_dev/oct_pipeline/unsup_4fac/analyze_v6_per_block_shifts.py

Optionally pass labels (default both v6 paired-DPO labels):

    uv run python ... \\
        --label conviction_amp_dpo_v6 conviction_sup_dpo_v6 \\
                conviction_amp_dpo_v6_singleteacher conviction_sup_dpo_v6_singleteacher
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Paths produced by validate_lora.py / inspect_factor_loadings.py.
FA_FIT_NPZ = Path("scratch/factor_inspect/fa_fit.npz")
FA_ITEMS = Path("scratch/factor_inspect/items.json")
VALIDATE_ROOT = Path("scratch/factor_inspect/validate")
HYDRATE_ROOT = Path("scratch/factor_inspect/hydrated")

V5_QNAME = (
    "questionnaire-rollouts-llama318binstruct-t1.0-15t-2500p-seed436-"
    "scenarios_v2-uprompt_v6-q_v5-likert-direct-lp20"
)
MCQ_QNAME = (
    "questionnaire-rollouts-llama318binstruct-t1.0-15t-2500p-seed436-"
    "scenarios_v2-uprompt_v6-q_trait_ocean_natural_v1-trait_mcq-aside-lp20-"
    "p2-pf2-tmv2"
)

CANONICAL_FACTOR_NAMES = ["F0_Conviction", "F1_Exuberance", "F2_Warmth", "F3_Didacticism"]
F0_INDEX = 0
LOADING_THRESHOLD = 0.4


def load_questionnaire_dir(dir_path: Path):
    """Return (M, sample_ids, items) for one questionnaire dir."""
    M = np.load(dir_path / "response_matrix.npy").astype(float)
    meta = [
        json.loads(line)
        for line in (dir_path / "metadata.jsonl").read_text().splitlines()
        if line.strip()
    ]
    items = json.loads((dir_path / "items.json").read_text())
    sample_ids = [m["sample_id"] for m in meta]
    if M.shape[0] != len(sample_ids):
        raise RuntimeError(f"row mismatch in {dir_path}: M={M.shape[0]} meta={len(sample_ids)}")
    if M.shape[1] != len(items):
        raise RuntimeError(f"col mismatch in {dir_path}: M={M.shape[1]} items={len(items)}")
    return M, sample_ids, items


def load_combined(parent: Path):
    """Load v5+trait_mcq, hstack, return (M, sample_ids, items)."""
    v5_dir = parent / "questionnaire_v5"
    mcq_dir = parent / "questionnaire_trait_ocean_natural_v1"
    if not v5_dir.exists() or not mcq_dir.exists():
        raise FileNotFoundError(
            f"expected {v5_dir} and {mcq_dir} (run validate_lora.py first)"
        )
    M_v5, sids_v5, items_v5 = load_questionnaire_dir(v5_dir)
    M_mcq, sids_mcq, items_mcq = load_questionnaire_dir(mcq_dir)
    if sids_v5 != sids_mcq:
        # Align on common sample_ids in v5 order.
        common = [s for s in sids_v5 if s in set(sids_mcq)]
        idx_v5 = [sids_v5.index(s) for s in common]
        idx_mcq = [sids_mcq.index(s) for s in common]
        M_v5 = M_v5[idx_v5]
        M_mcq = M_mcq[idx_mcq]
        sids_v5 = common
    items = []
    for it in items_v5:
        d = dict(it)
        d.setdefault("version", "v5")
        items.append(d)
    for it in items_mcq:
        d = dict(it)
        d.setdefault("version", "trait_ocean_natural_v1")
        items.append(d)
    return np.hstack([M_v5, M_mcq]), sids_v5, items


def align_to_fit_items(M_combined: np.ndarray, items_combined: list[dict], fit_items: list[dict]):
    """Slice columns of M_combined to match fit_items' (item_id, block) order."""
    new_index = {
        (it["item_id"], it.get("block", "")): i for i, it in enumerate(items_combined)
    }
    cols = []
    missing = []
    for it in fit_items:
        key = (it["item_id"], it.get("block", ""))
        if key not in new_index:
            missing.append(key)
        else:
            cols.append(new_index[key])
    if missing:
        raise RuntimeError(
            f"Subsample missing {len(missing)} items present in FA fit "
            f"(e.g. {missing[:3]})."
        )
    return M_combined[:, cols]


def subset_baseline_to(sids_target: list[str], M_baseline_full: np.ndarray, sids_baseline_full: list[str]):
    """Subset baseline rows to the target sample_ids, in target order."""
    sid_to_row = {s: r for r, s in enumerate(sids_baseline_full)}
    rows = []
    missing = []
    for s in sids_target:
        if s in sid_to_row:
            rows.append(sid_to_row[s])
        else:
            missing.append(s)
    if missing:
        raise RuntimeError(
            f"baseline missing {len(missing)} sample_ids present in the LoRA run "
            f"(e.g. {missing[:3]})."
        )
    return M_baseline_full[rows]


def _bucket_summary(label: str, signed_shifts: np.ndarray, n_total: int) -> str:
    if len(signed_shifts) == 0:
        return f"  {label:>26s}:  (no items in this bucket)"
    n = len(signed_shifts)
    high = int((signed_shifts > 0).sum())
    return (
        f"  {label:>26s}:  n={n:3d}  "
        f"mean_shift={signed_shifts.mean():+.3f}  "
        f"median={float(np.median(signed_shifts)):+.3f}  "
        f"toward HIGH={high:2d}/{n}  ({high/n*100:.0f}%)"
    )


def analyze_one(label: str, fa_loadings: np.ndarray, fit_items: list[dict],
                M_baseline_full: np.ndarray, sids_baseline_full: list[str]) -> None:
    parent = VALIDATE_ROOT / label
    if not parent.exists():
        print(f"[skip] {label}: {parent} does not exist")
        return

    print()
    print("=" * 88)
    print(f"  {label}")
    print("=" * 88)

    # LoRA matrix aligned to fit_items.
    M_lora_combined, sids_lora, items_lora = load_combined(parent)
    M_lora = align_to_fit_items(M_lora_combined, items_lora, fit_items)

    # Baseline subset to the same personas, same column order.
    M_base = subset_baseline_to(sids_lora, M_baseline_full, sids_baseline_full)

    if M_lora.shape != M_base.shape:
        raise RuntimeError(f"shape mismatch: M_lora={M_lora.shape}, M_base={M_base.shape}")

    delta_mean = M_lora.mean(axis=0) - M_base.mean(axis=0)  # (n_items,)

    # F0-dominant items.
    dominant = np.abs(fa_loadings).argmax(axis=1)
    load_f0 = fa_loadings[:, F0_INDEX]
    f0_mask = (dominant == F0_INDEX) & (np.abs(load_f0) >= LOADING_THRESHOLD)
    f0_idx = np.where(f0_mask)[0]

    n_total = int(f0_mask.sum())
    print(f"\nF0 dominant items at |loading|>={LOADING_THRESHOLD}: n={n_total}")

    # Pole-aligned shift per item: sign(loading) * delta_mean
    signed = np.sign(load_f0[f0_mask]) * delta_mean[f0_mask]

    # Bucket by block.
    blocks = np.array([fit_items[i].get("block", "?") for i in f0_idx])
    rev = np.array([bool(fit_items[i].get("reverse_keyed", False)) for i in f0_idx])

    print()
    print(f"Pole-aligned shift on F0 dominant items, by block (positive => moved toward F0+ pole):")
    print(_bucket_summary("ALL F0 dominant", signed, n_total))
    print(_bucket_summary("Likert (Sub-axis A+C)", signed[blocks == "likert"], (blocks == "likert").sum()))
    print(_bucket_summary("  ↳ non-reverse-keyed", signed[(blocks == "likert") & ~rev], 0))
    print(_bucket_summary("  ↳ reverse-keyed", signed[(blocks == "likert") & rev], 0))
    print(_bucket_summary("trait_mcq (Sub-axis B)", signed[blocks == "trait_mcq"], (blocks == "trait_mcq").sum()))

    # Inspect any reverse-keyed yielding items individually (the
    # acquiescence-contaminated trio + a few near-relatives).
    print()
    print("Per-item shifts on the most likely Sub-axis C / acquiescence-contaminated F0 Likerts:")
    candidates_substrings = [
        "rather over-correct",
        "thank the person and immediately update",
        "concede some ground",
        "rather say 'I don't know'",
    ]
    rows_seen = 0
    for i_pos, fit_idx in enumerate(f0_idx):
        if blocks[i_pos] != "likert" or not rev[i_pos]:
            continue
        text = fit_items[fit_idx].get("text", "")
        if not any(s in text for s in candidates_substrings):
            continue
        rows_seen += 1
        load = float(load_f0[fit_idx])
        d = float(delta_mean[fit_idx])
        signed_shift = float(signed[i_pos])
        # "Agree → HIGH F0?" — combine reverse-keyed flag with loading sign.
        agree_to_high = (rev[i_pos] and load < 0) or (not rev[i_pos] and load > 0)
        marker = "[Agree→HIGH F0]" if agree_to_high else "[Agree→LOW F0]"
        print(
            f"    loading={load:+.3f} {marker}  "
            f"raw_delta={d:+.3f}  pole-aligned={signed_shift:+.3f}  "
            f"text={text[:90]!r}"
        )
    if rows_seen == 0:
        print("    (no matches — item text didn't contain expected substrings)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        nargs="+",
        default=[
            "conviction_amp_dpo_v6",
            "conviction_sup_dpo_v6",
            "conviction_amp_dpo_v6_singleteacher",
            "conviction_sup_dpo_v6_singleteacher",
        ],
        help="Validation labels to analyse (must be subdirs of scratch/factor_inspect/validate/)",
    )
    args = parser.parse_args()

    if not FA_FIT_NPZ.exists():
        raise FileNotFoundError(
            f"missing {FA_FIT_NPZ}; run inspect_factor_loadings.py first."
        )
    fa = np.load(FA_FIT_NPZ)
    loadings = fa["loadings"]
    fit_items = json.loads(FA_ITEMS.read_text())
    if loadings.shape[0] != len(fit_items):
        raise RuntimeError(
            f"loadings rows ({loadings.shape[0]}) != fit_items ({len(fit_items)})"
        )
    print(f"loaded FA fit: loadings shape {loadings.shape}, items {len(fit_items)}")

    # Load baseline (combined) matrix once.
    base_v5, sids_v5, items_v5 = load_questionnaire_dir(HYDRATE_ROOT / V5_QNAME / "questionnaire")
    base_mcq, sids_mcq, items_mcq = load_questionnaire_dir(HYDRATE_ROOT / MCQ_QNAME / "questionnaire")
    if sids_v5 != sids_mcq:
        common = [s for s in sids_v5 if s in set(sids_mcq)]
        base_v5 = base_v5[[sids_v5.index(s) for s in common]]
        base_mcq = base_mcq[[sids_mcq.index(s) for s in common]]
        sids_v5 = common
    base_items = []
    for it in items_v5:
        d = dict(it); d.setdefault("version", "v5"); base_items.append(d)
    for it in items_mcq:
        d = dict(it); d.setdefault("version", "trait_ocean_natural_v1"); base_items.append(d)
    M_baseline_combined = np.hstack([base_v5, base_mcq])
    print(f"baseline combined: {M_baseline_combined.shape}, sample_ids={len(sids_v5)}")

    # Align baseline columns to fit_items order.
    M_baseline_full = align_to_fit_items(M_baseline_combined, base_items, fit_items)
    sids_baseline_full = sids_v5

    for label in args.label:
        try:
            analyze_one(label, loadings, fit_items, M_baseline_full, sids_baseline_full)
        except FileNotFoundError as e:
            print(f"[skip] {label}: {e}")
        except Exception as e:
            print(f"[error] {label}: {e}")


if __name__ == "__main__":
    main()
