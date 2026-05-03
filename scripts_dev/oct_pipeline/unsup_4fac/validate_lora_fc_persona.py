"""Persona-mediated forced-choice F0 validation + FC↔FA correspondence check.

Companion to ``validate_lora_forced_choice.py`` (which does direct, no-persona
administration). This script administers the same f0_forced_choice_v1
questionnaire to the **same 200-persona subsample** that ``validate_lora.py``
uses for the FA-based F0 scoring. That gives:

  * per-persona FC scores in [-1, +1] (hard +1/-1 per item averaged over 32 items)
  * per-persona FA F0 scores from ``fa_fit.npz`` (baseline) or
    ``validate/<label>/<label>_scores.npz`` (LoRA)

We can then correlate the two across the 200 personas to validate that the
FC questionnaire is measuring the same engaged-agency construct that F0 was
intended to capture (without the acquiescence flip).

Implementation note: the fc_pair admin in
``run_questionnaire_inference_async`` produces hard +1/-1 picks per (persona,
item) based on the top-logprob letter; we don't currently have soft-prob
encoding for fc_pair. With 32 items per persona, mean-over-items has 1/32
granularity which is fine for correlation. To counterbalance position bias,
we administer the questionnaire twice with options swapped and average the
two matrices.

Usage::

    # Baseline (no LoRA)
    uv run python scripts_dev/oct_pipeline/unsup_4fac/validate_lora_fc_persona.py \\
        --label baseline_fc_persona

    # v6 paired-DPO amp
    uv run python scripts_dev/oct_pipeline/unsup_4fac/validate_lora_fc_persona.py \\
        --label conviction_amp_dpo_v6_fc_persona \\
        --adapter persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/conviction/amplifier/vunsup_4fac_paired_dpo_v6/lora/conviction_amplifying_v6_unsup_4fac-dpo

Outputs:
  scratch/factor_inspect/validate_fc_persona/<label>/<label>_summary.json
      summary stats: mean FC score, per-facet means, FC↔FA correlation.
  scratch/factor_inspect/validate_fc_persona/<label>/<label>_scores.npz
      per-persona FC scores (n_personas,) aligned to sample_ids.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

SEED = 436
random.seed(SEED)
np.random.seed(SEED)

import torch  # noqa: E402

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Reuse the rollout-subsampling logic from the FA-based validate_lora.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_lora import (  # noqa: E402
    HF_REPO_ID,
    ROLLOUT_HF_PATH,
    ROLLOUT_LOCAL,
    NUM_CONVERSATION_TURNS,
    QUESTIONNAIRE_MODEL,
    stratified_sample_ids,
    write_subsample_rollout,
)
from src_dev.psychometric.config import QuestionnaireStageConfig  # noqa: E402
from src_dev.psychometric.questionnaire_inference import (  # noqa: E402
    run_questionnaire_inference_async,
)
from src_dev.psychometric.questionnaire_io import load_questionnaire  # noqa: E402
from src_dev.unsupervised_runs.io import hydrate_dataset_subtree  # noqa: E402

DEFAULT_QUESTIONNAIRE = Path("datasets/psychometric_questionnaires/f0_forced_choice_v1.json")
OUT_ROOT = Path("scratch/factor_inspect/validate_fc_persona")

# Where the FA F0 baseline scores live (per-persona, all 2500 personas).
FA_FIT_NPZ = Path("scratch/factor_inspect/fa_fit.npz")
# Where validate_lora.py saves per-LoRA per-persona F0 scores.
FA_VALIDATE_ROOT = Path("scratch/factor_inspect/validate")


# ── FC → fc_pair schema conversion ────────────────────────────────────────


def convert_fc_to_fc_pair_schema(
    fc_qsts: dict,
    swap_AB: bool,
    seed_for_random_order: int = 1234,
) -> dict:
    """Convert the f0_forced_choice_v1 schema into a block_fc_pairs schema.

    For ``swap_AB=False``: high_pole_text → A, low_pole_text → B (high=A).
    For ``swap_AB=True``:  high_pole_text → B, low_pole_text → A (high=B).
    These two orderings are administered separately and then averaged to
    counterbalance position bias.
    """
    items = fc_qsts["block_2_forced_choice"]["items"]

    out_items = []
    for it in items:
        if swap_AB:
            opt_A_text = it["low_pole_text"]
            opt_B_text = it["high_pole_text"]
            high_option = "B"
        else:
            opt_A_text = it["high_pole_text"]
            opt_B_text = it["low_pole_text"]
            high_option = "A"
        out_items.append({
            "id": it["id"],
            "axis": it["facet"],
            "stem": it["prompt"],
            "options": [
                {"label": "A", "text": opt_A_text},
                {"label": "B", "text": opt_B_text},
            ],
            "high_option": high_option,
        })

    converted = {
        "version": fc_qsts["version"] + ("_swapped" if swap_AB else ""),
        "description": fc_qsts.get("description", ""),
        "block_fc_pairs": {
            "prompt_template": (
                "{stem}\n\n"
                "A) {option_a}\n"
                "B) {option_b}\n\n"
                "Reply with just \"A\" or \"B\"."
            ),
            "prefill": "I'd go with ",
            "items": out_items,
        },
    }
    return converted


def write_temp_questionnaire(converted: dict) -> Path:
    """Write the converted questionnaire to a temp file and return its path."""
    tmp = Path(tempfile.mkdtemp(prefix="fc_persona_")) / "f0_fc_pair.json"
    tmp.write_text(json.dumps(converted, indent=2))
    return tmp


# ── Admin one ordering ─────────────────────────────────────────────────────


async def admin_one_ordering(
    *,
    rollout_dir: Path,
    questionnaire_path: Path,
    adapter_path: str | None,
    output_dir: Path,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Administer the FC questionnaire (in fc_pair schema) once.

    Returns (matrix, column_defs, metadata) where matrix is
    (n_personas, n_items), with values in {-1, +1, NaN} (+1 = high pole picked).
    """
    items, column_defs = load_questionnaire(
        questionnaire_path, fa_blocks=("fc_pair",)
    )

    cfg = QuestionnaireStageConfig(
        ctx=None,
        questionnaire_path=questionnaire_path,
        questionnaire_version="f0_forced_choice_v1_fc_pair",
        fa_blocks=("fc_pair",),
        use_logprobs=True,
        phrasing="direct",
        provider="vllm",
        model=QUESTIONNAIRE_MODEL,
        adapter_path=adapter_path,
        max_new_tokens=8,
        max_concurrent=32,
        timeout=120,
        vllm_personas_per_batch=8,
        vllm_gpu_memory_utilization=0.92,
        top_logprobs=20,
    )

    matrix, metadata = await run_questionnaire_inference_async(
        cfg,
        rollout_dir=rollout_dir,
        items=items,
        column_defs=column_defs,
        output_dir=output_dir,
        num_conversation_turns=NUM_CONVERSATION_TURNS,
    )
    return matrix, list(column_defs), metadata


# ── FA F0 score loading ────────────────────────────────────────────────────


def load_fa_f0_scores_for_sample_ids(
    sample_ids: list[str],
    adapter: str | None,
    label: str,
) -> np.ndarray | None:
    """Try to load per-persona FA F0 scores aligned to ``sample_ids``.

    Strategy:
      - If ``label`` ends with ``_fc_persona`` (or matches the
        baseline name), interpret as baseline: pull ``baseline_scores``
        from any existing ``validate/<lora_label>_scores.npz`` (all such
        files store the same 200-persona baseline at the same sample_ids
        because validate_lora.py uses ``seed=436`` for stratified sampling).
      - If ``label`` matches a known LoRA: pull ``lora_scores`` from the
        corresponding ``validate/<lora_label>_scores.npz``.
      - If nothing matches or the file's missing, return ``None``.

    Returns ndarray of shape (n_personas,) of F0 scores aligned to
    ``sample_ids``, or ``None``.
    """
    # Map FC labels to (FA label, score key).
    fc_to_fa = {
        "baseline_fc_persona": (None, "baseline_scores"),  # any LoRA file's baseline
        "conviction_amp_dpo_v6_fc_persona": ("conviction_amp_dpo_v6", "lora_scores"),
        "conviction_sup_dpo_v6_fc_persona": ("conviction_sup_dpo_v6", "lora_scores"),
        "conviction_amp_dpo_v6_fc_persona_singleteacher": ("conviction_amp_dpo_v6_singleteacher", "lora_scores"),
        "conviction_sup_dpo_v6_fc_persona_singleteacher": ("conviction_sup_dpo_v6_singleteacher", "lora_scores"),
    }
    if label not in fc_to_fa:
        print(f"[fc_persona] no FA-label mapping for FC-label {label!r}; skipping FA correlation")
        return None

    fa_label, score_key = fc_to_fa[label]

    # Baseline: try any LoRA's saved npz (they all carry baseline_scores).
    if fa_label is None:
        candidates = [
            "conviction_amp_dpo_v6", "conviction_sup_dpo_v6",
            "conviction_amp_dpo_v6_singleteacher", "conviction_sup_dpo_v6_singleteacher",
        ]
        for cand in candidates:
            npz_path = FA_VALIDATE_ROOT / cand / f"{cand}_scores.npz"
            if npz_path.exists():
                fa_label = cand
                break
        if fa_label is None:
            print("[fc_persona] no validate_lora output found for baseline scores; skipping FA correlation")
            return None
        print(f"[fc_persona] baseline FA scores: pulling from {fa_label}'s baseline_scores")

    npz_path = FA_VALIDATE_ROOT / fa_label / f"{fa_label}_scores.npz"
    if not npz_path.exists():
        print(f"[fc_persona] FA scores not found at {npz_path}; skipping FA correlation")
        return None

    z = np.load(npz_path, allow_pickle=True)
    if "sample_ids" not in z.files or score_key not in z.files:
        print(f"[fc_persona] {npz_path} missing 'sample_ids' or '{score_key}'; skipping FA correlation")
        return None
    fa_sids = list(z["sample_ids"])
    fa_scores = z[score_key]  # (n_personas, k); F0 is column 0
    sid_to_row = {s: r for r, s in enumerate(fa_sids)}
    rows: list[int] = []
    missing = 0
    for s in sample_ids:
        if s not in sid_to_row:
            missing += 1
            rows.append(-1)
        else:
            rows.append(sid_to_row[s])
    if missing == len(sample_ids):
        print(f"[fc_persona] all sample_ids missing from {npz_path}; skipping correlation")
        return None
    aligned = np.full(len(sample_ids), np.nan)
    for i, r in enumerate(rows):
        if r >= 0:
            aligned[i] = float(fa_scores[r, 0])
    if missing > 0:
        print(f"[fc_persona] {missing}/{len(sample_ids)} sample_ids missing from FA scores")
    return aligned


# ── Main ───────────────────────────────────────────────────────────────────


async def main_async(args: argparse.Namespace) -> None:
    out_dir = OUT_ROOT / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    fc_qsts = json.loads(Path(args.questionnaire).read_text())
    n_items = len(fc_qsts["block_2_forced_choice"]["items"])
    print(f"[fc_persona] loaded {n_items} FC items from {args.questionnaire}")

    # Hydrate rollout dir if needed.
    if not ROLLOUT_LOCAL.exists() or not any(ROLLOUT_LOCAL.rglob("*.jsonl")):
        print(f"[fc_persona] hydrating rollout dir: {ROLLOUT_HF_PATH}")
        hydrate_dataset_subtree(
            repo_id=HF_REPO_ID,
            path_in_repo=ROLLOUT_HF_PATH,
            local_dir=ROLLOUT_LOCAL,
            required=True,
        )

    # Stratified subsample (same seed as validate_lora.py).
    sample_ids = stratified_sample_ids(ROLLOUT_LOCAL, args.n_personas, SEED)
    sample_ids_set = set(sample_ids)
    print(f"[fc_persona] subsampling {len(sample_ids)} personas")

    sub_dir = out_dir / "rollout_sub"
    write_subsample_rollout(ROLLOUT_LOCAL, sub_dir, sample_ids_set)

    # Two orderings (high as A, then high as B).
    all_matrices: list[np.ndarray] = []
    all_meta: list[list[dict]] = []
    item_id_order: list[str] | None = None

    for swap_AB, ord_label in [(False, "high_as_A"), (True, "high_as_B")]:
        print(f"\n[fc_persona] ordering: {ord_label}")
        converted = convert_fc_to_fc_pair_schema(fc_qsts, swap_AB=swap_AB)
        tmp_qsts = write_temp_questionnaire(converted)
        ord_out = out_dir / f"questionnaire_{ord_label}"
        matrix, cols, meta = await admin_one_ordering(
            rollout_dir=sub_dir,
            questionnaire_path=tmp_qsts,
            adapter_path=args.adapter,
            output_dir=ord_out,
        )
        all_matrices.append(matrix)
        all_meta.append(meta)
        if item_id_order is None:
            item_id_order = [c["item_id"] for c in cols]
        print(f"[fc_persona] ordering {ord_label}: matrix shape {matrix.shape}")

    # Both matrices are encoded so +1 = high pole picked. Average element-wise.
    M_high = np.where(np.isnan(all_matrices[0]), all_matrices[1], all_matrices[0])
    M_low = np.where(np.isnan(all_matrices[1]), all_matrices[0], all_matrices[1])
    M_avg = np.nanmean(np.stack([all_matrices[0], all_matrices[1]], axis=0), axis=0)
    # M_avg is in [-1, +1]; +1 = picked high in both orderings.

    # Per-persona FC score = mean across items in [-1, +1].
    per_persona_fc = np.nanmean(M_avg, axis=1)  # (n_personas,)
    n_valid_personas = int(np.isfinite(per_persona_fc).sum())
    print(f"\n[fc_persona] per-persona FC scores: {per_persona_fc.shape}, "
          f"valid={n_valid_personas}/{len(per_persona_fc)}")
    print(f"  mean over personas: {np.nanmean(per_persona_fc):+.4f}")
    print(f"  std  over personas: {np.nanstd(per_persona_fc):.4f}")

    # Pull sample_ids in matrix-row order from the metadata.
    meta_sids = [m["sample_id"] for m in all_meta[0]]

    # Per-facet aggregation.
    items = fc_qsts["block_2_forced_choice"]["items"]
    item_to_facet = {it["id"]: it["facet"] for it in items}
    facet_to_cols: dict[str, list[int]] = {}
    for col_idx, item_id in enumerate(item_id_order):
        facet_to_cols.setdefault(item_to_facet[item_id], []).append(col_idx)

    per_facet_means: dict[str, float] = {}
    for facet, cols_list in facet_to_cols.items():
        sub = M_avg[:, cols_list]
        per_facet_means[facet] = float(np.nanmean(sub))

    # Optional FA F0 comparison.
    fa_f0_aligned = load_fa_f0_scores_for_sample_ids(meta_sids, args.adapter, args.label)
    correlation = None
    if fa_f0_aligned is not None:
        good = np.isfinite(per_persona_fc) & np.isfinite(fa_f0_aligned)
        if good.sum() >= 5:
            r = float(np.corrcoef(per_persona_fc[good], fa_f0_aligned[good])[0, 1])
            correlation = {
                "n_personas": int(good.sum()),
                "pearson_r": r,
                "fc_mean": float(np.mean(per_persona_fc[good])),
                "fa_f0_mean": float(np.mean(fa_f0_aligned[good])),
            }
            print(f"\n[fc_persona] FC↔FA F0 correlation: r={r:+.3f}  "
                  f"(n={int(good.sum())}, fc_mean={correlation['fc_mean']:+.3f}, "
                  f"fa_mean={correlation['fa_f0_mean']:+.3f})")
        else:
            print("[fc_persona] not enough valid pairs for correlation")

    # Save scores npz + summary json.
    scores_path = out_dir / f"{args.label}_scores.npz"
    np.savez(
        scores_path,
        sample_ids=np.array(meta_sids),
        fc_per_persona=per_persona_fc,
        fa_f0_per_persona=fa_f0_aligned if fa_f0_aligned is not None else np.array([]),
        item_ids=np.array(item_id_order),
    )

    summary = {
        "label": args.label,
        "adapter": args.adapter,
        "questionnaire_version": fc_qsts["version"],
        "n_personas_target": args.n_personas,
        "n_personas_valid": n_valid_personas,
        "n_items": n_items,
        "fc_score_overall_mean": float(np.nanmean(per_persona_fc)),
        "fc_score_overall_std": float(np.nanstd(per_persona_fc)),
        "fc_score_by_facet": per_facet_means,
        "fc_fa_correlation": correlation,
        "scores_npz_path": str(scores_path),
    }
    summary_path = out_dir / f"{args.label}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[fc_persona] wrote {summary_path}")
    print(f"[fc_persona] wrote {scores_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--questionnaire", default=str(DEFAULT_QUESTIONNAIRE))
    parser.add_argument("--n-personas", type=int, default=200)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
