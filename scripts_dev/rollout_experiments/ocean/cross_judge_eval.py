"""Re-evaluate existing rollouts with an additional judge metric.

Use case: a cross-LoRA cell was originally judged with the LoRA's own
trait metric (e.g. c_minus -> conscientiousness_v2) but for the
cross-trait correlation analysis we also need the extraversion judge
applied to the same rollouts. This script downloads existing rollouts
from HF, runs the requested extra judge(s), and re-uploads the merged
rollouts_evaluated.jsonl back to HF.

Usage:
    uv run python scripts_dev/rollout_experiments/ocean/cross_judge_eval.py \\
        --hf-prefix fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/vanton4_paired_dpo/rollouts/rollout_sweep_lora_t0.7_crossLoRA \\
        --variant-pattern "scale_+0.25,scale_+0.50,scale_+0.75,scale_+1.00" \\
        --condition baseline \\
        --add-evaluations extraversion_v2

The script:
  1. For each variant under --hf-prefix/<variant>/<condition>/, downloads
     rollouts/rollouts.jsonl + per_message_metrics.jsonl + rollouts_evaluated.jsonl
  2. Runs the new evaluation on those rollouts
  3. Merges new scores into rollouts_evaluated.jsonl (preserves existing scores)
  4. Updates per_message_metrics.jsonl with the new metric rows
  5. Re-uploads the modified files to the same HF path
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from src_dev.persona_metrics.conversation_eval import (
    ConversationMetricsConfig,
    MessageSelector,
    run_conversation_metrics_async,
)
from src_dev.utils.hf_hub import (
    download_from_dataset_repo,
    login_from_env,
    upload_folder_to_dataset_repo,
)

load_dotenv()

HF_REPO = "persona-shattering-lasr/monorepo"


def _merge_scores_into_evaluated(
    evals_dir: Path,
    new_scores_by_msg: dict[str, dict],
) -> None:
    """Merge new per-message scores into rollouts_evaluated.jsonl in place.

    Args:
        evals_dir: Directory containing rollouts_evaluated.jsonl.
        new_scores_by_msg: Map of message_id -> nested score dict (e.g.
            {"extraversion_v2": {"score": 3, ...}}).
    """
    eval_path = evals_dir / "rollouts_evaluated.jsonl"
    if not eval_path.exists():
        print(f"  WARN: {eval_path} not found; skipping merge")
        return
    entries = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]
    n_merged = 0
    for entry in entries:
        for r_idx, msgs in entry.get("messages", {}).items():
            for msg in msgs:
                # The evaluated file doesn't preserve message_id directly;
                # we match by (seed_id, rollout_idx, role, turn_index).
                # But cleaner: the sample that produced this msg has a
                # deterministic message_id; we use a (seed, role, turn_index)
                # index built from the new scores when we collect them.
                key = (entry["seed_id"], r_idx, msg.get("role"), msg.get("turn_index"))
                if key in new_scores_by_msg:
                    existing = msg.setdefault("scores", {})
                    existing.update(new_scores_by_msg[key])
                    n_merged += 1
    eval_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    print(f"  Merged {n_merged} new score rows into {eval_path.name}")


def _build_keyed_scores(
    per_message_scores: list[dict],
    samples_by_msg_id: dict[str, dict],
    new_metric_names: list[str],
) -> dict[tuple, dict]:
    """Return {(seed_id, rollout_idx, role, turn_index): {metric_name: score_dict}}.

    Maps each message_id back to its (seed, rollout, role, turn) coordinates
    so we can merge into rollouts_evaluated.jsonl.
    """
    keyed: dict[tuple, dict] = {}
    for item in per_message_scores:
        msg_id = item["message_id"]
        sample_info = samples_by_msg_id.get(msg_id)
        if not sample_info:
            continue
        key = sample_info  # already a tuple
        flat = item.get("scores", {})
        # Re-nest scores by metric name (mirrors _nest_scores in sweep.py).
        nested: dict[str, dict] = {}
        for full_key, val in flat.items():
            if "." in full_key:
                metric, attr = full_key.split(".", 1)
            else:
                metric, attr = full_key, "score"
            if metric not in new_metric_names:
                continue
            nested.setdefault(metric, {})[attr] = val
        if nested:
            keyed[key] = nested
    return keyed


def _index_samples_by_msg_id(run_dir: Path) -> dict[str, tuple]:
    """Walk samples and return {message_id: (seed_id, rollout_idx, role, turn_index)}."""
    from src_dev.datasets import load_samples, materialize_canonical_samples
    materialize_canonical_samples(run_dir)
    samples = load_samples(run_dir)
    out: dict[str, tuple] = {}
    for s in samples:
        seed_id = s.input_group_id or s.sample_id
        for msg in s.messages:
            meta = msg.message_metadata or {}
            if meta.get("source_stage") == "seed":
                continue
            out[msg.message_id] = (
                seed_id,
                str(s.response_index),
                msg.role,
                meta.get("turn_index"),
            )
    return out


async def re_eval_one_variant(
    hf_subpath: str,
    new_evaluations: list[str],
    local_root: Path,
) -> None:
    """Re-eval one variant dir on HF: download, judge, merge, upload."""
    print(f"\n=== {hf_subpath} ===")
    local_root.mkdir(parents=True, exist_ok=True)

    # Download the run_dir contents we need. download_from_dataset_repo
    # replicates the repo path under local_root, so the actual run_dir is
    # at local_root/<hf_subpath>/.
    print(f"  Downloading {hf_subpath} into {local_root}/{hf_subpath}")
    download_from_dataset_repo(
        repo_id=HF_REPO,
        path_in_repo=hf_subpath,
        local_dir=local_root,
    )
    local_dir = local_root / hf_subpath

    rollouts_jsonl = local_dir / "rollouts" / "rollouts.jsonl"
    if not rollouts_jsonl.exists():
        print(f"  ERR: {rollouts_jsonl} not found; skipping")
        return

    # Sanity: how many entries
    n_entries = sum(1 for _ in rollouts_jsonl.open())
    print(f"  Found {n_entries} rollout entries")

    # Run the new evaluation. ConversationMetricsConfig writes
    # per_message_metrics.jsonl into local_dir.
    config = ConversationMetricsConfig(
        evaluations=new_evaluations,
        run_dir=local_dir,
        message_selector=MessageSelector(exclude_seed=True),
        output_path=local_dir / "per_message_metrics_extra.jsonl",
    )
    print(f"  Running judges: {new_evaluations}")
    result = await run_conversation_metrics_async(config)
    print(
        f"  -> Evaluated {result.num_messages_evaluated} messages "
        f"across {result.num_conversations} conversations"
    )

    # Build {msg_id: (seed, rollout, role, turn)} index from materialized samples.
    samples_by_msg = _index_samples_by_msg_id(local_dir)

    # Build keyed scores by message (seed, rollout, role, turn) -> {metric: score_dict}.
    keyed = _build_keyed_scores(
        result.per_message_scores, samples_by_msg, new_evaluations
    )
    print(f"  Built {len(keyed)} keyed score rows for merge")

    # Merge into rollouts_evaluated.jsonl
    evals_dir = local_dir / "evals"
    _merge_scores_into_evaluated(evals_dir, keyed)

    # Upload modified evals dir + new per_message_metrics file.
    print(f"  Uploading merged evals back to {hf_subpath}/evals/")
    upload_folder_to_dataset_repo(
        repo_id=HF_REPO,
        local_dir=evals_dir,
        path_in_repo=f"{hf_subpath}/evals",
        commit_message=f"add {','.join(new_evaluations)} judge to {hf_subpath}",
    )
    print(f"  Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-prefix",
        required=True,
        help="HF dir under monorepo root containing per-variant subdirs (no 'datasets/' prefix).",
    )
    parser.add_argument(
        "--variant-pattern",
        required=True,
        help="Comma-separated variant names (e.g. 'scale_+0.25,scale_+0.50,...').",
    )
    parser.add_argument(
        "--condition",
        required=True,
        help="Condition subdir name under each variant (e.g. 'baseline').",
    )
    parser.add_argument(
        "--add-evaluations",
        required=True,
        help="Comma-separated evaluation metric names to add (e.g. 'extraversion_v2').",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path("scratch/cross_judge_reeval"),
        help="Local download root.",
    )
    return parser.parse_args()


async def amain() -> None:
    args = parse_args()
    login_from_env()

    variants = [v.strip() for v in args.variant_pattern.split(",") if v.strip()]
    new_evals = [e.strip() for e in args.add_evaluations.split(",") if e.strip()]

    print(f"HF prefix: {args.hf_prefix}")
    print(f"Variants: {variants}")
    print(f"Adding evaluations: {new_evals}")
    print(f"Local root: {args.local_root}")

    for variant in variants:
        sub = f"{args.hf_prefix}/{variant}/{args.condition}"
        try:
            await re_eval_one_variant(sub, new_evals, args.local_root)
        except Exception as e:
            print(f"  FAILED on {sub}: {e.__class__.__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(amain())
