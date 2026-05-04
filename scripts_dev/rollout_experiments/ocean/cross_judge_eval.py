"""Re-evaluate existing rollouts with an additional judge metric, in place.

Use case: a cross-LoRA cell was originally judged with the LoRA's own
trait metric (e.g. c_minus -> conscientiousness_v2) but for the
cross-trait correlation analysis we also need the extraversion judge
applied to the same rollouts. This script reads the existing
rollouts_evaluated.jsonl, runs the requested extra judge directly on
each assistant message, merges the new scores in, and re-uploads.

Bypasses the canonical-samples / event-store machinery — works directly
on the export format.

Usage:
    uv run python scripts_dev/rollout_experiments/ocean/cross_judge_eval.py \\
        --hf-prefix fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/vanton4_paired_dpo/rollouts/rollout_sweep_lora_t0.7_crossLoRA \\
        --variant-pattern "scale_+0.25,scale_+0.50,scale_+0.75,scale_+1.00" \\
        --condition baseline \\
        --add-evaluations extraversion_v2

For each variant the script:
  1. Downloads only evals/rollouts_evaluated.jsonl from HF.
  2. For each assistant message, calls the new judge with the seed_input
     as the "question" and the message content as the "response".
  3. Merges the new score dict (e.g. {"score": 3, "reasoning": "..."})
     into the message's existing scores under the new metric name.
  4. Re-uploads the modified rollouts_evaluated.jsonl back to HF.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import src_dev.persona_metrics.metrics  # noqa: F401  (side-effect: registers all metrics)
from src_dev.persona_metrics.registry import get_persona_metric
from src_dev.utils.hf_hub import (
    download_from_dataset_repo,
    login_from_env,
    upload_folder_to_dataset_repo,
)

load_dotenv()

HF_REPO = "persona-shattering-lasr/monorepo"


async def re_eval_one_variant(
    hf_subpath: str,
    new_metric_names: list[str],
    local_root: Path,
) -> None:
    """Download evals, judge missing metrics on every assistant message, re-upload."""
    print(f"\n=== {hf_subpath} ===")
    local_root.mkdir(parents=True, exist_ok=True)

    # Download just the evals/ subdir.
    evals_subpath = f"{hf_subpath}/evals"
    print(f"  Downloading {evals_subpath}/rollouts_evaluated.jsonl")
    download_from_dataset_repo(
        repo_id=HF_REPO,
        path_in_repo=evals_subpath,
        local_dir=local_root,
        allow_patterns=["rollouts_evaluated.jsonl"],
    )
    evals_dir = local_root / evals_subpath
    eval_path = evals_dir / "rollouts_evaluated.jsonl"
    if not eval_path.exists():
        print(f"  ERR: {eval_path} not found; skipping")
        return

    # Load entries
    entries = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]
    print(f"  Loaded {len(entries)} entries from rollouts_evaluated.jsonl")

    # Build flat list of (entry_idx, rollout_key, msg_idx, question, response)
    # for all assistant messages that don't already have *all* the new metrics.
    work: list[tuple[int, str, int, str, str]] = []
    for e_idx, entry in enumerate(entries):
        seed_input = entry.get("seed_input") or ""
        for r_idx, msgs in entry.get("messages", {}).items():
            for m_idx, msg in enumerate(msgs):
                if msg.get("role") != "assistant":
                    continue
                existing = msg.get("scores", {}) or {}
                missing = [m for m in new_metric_names if m not in existing]
                if not missing:
                    continue
                content = msg.get("content", "") or ""
                if not content.strip():
                    continue
                work.append((e_idx, r_idx, m_idx, seed_input, content))
    print(f"  {len(work)} assistant messages need scoring with {new_metric_names}")
    if not work:
        print("  Nothing to do; skipping upload")
        return

    # For each new metric, batch-evaluate.
    for metric_name in new_metric_names:
        print(f"  Running judge: {metric_name}")
        metric = get_persona_metric(metric_name)
        responses = [row[4] for row in work]
        questions = [row[3] for row in work]
        scores = await metric.evaluate_batch_async(responses, questions)
        if len(scores) != len(work):
            print(f"  WARN: judge returned {len(scores)} scores for {len(work)} requests")
        # Merge results
        n_merged = 0
        for (e_idx, r_idx, m_idx, _q, _r), score_dict in zip(work, scores):
            msg = entries[e_idx]["messages"][r_idx][m_idx]
            existing = msg.setdefault("scores", {})
            # score_dict from evaluate_batch_async is flat: {"<metric>.score": N, "<metric>.reasoning": "..."}
            nested: dict[str, Any] = {}
            for full_key, val in score_dict.items():
                if "." in full_key:
                    metric_root, attr = full_key.split(".", 1)
                else:
                    metric_root, attr = full_key, "score"
                if metric_root != metric_name:
                    continue
                nested[attr] = val
            if nested:
                existing[metric_name] = nested
                n_merged += 1
        print(f"    merged {n_merged}/{len(work)} score rows for {metric_name}")

    # Write merged eval file back
    eval_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    print(f"  Wrote merged file to {eval_path}")

    # Upload
    print(f"  Uploading merged evals to {evals_subpath}")
    upload_folder_to_dataset_repo(
        repo_id=HF_REPO,
        local_dir=evals_dir,
        path_in_repo=evals_subpath,
        commit_message=f"add {','.join(new_metric_names)} judge to {hf_subpath}",
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
