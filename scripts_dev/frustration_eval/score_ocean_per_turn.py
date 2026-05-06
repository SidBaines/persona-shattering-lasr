"""Score per-turn OCEAN traits on existing frustration_eval rollouts.

For each (condition, conversation, assistant turn), call the OCEAN v2 judge for
all 5 traits using the immediately-preceding user message as the question.
Caches per-prompt scores locally and on the HF monorepo so reruns rehydrate.

Usage
-----
    uv run python -m scripts_dev.frustration_eval.score_ocean_per_turn \\
        --prompt-idx 3 \\
        --upload

Outputs
-------
Local JSONL per (run_name, prompt_hash):
    scratch/frustration_eval/<run_name>/ocean_per_turn/<prompt_hash>.jsonl

HF mirror (with --upload):
    persona-shattering-lasr/monorepo:
      evals/frustration_eval/<run_name>/ocean_per_turn/<prompt_hash>.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.metrics.ocean_v2 import (
    AgreeablenessV2Evaluation,
    ConscientiousnessV2Evaluation,
    ExtraversionV2Evaluation,
    NeuroticismV2Evaluation,
    OpennessV2Evaluation,
)
from src_dev.utils.hf_hub import (
    download_file_from_dataset_repo,
    upload_file_to_dataset_repo,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

logger = logging.getLogger(__name__)

HF_REPO = "persona-shattering-lasr/monorepo"
LOCAL_ROOT = Path("scratch/frustration_eval/evals/frustration_eval")
DEFAULT_RUNS: dict[str, str] = {
    "base":     "gemma3_27b_base_8turn_100prompt_1rollout",
    "n_minus":  "gemma3_27b_n_minus_vanton4_paired_dpo_persona_8turn_100prompt_1rollout",
    "control":  "gemma3_27b_control_vanton4_paired_dpo_s1vs2_persona_8turn_100prompt_1rollout",
    "n_neg":    "gemma3_27b_n_minus_vanton4_paired_dpo_persona_negscale_8turn_100prompt_1rollout",
    "axiscap":  "gemma3_27b_n_minus_axiscap_recommended_p1p0_8turn_10prompt_1rollout",
}
CATEGORY = "impossible_numeric_3turn"

TRAIT_CLASSES = [
    OpennessV2Evaluation,
    ConscientiousnessV2Evaluation,
    ExtraversionV2Evaluation,
    AgreeablenessV2Evaluation,
    NeuroticismV2Evaluation,
]
TRAIT_NAMES = [c.name for c in TRAIT_CLASSES]  # e.g. "openness_v2"


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def hf_path(run_name: str, p_hash: str) -> str:
    return f"evals/frustration_eval/{run_name}/ocean_per_turn/{p_hash}.jsonl"


def local_path(run_name: str, p_hash: str) -> Path:
    return Path(f"scratch/frustration_eval/{run_name}/ocean_per_turn/{p_hash}.jsonl")


def results_path(run_name: str) -> Path:
    return LOCAL_ROOT / run_name / CATEGORY / "results.jsonl"


def load_conversation(run_name: str, prompt: str) -> dict | None:
    """Load the unique conversation matching `prompt` from a run's results.jsonl."""
    p = results_path(run_name)
    if not p.exists():
        raise FileNotFoundError(f"Run results not found locally: {p}")
    matches = []
    with open(p) as f:
        for line in f:
            c = json.loads(line)
            if c.get("prompt") == prompt:
                matches.append(c)
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning("%s: %d conversations match prompt; using first.", run_name, len(matches))
    return matches[0]


def previous_user_message(messages: list[dict], turn_index: int) -> str:
    """Return the user message immediately preceding assistant turn `turn_index`.

    The frustration_eval format alternates user/assistant starting with user, so
    the preceding user message for assistant turn t is messages[2*t].
    """
    user_idx = 2 * turn_index
    if user_idx >= len(messages) or messages[user_idx].get("role") != "user":
        # Fallback: search backwards from the assistant message position
        # for the nearest user message.
        for j in range(min(user_idx, len(messages) - 1), -1, -1):
            if messages[j].get("role") == "user":
                return messages[j]["content"]
        return ""
    return messages[user_idx]["content"]


@dataclass
class TurnRow:
    run_name: str
    condition: str
    prompt: str
    prompt_hash: str
    turn_index: int
    user_message: str
    response: str
    scores: dict[str, int]
    reasonings: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "condition": self.condition,
            "prompt": self.prompt,
            "prompt_hash": self.prompt_hash,
            "turn_index": self.turn_index,
            "user_message": self.user_message,
            "response": self.response,
            "scores": self.scores,
            "reasonings": self.reasonings,
        }


async def score_one_conversation(
    *,
    judges: dict[str, object],
    run_name: str,
    condition: str,
    conversation: dict,
) -> list[TurnRow]:
    """Score every assistant turn for all 5 OCEAN traits."""
    prompt = conversation["prompt"]
    p_hash = prompt_hash(prompt)
    messages = conversation["messages"]
    turns = conversation["turn_results"]

    rows: list[TurnRow] = [
        TurnRow(
            run_name=run_name,
            condition=condition,
            prompt=prompt,
            prompt_hash=p_hash,
            turn_index=t["turn_index"],
            user_message=previous_user_message(messages, t["turn_index"]),
            response=t["response"],
            scores={},
            reasonings={},
        )
        for t in turns
    ]

    # Skip turns that are generation errors.
    judgable = [r for r in rows if not r.response.startswith("[GENERATION ERROR")]
    if not judgable:
        return rows

    # For each trait, batch-judge all turns of this conversation.
    for trait_name, judge in judges.items():
        results = await judge.evaluate_batch_async(
            responses=[r.response for r in judgable],
            questions=[r.user_message for r in judgable],
        )
        score_key = f"{trait_name}.score"
        reason_key = f"{trait_name}.reasoning"
        for r, res in zip(judgable, results):
            r.scores[trait_name] = int(res.get(score_key, 0))
            r.reasonings[trait_name] = str(res.get(reason_key, ""))

    return rows


def write_jsonl(rows: list[TurnRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict()) + "\n")


def try_rehydrate(run_name: str, p_hash: str) -> Path | None:
    """Return local path if cached scores exist (locally or on HF); else None.

    On HF hit, the file lands at ``<staging_dir>/<full_path_in_repo>``; we copy
    it to the canonical local cache location for predictable downstream reads.
    """
    out = local_path(run_name, p_hash)
    if out.exists() and out.stat().st_size > 0:
        return out
    staging = Path(f"scratch/frustration_eval/_hf_dl/{run_name}_{p_hash}")
    try:
        downloaded = download_file_from_dataset_repo(
            repo_id=HF_REPO,
            path_in_repo=hf_path(run_name, p_hash),
            local_dir=staging,
        )
        if downloaded.exists() and downloaded.stat().st_size > 0:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(downloaded.read_bytes())
            return out
    except Exception as exc:
        logger.debug("HF rehydrate miss for %s/%s: %s", run_name, p_hash, exc)
    return None


def resolve_prompts(args: argparse.Namespace, runs: dict[str, str]) -> list[str]:
    """Return the list of prompt strings to score, in order."""
    if args.prompt_text:
        return [args.prompt_text]
    if args.shared_axiscap:
        # Use the 10 prompts present in the axiscap run as the canonical shared set.
        ax_path = results_path(DEFAULT_RUNS["axiscap"])
        return [json.loads(l)["prompt"] for l in open(ax_path)]
    if args.prompt_idxs:
        base_path = results_path(runs.get("base", DEFAULT_RUNS["base"]))
        base_convs = [json.loads(l) for l in open(base_path)]
        return [base_convs[i]["prompt"] for i in args.prompt_idxs]
    if args.prompt_idx is not None:
        base_path = results_path(runs.get("base", DEFAULT_RUNS["base"]))
        base_convs = [json.loads(l) for l in open(base_path)]
        return [base_convs[args.prompt_idx]["prompt"]]
    raise SystemExit(
        "Must pass one of --prompt-idx, --prompt-idxs, --prompt-text, or --shared-axiscap."
    )


async def main_async(args: argparse.Namespace) -> None:
    runs = {k: DEFAULT_RUNS[k] for k in args.conditions} if args.conditions else DEFAULT_RUNS
    prompts = resolve_prompts(args, runs)
    logger.info("Scoring %d prompt(s) × %d condition(s).", len(prompts), len(runs))

    judge_cfg = JudgeLLMConfig(
        provider=args.judge_provider,
        model=args.judge_model,
        max_concurrent=args.max_concurrent,
        temperature=0.0,
        max_tokens=512,
    )
    judges = {cls.name: cls(judge_config=judge_cfg) for cls in TRAIT_CLASSES}

    all_rows: list[TurnRow] = []
    summary_dir = Path("scratch/frustration_eval/_combined")
    summary_dir.mkdir(parents=True, exist_ok=True)

    for pi, target_prompt in enumerate(prompts):
        p_hash = prompt_hash(target_prompt)
        logger.info("[prompt %d/%d hash=%s] %r",
                    pi + 1, len(prompts), p_hash, target_prompt[:80])

        for cond, run_name in runs.items():
            cached = try_rehydrate(run_name, p_hash)
            if cached and not args.force:
                with open(cached) as f:
                    rows = [TurnRow(**json.loads(l)) for l in f]
                logger.info("  [%s] rehydrated %d turns from %s", cond, len(rows), cached)
                all_rows.extend(rows)
                continue

            conv = load_conversation(run_name, target_prompt)
            if conv is None:
                logger.warning("  [%s] no conversation matches prompt; skipping.", cond)
                continue

            t0 = time.time()
            rows = await score_one_conversation(
                judges=judges, run_name=run_name, condition=cond, conversation=conv,
            )
            logger.info("  [%s] judged %d turns × %d traits in %.1fs",
                        cond, len(rows), len(TRAIT_NAMES), time.time() - t0)

            out = local_path(run_name, p_hash)
            write_jsonl(rows, out)
            if args.upload:
                url = upload_file_to_dataset_repo(
                    local_path=out,
                    repo_id=HF_REPO,
                    path_in_repo=hf_path(run_name, p_hash),
                    commit_message=f"OCEAN per-turn scores for {run_name} prompt={p_hash}",
                )
                logger.info("  [%s] uploaded → %s", cond, url)
            all_rows.extend(rows)

        # Per-prompt combined dump (kept for backward compatibility / per-prompt plots).
        per_prompt_rows = [r for r in all_rows if r.prompt_hash == p_hash]
        write_jsonl(per_prompt_rows, summary_dir / f"ocean_per_turn_{p_hash}.jsonl")

    # Master combined dump across all scored prompts (for averaging).
    if len(prompts) > 1:
        tag = f"{len(prompts)}prompts_" + hashlib.sha256(
            "|".join(prompt_hash(p) for p in prompts).encode()
        ).hexdigest()[:8]
        master = summary_dir / f"ocean_per_turn_multi_{tag}.jsonl"
        write_jsonl(all_rows, master)
        logger.info("Master combined %d rows → %s", len(all_rows), master)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-idx", type=int, default=None,
                    help="Single index into the base run's results.jsonl.")
    ap.add_argument("--prompt-idxs", type=int, nargs="*", default=None,
                    help="Multiple indices into the base run's results.jsonl.")
    ap.add_argument("--prompt-text", type=str, default=None,
                    help="Override prompt text directly (single prompt).")
    ap.add_argument("--shared-axiscap", action="store_true",
                    help="Use the 10 prompts present in the axiscap run (the set common to all 5 conditions).")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help=f"Subset of {list(DEFAULT_RUNS)} (default: all).")
    ap.add_argument("--judge-provider", default="openrouter")
    ap.add_argument("--judge-model", default="qwen/qwen3-235b-a22b-2507")
    ap.add_argument("--max-concurrent", type=int, default=20)
    ap.add_argument("--upload", action="store_true",
                    help="Upload per-condition score files to HF monorepo.")
    ap.add_argument("--force", action="store_true",
                    help="Re-judge even if cached scores exist.")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
