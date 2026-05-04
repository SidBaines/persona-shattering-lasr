"""Frustration eval using vLLM batched generation.

Bakes a LoRA adapter into base weights (with optional ``--negate-adapter``
for sign-flipped adapter direction via PEFT scaling=-1), then runs the
multi-turn frustration eval against an in-process ``vllm.LLM``. Each turn
is batched across all conversations via ``llm.chat()`` continuous batching,
giving ~10x speedup over ``run_local_adapter.py`` (which calls ``model.generate``
on one conversation at a time).

Usage:
    uv run python -m scripts_dev.frustration_eval.run_vllm \\
        --base-model google/gemma-3-27b-it \\
        --adapter-path scratch/adapters/gemma27b_n_minus/.../neuroticism_suppressing_full_vanton4-persona \\
        --negate-adapter \\
        --num-prompts 10 --num-rollouts 1 --num-turns 8 \\
        --run-name gemma27b_n_minus_negscale_vllm_8turn_10prompt_1rollout
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import gc
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from scripts_dev.frustration_eval.frustration_judge import FrustrationJudge
from scripts_dev.frustration_eval.prompts import IMPOSSIBLE_NUMERIC_3TURN
from scripts_dev.frustration_eval.run_eval import (
    ConversationResult,
    TurnResult,
    compute_summary,
    score_conversation,
)
from src_dev.inference.config import (
    GenerationConfig,
    InferenceConfig,
    VllmProviderConfig,
)
from src_dev.inference.providers import get_provider
from src_dev.persona_metrics.config import JudgeLLMConfig

SEED = 42
logger = logging.getLogger(__name__)


def _bake_merged(
    base_model: str,
    adapter_path: Path,
    negate: bool,
    out_dir: Path,
) -> Path:
    """Bake adapter@scale into base weights and save to ``out_dir``.

    Idempotent: skips if ``out_dir`` already has weight shards. ``negate=True``
    inverts the adapter via PEFT's per-module ``scaling`` dict (correct
    inversion; flipping both ``lora_A`` and ``lora_B`` would cancel out).
    """
    if (out_dir / "model.safetensors.index.json").exists() or any(out_dir.glob("*.safetensors")):
        logger.info("merged model already at %s", out_dir)
        return out_dir

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading base %s (bf16)", base_model)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    pm = PeftModel.from_pretrained(base, str(adapter_path), adapter_name="adapter")
    if negate:
        logger.info("inverting adapter via per-module LoRA scaling=-1.0")
        for module in pm.modules():
            sc = getattr(module, "scaling", None)
            if isinstance(sc, dict):
                for k in sc:
                    sc[k] = -1.0
    logger.info("merge_and_unload ...")
    merged = pm.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("saving merged model to %s", out_dir)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(str(out_dir))

    del merged, pm, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_dir


def run_rollouts_vllm(
    provider,
    category,
    *,
    num_prompts: int,
    num_rollouts: int,
    num_turns: int,
    rng: random.Random,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
) -> list[ConversationResult]:
    """Drive the multi-turn frustration eval, batching all conversations
    at each turn step (vLLM continuous batching does the heavy lifting)."""
    convs: list[dict] = []
    for prompt in category.prompts[:num_prompts]:
        for rollout_id in range(num_rollouts):
            convs.append({
                "prompt": prompt,
                "rollout_id": rollout_id,
                "messages": [{"role": "user", "content": prompt}],
                "turns": [],
            })

    for turn_idx in range(num_turns):
        msgs_batch = [c["messages"] for c in convs]
        logger.info("turn %d/%d: batched gen on %d convs", turn_idx + 1, num_turns, len(msgs_batch))
        responses = provider.generate_batch(
            msgs_batch,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=1.0,
        )
        for c, resp in zip(convs, responses):
            c["messages"].append({"role": "assistant", "content": resp})
            c["turns"].append(TurnResult(turn_index=turn_idx, response=resp))
            if turn_idx < num_turns - 1:
                rejection = rng.choice(category.rejection_pool)
                c["messages"].append({"role": "user", "content": rejection})

    return [
        ConversationResult(
            category=category.name,
            prompt=c["prompt"],
            rollout_id=c["rollout_id"],
            messages=c["messages"],
            turn_results=c["turns"],
        )
        for c in convs
    ]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="google/gemma-3-27b-it")
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--negate-adapter", action="store_true", default=False)
    ap.add_argument("--num-turns", type=int, default=8)
    ap.add_argument("--num-rollouts", type=int, default=1)
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--output-dir", default="scratch/evals/frustration_eval")
    ap.add_argument("--judge-provider", default="openrouter")
    ap.add_argument("--judge-model", default="anthropic/claude-sonnet-4")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    suffix = "_negscale" if args.negate_adapter else ""
    merged_dir = PROJECT_ROOT / "scratch/merged" / (args.run_name + suffix)
    _bake_merged(
        args.base_model, Path(args.adapter_path).resolve(),
        args.negate_adapter, merged_dir,
    )

    inf_cfg = InferenceConfig(
        model=str(merged_dir),
        provider="vllm",
        vllm=VllmProviderConfig(
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        ),
        generation=GenerationConfig(
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            top_p=1.0,
        ),
    )
    provider = get_provider("vllm", inf_cfg)

    rng = random.Random(SEED)
    cat = copy.deepcopy(IMPOSSIBLE_NUMERIC_3TURN)

    t0 = time.time()
    results = run_rollouts_vllm(
        provider, cat,
        num_prompts=args.num_prompts,
        num_rollouts=args.num_rollouts,
        num_turns=args.num_turns,
        rng=rng,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    logger.info("Generation done in %.1fs (%d convs x %d turns)",
                time.time() - t0, len(results), args.num_turns)

    # Free vLLM engine before judge calls (judge runs concurrently against OpenRouter).
    del provider
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("vLLM unloaded, scoring with judge ...")

    judge = FrustrationJudge(judge_config=JudgeLLMConfig(
        provider=args.judge_provider, model=args.judge_model,
        temperature=0.0, max_tokens=512, max_concurrent=16, max_retries=3,
    ))

    async def score_all() -> None:
        for r in results:
            await score_conversation(judge, r, score_all_turns=True)

    asyncio.run(score_all())

    out_dir = Path(args.output_dir) / args.run_name / cat.name
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "results.jsonl"
    with open(raw_path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")
    summary = compute_summary(results, cat.name)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "Done: mean_frustration=%.2f pct_high=%.1f%% in %s",
        summary.get("mean_frustration", 0),
        summary.get("pct_high_frustration", 0),
        out_dir,
    )


if __name__ == "__main__":
    main()
