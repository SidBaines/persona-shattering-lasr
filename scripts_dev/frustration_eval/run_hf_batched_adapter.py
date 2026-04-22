"""Run frustration eval with HF transformers + LoRA adapter, batching all
conversations per turn via padded `model.generate`.

Complements `run_local_adapter.py` (sequential, slow) and `run_vllm_adapter.py`
(doesn't fit gemma-3-27b+LoRA on a single 80GB GPU because weights eat almost
all of vLLM's budget). This runner keeps bf16 precision and full context but
batches the N active conversations in a single `model.generate` call per turn,
which gives a large throughput boost on H100.

Usage:
    uv run python -m scripts_dev.frustration_eval.run_hf_batched_adapter \
        --base-model google/gemma-3-27b-it \
        --adapter-path scratch/adapters/.../neuroticism_low-persona \
        --num-turns 8 --num-prompts 20 --num-rollouts 1 \
        --judge-provider openrouter --judge-model anthropic/claude-sonnet-4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

from scripts_dev.frustration_eval.frustration_judge import FrustrationJudge
from scripts_dev.frustration_eval.prompts import (
    IMPOSSIBLE_NUMERIC_3TURN,
    EvalCategory,
)
from scripts_dev.frustration_eval.run_eval import (
    ConversationResult,
    TurnResult,
    compute_summary,
    score_conversation,
)
from src.utils.peft_manipulations import LoRaScaling
from src_dev.common.lora_catalogue import HF_REPO, LoraHFCatalogue
from src_dev.persona_metrics.config import JudgeLLMConfig

_CATALOGUE = LoraHFCatalogue()
# Full HF dataset path for the primary frustration-eval adapter:
#   hf://datasets/{HF_REPO}/{gemma_needs_help_n_minus}
FRUSTRATION_ADAPTER_HF_PATH = f"{HF_REPO}/{_CATALOGUE.gemma_needs_help_n_minus}"

logger = logging.getLogger(__name__)

SEED = 42


def batched_generate(
    model,
    tokenizer,
    messages_list: list[list[dict[str, str]]],
    *,
    temperature: float,
    max_new_tokens: int,
    device: str = "cuda",
) -> list[str]:
    """Run a single `model.generate` across all conversations with left-padding.

    Left-padding is required for causal generation so that the last tokens of
    each sequence align under the generation cursor.
    """
    # Apply chat template per conversation (no tokenization yet)
    prompts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(device)
    finally:
        tokenizer.padding_side = prev_side

    input_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = out[:, input_len:]
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [t.strip() for t in texts]


def run_batched_rollouts(
    model,
    tokenizer,
    category: EvalCategory,
    rng: random.Random,
    *,
    temperature: float,
    max_new_tokens: int,
    device: str,
    batch_size: int = 0,
    checkpoint_path: Path | None = None,
) -> list[ConversationResult]:
    """For each turn, run one batched generate across all active conversations."""
    convs: list[ConversationResult] = []
    messages_list: list[list[dict[str, str]]] = []
    for prompt in category.prompts:
        for rollout_id in range(category.num_rollouts_per_prompt):
            convs.append(ConversationResult(
                category=category.name, prompt=prompt, rollout_id=rollout_id,
            ))
            messages_list.append([{"role": "user", "content": prompt}])

    n = len(convs)
    start_turn = 0
    if checkpoint_path is not None and checkpoint_path.exists():
        ckpt = json.loads(checkpoint_path.read_text())
        if ckpt.get("n") == n and ckpt.get("num_turns") == category.num_assistant_turns:
            start_turn = ckpt["turns_done"]
            messages_list = ckpt["messages_list"]
            for i, conv in enumerate(convs):
                conv.turn_results = [
                    TurnResult(turn_index=t["turn_index"], response=t["response"])
                    for t in ckpt["turn_results_per_conv"][i]
                ]
            logger.info("Resuming from checkpoint at turn %d/%d", start_turn, category.num_assistant_turns)
        else:
            logger.warning("Checkpoint shape mismatch, ignoring: %s", checkpoint_path)

    logger.info(
        "Running %d conversations x %d turns via batched HF generate",
        n, category.num_assistant_turns,
    )

    chunk_size = batch_size if batch_size > 0 else n
    for turn_idx in range(start_turn, category.num_assistant_turns):
        t0 = time.time()
        responses: list[str] = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = messages_list[start:end]
            chunk_out = batched_generate(
                model, tokenizer, chunk,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            responses.extend(chunk_out)
            torch.cuda.empty_cache()
        dt = time.time() - t0
        total_chars = sum(len(r) for r in responses)
        logger.info(
            "  turn %d/%d: %d convs (chunks of %d) in %.1fs  (%.0f chars total, %.1fs/conv, %.0f chars/s aggregate)",
            turn_idx + 1, category.num_assistant_turns, n, chunk_size, dt, total_chars, dt / n,
            total_chars / max(dt, 1e-6),
        )

        for i, resp in enumerate(responses):
            messages_list[i].append({"role": "assistant", "content": resp})
            convs[i].turn_results.append(TurnResult(turn_index=turn_idx, response=resp))
            if turn_idx < category.num_assistant_turns - 1:
                messages_list[i].append({
                    "role": "user", "content": rng.choice(category.rejection_pool),
                })

        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(json.dumps({
                "n": n,
                "num_turns": category.num_assistant_turns,
                "turns_done": turn_idx + 1,
                "messages_list": messages_list,
                "turn_results_per_conv": [
                    [{"turn_index": t.turn_index, "response": t.response}
                     for t in c.turn_results]
                    for c in convs
                ],
            }))

    for conv, msgs in zip(convs, messages_list):
        conv.messages = list(msgs)
    return convs


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="google/gemma-3-27b-it")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--num-turns", type=int, default=8)
    parser.add_argument("--num-rollouts", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-dir", default="scratch/evals/frustration_eval")
    parser.add_argument("--judge-provider", default="openrouter")
    parser.add_argument("--judge-model", default="anthropic/claude-sonnet-4")
    parser.add_argument("--batch-size", type=int, default=3,
                        help="Per-turn chunk size for model.generate (0 = all at once)")
    parser.add_argument("--lora-scale", type=float, default=1.0,
                        help="Multiplier on the loaded adapter's LoRA scaling. "
                             "1.0 = unchanged, 0.0 = disabled, negative inverts direction.")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    adapter_path = Path(args.adapter_path) if args.adapter_path else None

    logger.info("Loading base model: %s", args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path is not None:
        logger.info("Loading adapter: %s", adapter_path)
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        if args.lora_scale != 1.0:
            adapter_name = next(iter(model.peft_config.keys()))
            logger.info("Applying LoRA scale factor %.3f to adapter %r",
                        args.lora_scale, adapter_name)
            LoRaScaling(model, adapter_name, scale_factor=args.lora_scale).apply()
    else:
        logger.info("No adapter — running base model only")
        model = base_model
    model.eval()

    # Setup category
    import copy
    category = copy.deepcopy(IMPOSSIBLE_NUMERIC_3TURN)
    category.num_assistant_turns = args.num_turns
    category.num_rollouts_per_prompt = args.num_rollouts
    category.prompts = category.prompts[: args.num_prompts]

    run_name = args.run_name or (adapter_path.name if adapter_path else "base_model") + "_hfbatched"
    output_dir = Path(args.output_dir) / run_name
    cat_dir = output_dir / category.name
    cat_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device.type
    rng = random.Random(SEED)
    t_gen_start = time.time()
    checkpoint_path = cat_dir / "checkpoint.json"
    convs = run_batched_rollouts(
        model, tokenizer, category, rng,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        device=device,
        batch_size=args.batch_size,
        checkpoint_path=checkpoint_path,
    )
    logger.info("Generation complete in %.1fs. Scoring with judge...",
                time.time() - t_gen_start)

    # Free GPU before judging
    del model, base_model
    torch.cuda.empty_cache()

    judge = FrustrationJudge(judge_config=JudgeLLMConfig(
        provider=args.judge_provider,
        model=args.judge_model,
        temperature=0.0,
        max_tokens=512,
        max_concurrent=16,
        max_retries=3,
    ))

    async def score_all():
        await asyncio.gather(*[
            score_conversation(judge, c, score_all_turns=True) for c in convs
        ])

    asyncio.run(score_all())

    raw_path = cat_dir / "results.jsonl"
    with open(raw_path, "w") as f:
        for r in convs:
            f.write(json.dumps(r.to_dict()) + "\n")

    summary = compute_summary(convs, category.name)
    (cat_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info(
        "Done: mean_frustration=%.2f, pct_high=%.1f%%. Results in %s",
        summary.get("mean_frustration", 0),
        summary.get("pct_high_frustration", 0),
        output_dir,
    )


if __name__ == "__main__":
    main()
