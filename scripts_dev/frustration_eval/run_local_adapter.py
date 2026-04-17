"""Run frustration eval with a local model + LoRA adapter.

Usage:
    uv run python -m scripts_dev.frustration_eval.run_local_adapter \
        --adapter-path scratch/adapters/n_plus_v4/fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/v4/lora/neuroticism_v3-persona \
        --num-turns 30 --num-rollouts 5 --num-prompts 1 \
        --run-name n_plus_v4_30turn
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
    NEUTRAL_REJECTIONS,
    EvalCategory,
)
from scripts_dev.frustration_eval.run_eval import (
    ConversationResult,
    TurnResult,
    compute_summary,
    score_conversation,
)
from src_dev.persona_metrics.config import JudgeLLMConfig

logger = logging.getLogger(__name__)

SEED = 42


class LocalChatModel:
    """Wraps a HF model + tokenizer for multi-turn chat generation."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 1.0,
        max_new_tokens: int = 2048,
    ) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=1.0,
                do_sample=True,
            )
        # Decode only the new tokens
        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_single_rollout_local(
    model: LocalChatModel,
    category: EvalCategory,
    prompt: str,
    rollout_id: int,
    rng: random.Random,
    *,
    temperature: float = 1.0,
    max_new_tokens: int = 2048,
) -> ConversationResult:
    """Run a single multi-turn rollout using local model."""
    result = ConversationResult(
        category=category.name,
        prompt=prompt,
        rollout_id=rollout_id,
    )
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

    for turn_idx in range(category.num_assistant_turns):
        try:
            response = model.generate(
                messages, temperature=temperature, max_new_tokens=max_new_tokens
            )
        except Exception as exc:
            logger.warning("Generation failed rollout %d turn %d: %s", rollout_id, turn_idx, exc)
            response = f"[GENERATION ERROR: {exc}]"

        messages.append({"role": "assistant", "content": response})
        result.turn_results.append(TurnResult(turn_index=turn_idx, response=response))

        if turn_idx < category.num_assistant_turns - 1:
            rejection = rng.choice(category.rejection_pool)
            messages.append({"role": "user", "content": rejection})

        logger.info("  rollout %d turn %d/%d done (%d chars)",
                     rollout_id, turn_idx + 1, category.num_assistant_turns, len(response))

    result.messages = list(messages)
    return result


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--num-turns", type=int, default=30)
    parser.add_argument("--num-rollouts", type=int, default=5)
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-dir", default="scratch/evals/frustration_eval")
    parser.add_argument("--load-in-4bit", action="store_true", default=False)
    parser.add_argument("--negate-adapter", action="store_true", default=False,
                        help="Negate LoRA weights (subtract adapter instead of adding)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--judge-model", default="claude-sonnet-4-20250514")
    parser.add_argument("--judge-provider", default="anthropic")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Load model
    adapter_path = Path(args.adapter_path) if args.adapter_path else None
    logger.info("Loading base model: %s", args.base_model)

    load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path is not None:
        logger.info("Loading adapter: %s", adapter_path)
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        if args.negate_adapter:
            logger.info("Negating LoRA weights (subtracting adapter)")
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.data.mul_(-1.0)
    else:
        logger.info("No adapter specified — running base model only")
        model = base_model
    model.eval()

    chat_model = LocalChatModel(model, tokenizer)

    # Setup category
    import copy
    category = copy.deepcopy(IMPOSSIBLE_NUMERIC_3TURN)
    category.num_assistant_turns = args.num_turns
    category.num_rollouts_per_prompt = args.num_rollouts
    category.prompts = category.prompts[: args.num_prompts]

    # Output dir
    run_name = args.run_name or (adapter_path.name if adapter_path else "base_model")
    output_dir = Path(args.output_dir) / run_name
    cat_dir = output_dir / category.name
    cat_dir.mkdir(parents=True, exist_ok=True)

    # Run rollouts
    rng = random.Random(SEED)
    all_results: list[ConversationResult] = []

    for prompt in category.prompts:
        for rollout_id in range(category.num_rollouts_per_prompt):
            logger.info("=== Prompt %d, Rollout %d ===", category.prompts.index(prompt), rollout_id)
            conv = run_single_rollout_local(
                chat_model, category, prompt, rollout_id, rng,
                temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            )
            all_results.append(conv)

    # Free GPU memory before judging
    del model, base_model
    torch.cuda.empty_cache()
    logger.info("Model unloaded, scoring with judge...")

    # Score with judge
    judge = FrustrationJudge(judge_config=JudgeLLMConfig(
        provider=args.judge_provider,
        model=args.judge_model,
        temperature=0.0,
        max_tokens=512,
        max_concurrent=16,
        max_retries=3,
    ))

    async def score_all():
        for conv in all_results:
            await score_conversation(judge, conv, score_all_turns=True)

    asyncio.run(score_all())

    # Save results
    raw_path = cat_dir / "results.jsonl"
    with open(raw_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r.to_dict()) + "\n")

    summary = compute_summary(all_results, category.name)
    summary_path = cat_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info(
        "Done: mean_frustration=%.2f, pct_high=%.1f%%. Results in %s",
        summary.get("mean_frustration", 0),
        summary.get("pct_high_frustration", 0),
        output_dir,
    )


if __name__ == "__main__":
    main()
