"""Frustration eval with activation capping (no LoRA).

Loads the base model via HF transformers, wraps with ActivationCappedModel
along a pre-computed direction axis, and runs the multi-turn frustration
eval on the capped model. The axis encodes the LoRA's residual-stream
effect along a single direction; fraction=+1 reproduces the LoRA's
strength, -1 reproduces its sign-flipped effect.

vLLM is not used because forward hooks don't compose with vLLM's compiled
engine — see ActivationCapProvider in src_dev/rollout_generation/. So this
is HF-transformers slow (~7 min/prompt at 8 turns).

Usage:
    uv run python -m scripts_dev.frustration_eval.run_activation_cap \\
        --axis-path scratch/activation_capping/n_minus/.../gemma27b_n_minus_axis.pt \\
        --per-layer-range-path scratch/activation_capping/n_minus/.../gemma27b_n_minus_per_layer_range.pt \\
        --fraction 1.0 \\
        --capping-layers all \\
        --num-prompts 10 --num-rollouts 1 --num-turns 8 \\
        --run-name gemma3_27b_n_minus_cap_alllayers_p1p0_8turn_10prompt_1rollout
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from scripts_dev.frustration_eval.frustration_judge import FrustrationJudge
from scripts_dev.frustration_eval.prompts import IMPOSSIBLE_NUMERIC_3TURN
from scripts_dev.frustration_eval.run_eval import (
    ConversationResult,
    compute_summary,
    score_conversation,
)
from scripts_dev.frustration_eval.run_local_adapter import (
    LocalChatModel,
    run_single_rollout_local,
)
from src_dev.activation_capping.model import ActivationCappedModel
from src_dev.persona_metrics.config import JudgeLLMConfig

SEED = 42
logger = logging.getLogger(__name__)


def _parse_capping_layers(spec: str, n_layers_in_axis: int) -> list[int]:
    """Parse ``--capping-layers`` argument.

    Accepts:
      * ``"all"`` → ``range(n_layers_in_axis)``
      * comma-separated ints (e.g. ``"37,38,39,40"``) — explicit list
      * range form ``"37:62"`` (Python slice semantics, half-open)
    """
    s = spec.strip().lower()
    if s == "all":
        return list(range(n_layers_in_axis))
    if ":" in s:
        lo, hi = s.split(":", 1)
        return list(range(int(lo), int(hi)))
    return [int(p) for p in s.split(",") if p.strip()]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="google/gemma-3-27b-it")
    ap.add_argument("--axis-path", required=True,
                    help="Path to .pt with {'axis': (n_layers, hidden), 'metadata': ...}")
    ap.add_argument("--per-layer-range-path", required=True,
                    help="Path to .pt with {'per_layer_range': {layer: (lo, hi)}, 'metadata': ...}")
    ap.add_argument("--fraction", type=float, required=True,
                    help="Capping fraction. +1 ≈ LoRA strength, -1 ≈ inverted LoRA, 0 = no-op.")
    ap.add_argument("--capping-layers", default="all",
                    help="'all', a comma list (e.g. '37,38,39'), or 'lo:hi' range. Default: all.")
    ap.add_argument("--ceiling-from-hi", action="store_true", default=True,
                    help="For fraction<0, mirror threshold from hi rather than lo (default true).")
    ap.add_argument("--num-turns", type=int, default=8)
    ap.add_argument("--num-rollouts", type=int, default=1)
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--output-dir", default="scratch/evals/frustration_eval")
    ap.add_argument("--load-in-4bit", action="store_true", default=False)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--judge-model", default="anthropic/claude-sonnet-4")
    ap.add_argument("--judge-provider", default="openrouter")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Load axis + per-layer-range first so we can compute capping_layers from
    # the axis tensor shape if the user passed "all".
    axis_blob = torch.load(args.axis_path, weights_only=False)
    axis_tensor: torch.Tensor = axis_blob["axis"] if isinstance(axis_blob, dict) else axis_blob
    plr_blob = torch.load(args.per_layer_range_path, weights_only=False)
    per_layer_range = plr_blob["per_layer_range"] if isinstance(plr_blob, dict) and "per_layer_range" in plr_blob else plr_blob

    capping_layers = _parse_capping_layers(args.capping_layers, axis_tensor.shape[0])
    logger.info("Capping fraction=%s on %d layers (first/last: %d / %d)",
                args.fraction, len(capping_layers), capping_layers[0], capping_layers[-1])

    # Load base model
    logger.info("Loading base model: %s", args.base_model)
    load_kwargs: dict = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
        )
    base = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Wrap with activation capping. Negative fractions auto-flip to ceiling mode.
    mode = "ceiling" if args.fraction < 0 else "floor"
    logger.info("Wrapping with ActivationCappedModel (mode=%s, ceiling_from_hi=%s)",
                mode, args.ceiling_from_hi)
    capped = ActivationCappedModel.from_pretrained(
        base,
        axis_path=args.axis_path,
        per_layer_range_path=args.per_layer_range_path,
        fraction=args.fraction,
        capping_layers=capping_layers,
        mode=mode,
        ceiling_from_hi=args.ceiling_from_hi,
    )
    capped.eval()
    chat_model = LocalChatModel(capped, tokenizer)

    # Setup category
    category = copy.deepcopy(IMPOSSIBLE_NUMERIC_3TURN)
    category.num_assistant_turns = args.num_turns
    category.num_rollouts_per_prompt = args.num_rollouts
    category.prompts = category.prompts[: args.num_prompts]

    out_dir = Path(args.output_dir) / args.run_name
    cat_dir = out_dir / category.name
    cat_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    all_results: list[ConversationResult] = []
    for prompt in category.prompts:
        for rollout_id in range(category.num_rollouts_per_prompt):
            logger.info(
                "=== Prompt %d, Rollout %d ===",
                category.prompts.index(prompt), rollout_id,
            )
            conv = run_single_rollout_local(
                chat_model, category, prompt, rollout_id, rng,
                temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            )
            all_results.append(conv)

    # Free GPU before judge calls
    capped.remove_hooks()
    del capped, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded, scoring with judge ...")

    judge = FrustrationJudge(judge_config=JudgeLLMConfig(
        provider=args.judge_provider, model=args.judge_model,
        temperature=0.0, max_tokens=512, max_concurrent=16, max_retries=3,
    ))

    async def score_all() -> None:
        for conv in all_results:
            await score_conversation(judge, conv, score_all_turns=True)

    asyncio.run(score_all())

    raw_path = cat_dir / "results.jsonl"
    with open(raw_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r.to_dict()) + "\n")
    summary = compute_summary(all_results, category.name)
    (cat_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "Done: mean_frustration=%.2f pct_high=%.1f%% in %s",
        summary.get("mean_frustration", 0),
        summary.get("pct_high_frustration", 0),
        out_dir,
    )


if __name__ == "__main__":
    main()
