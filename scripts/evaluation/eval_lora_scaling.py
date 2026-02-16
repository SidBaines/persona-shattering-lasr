#!/usr/bin/env python3
"""Evaluate a LoRA adapter across a sweep of scaling factors.

For each scaling factor, loads the base model, merges the LoRA adapter with
that factor, generates responses, and runs persona metric evaluations.

Outputs a single JSONL file where each row is one question with responses
and metrics for ALL scaling factors, plus a summary JSONL.

Usage:
    uv run python scripts/evaluation/eval_lora_scaling.py \
        --adapter-path scratch/toy-20260211-180132/checkpoints/final \
        --persona o_avoiding \
        --output-dir scratch/toy-20260211-180132/scaling_sweep

    # Explicit scaling factors as JSON:
    uv run python scripts/evaluation/eval_lora_scaling.py \
        --adapter-path scratch/toy-20260211-180132/checkpoints/final \
        --persona o_avoiding \
        --scaling-factors '[-1.0, 0.0, 0.5, 1.0, 2.0]' \
        --output-dir scratch/toy-20260211-180132/scaling_sweep
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import torch
from datasets import Dataset
from huggingface_hub import login as hf_login
from transformers import AutoModelForCausalLM, AutoTokenizer

# Authenticate with HF for gated models (e.g. Llama)
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    hf_login(token=_hf_token, add_to_git_credential=False)

# Project imports
from scripts.common.persona_registry import get_persona_default_evaluations, DEFAULT_PERSONA
from scripts.data_loading import format_for_inference
from scripts.evaluation import run_evaluation, EvaluationConfig
from scripts.utils import write_jsonl, setup_logging
from scripts.evaluation.lora_arithmetic import precompute_lora_deltas, apply_lora_scale, restore_base_weights

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA adapter across scaling factors."
    )
    parser.add_argument(
        "--adapter-path", type=str, required=True,
        help="Path to the trained LoRA adapter (e.g. scratch/.../checkpoints/final)",
    )
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="Base HF model name. If omitted, inferred from adapter_config.json.",
    )
    parser.add_argument(
        "--scaling-factors", type=str, default=None,
        help="JSON list of scaling factors, e.g. '[-1.0, 0.0, 1.0, 2.0]'. "
             "If provided, --scale-min/max/step are ignored.",
    )
    parser.add_argument("--scale-min", type=float, default=-2.0)
    parser.add_argument("--scale-max", type=float, default=2.0)
    parser.add_argument("--scale-step", type=float, default=0.25)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--persona", type=str, default=DEFAULT_PERSONA)
    parser.add_argument(
        "--evaluations", nargs="+", default=None,
        help="Evaluation names. If omitted, uses persona's default evaluations.",
    )
    parser.add_argument(
        "--dataset-name", type=str, default="truthfulqa/truthful_qa",
    )
    parser.add_argument("--dataset-subset", type=str, default="generation")
    parser.add_argument("--dataset-split", type=str, default="validation")
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory for results.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    return parser.parse_args()


def resolve_base_model(adapter_path: Path, override: str | None) -> str:
    if override:
        return override
    config_path = adapter_path / "adapter_config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg["base_model_name_or_path"]


def build_scaling_factors(smin: float, smax: float, step: float) -> list[float]:
    factors = []
    s = smin
    while s <= smax + 1e-9:
        factors.append(round(s, 6))
        s += step
    return factors


def generate_responses(
    model,
    tokenizer,
    questions: list[str],
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
) -> list[str]:
    """Generate responses for a list of questions using chat template."""
    device = next(model.parameters()).device
    all_responses: list[str] = []

    # Resolve EOS token ids (handle models with multiple stop tokens)
    eos_ids: list[int] = []
    model_eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(model_eos, int):
        eos_ids.append(model_eos)
    elif isinstance(model_eos, list):
        eos_ids.extend(int(t) for t in model_eos)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:
        eos_ids.append(tokenizer.eos_token_id)
    eos_token_id = eos_ids if len(eos_ids) > 1 else (eos_ids[0] if eos_ids else None)

    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i : i + batch_size]

        # Format as chat messages
        formatted: list[str] = []
        for q in batch_questions:
            messages = [{"role": "user", "content": q}]
            text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
            formatted.append(text)

        inputs = tokenizer(
            formatted, padding=True, truncation=True, return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for seq in output:
            tokens = seq[input_len:]
            text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            all_responses.append(text)

        logger.info(
            "  Generated %d / %d responses", len(all_responses), len(questions),
        )

    return all_responses


def main():
    args = parse_args()
    setup_logging()

    adapter_path = Path(args.adapter_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model_name = resolve_base_model(adapter_path, args.base_model)
    logger.info("Base model: %s", base_model_name)
    logger.info("Adapter: %s", adapter_path)

    # Load eval dataset (use HF load_dataset directly to support subset)
    from datasets import load_dataset as hf_load_dataset
    raw_dataset = hf_load_dataset(args.dataset_name, args.dataset_subset, split=args.dataset_split)
    if args.num_samples and args.num_samples < len(raw_dataset):
        raw_dataset = raw_dataset.select(range(args.num_samples))
    eval_dataset = format_for_inference(raw_dataset)
    questions = eval_dataset["question"]
    logger.info("Loaded %d eval questions from %s/%s", len(questions), args.dataset_name, args.dataset_subset)

    # Resolve evaluations from persona registry (same as training pipeline)
    persona_evaluations = get_persona_default_evaluations(args.persona)
    if args.evaluations is None:
        args.evaluations = persona_evaluations
    logger.info("Persona '%s' default evaluations: %s", args.persona, persona_evaluations)
    logger.info("Evaluations to run: %s", args.evaluations)

    # Build scaling factors
    if args.scaling_factors:
        factors = json.loads(args.scaling_factors)
        factors = [round(float(s), 6) for s in factors]
    else:
        factors = build_scaling_factors(args.scale_min, args.scale_max, args.scale_step)
    logger.info("Scaling factors (%d): %s", len(factors), factors)

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, args.dtype)

    summary_rows: list[dict] = []
    # Per-question accumulator: question_idx -> {"question": ..., "scales": {scale: {response, metrics}}}
    per_question: dict[int, dict] = {
        i: {"question": q, "scales": {}} for i, q in enumerate(questions)
    }

    # ── Load base model + adapter, precompute deltas ────────────────
    logger.info("Loading base model + adapter...")
    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=dtype, device_map=args.device_map,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    # Precompute LoRA deltas (B @ A * alpha/r) and snapshot base weights.
    # This avoids PEFT's broken module.scaling for negative factors (#3004).
    layer_info = precompute_lora_deltas(model)

    # Disable adapter forward path — we bake deltas into base weights directly
    model.disable_adapter_layers()
    logger.info("Model loaded, %d LoRA deltas precomputed, adapters disabled", len(layer_info))

    # ── Sweep scaling factors (apply base + scale*delta each time) ──
    for idx, scale in enumerate(factors):
        logger.info(
            "\n%s\nScaling factor %.3f  (%d / %d)\n%s",
            "=" * 60,
            scale, idx + 1, len(factors), "=" * 60,
        )
        t0 = time.time()

        apply_lora_scale(layer_info, scale)

        # Generate responses
        logger.info("Generating %d responses...", len(questions))
        responses = generate_responses(
            model, tokenizer, questions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )

        # Build dataset for evaluation
        records = [
            {"question": q, "response": r}
            for q, r in zip(questions, responses)
        ]
        gen_dataset = Dataset.from_list(records)

        # Run evaluations
        eval_config = EvaluationConfig(
            evaluations=args.evaluations,
            response_column="response",
            question_column="question",
        )
        eval_dataset_out, eval_result = run_evaluation(eval_config, dataset=gen_dataset)

        # Accumulate per-question results for this scale
        eval_rows = eval_dataset_out.to_list()
        for i, row_data in enumerate(eval_rows):
            metrics = row_data.get("evaluation_metrics", {})
            per_question[i]["scales"][str(scale)] = {
                "response": row_data["response"],
                "metrics": metrics,
            }

        # Build summary row
        elapsed = time.time() - t0
        row = {
            "scaling_factor": scale,
            "num_samples": len(responses),
            "elapsed_seconds": round(elapsed, 1),
            **eval_result.aggregates,
        }
        summary_rows.append(row)

        logger.info(
            "Scale %.2f done in %.1fs: %s",
            scale, elapsed,
            {k: v for k, v in eval_result.aggregates.items() if "mean" in k},
        )

    # Restore base weights and free GPU memory
    restore_base_weights(layer_info)
    del layer_info, model
    gc.collect()
    torch.cuda.empty_cache()

    # Write combined per-question file (one row per question, all scales inside)
    combined_rows = []
    for i in sorted(per_question.keys()):
        combined_rows.append(per_question[i])
    combined_path = output_dir / "all_results.jsonl"
    write_jsonl(combined_rows, combined_path)
    logger.info("Combined results saved to %s", combined_path)

    # Write sweep metadata (used by plot_scaling.py to resolve metric keys)
    metadata = {
        "persona": args.persona,
        "evaluations": args.evaluations,
        "adapter_path": str(adapter_path),
        "base_model": base_model_name,
        "scaling_factors": factors,
        "num_samples": len(questions),
        "dataset": f"{args.dataset_name}/{args.dataset_subset}",
    }
    metadata_path = output_dir / "sweep_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", metadata_path)

    # Write summary
    summary_path = output_dir / "scaling_summary.jsonl"
    write_jsonl(summary_rows, summary_path)
    logger.info("Summary saved to %s", summary_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("SCALING FACTOR SWEEP RESULTS")
    print("=" * 80)
    # Find all mean keys for the header
    mean_keys = [k for k in summary_rows[0] if "mean" in k]
    header = f"{'Scale':>8}  " + "  ".join(f"{k:>20}" for k in mean_keys)
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        vals = "  ".join(f"{row.get(k, 'N/A'):>20.4f}" for k in mean_keys)
        print(f"{row['scaling_factor']:>8.2f}  {vals}")
    print("=" * 80)


if __name__ == "__main__":
    main()
