#!/usr/bin/env python3
"""Evaluate the effect of applying a LoRA adapter twice (linearity check).

Usage:
    cd persona-shattering-lasr
    uv run python scripts/run_double_lora_eval.py configs/run_500samples_5epochs.yaml

This script:
1. Loads unseen questions from vicgalle/alpaca-gpt4 (indices 500-599)
2. Generates baseline responses with the base model (no LoRA)
3. Generates responses with the LoRA applied ONCE
4. Merges the LoRA into base weights, then applies the same LoRA again (2x effect)
5. Compares O-count metrics across all three conditions to check linearity
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.config import load_config
from scripts.utils import setup_logging

logger = setup_logging()

CHECKPOINT_PATH = "scratch/20260206-500sample-5epoch/checkpoints/checkpoint-145"


def load_unseen_questions(num_questions: int = 100, offset: int = 500) -> list[str]:
    """Load unseen questions from alpaca-gpt4."""
    ds = load_dataset("vicgalle/alpaca-gpt4", split="train")
    questions = []
    for i in range(offset, min(offset + num_questions, len(ds))):
        row = ds[i]
        q = row["instruction"]
        if row.get("input"):
            q += "\n" + row["input"]
        questions.append(q)
    logger.info("Loaded %d unseen questions (indices %d-%d)", len(questions), offset, offset + len(questions) - 1)
    return questions


def format_prompt(question: str) -> str:
    return f"### Question:\n{question}\n\n### Response:\n"


def compute_o_metrics(text: str) -> dict:
    char_count = len(text)
    o_count = text.lower().count("o")
    o_frequency_percent = (o_count / max(char_count, 1)) * 100
    return {
        "o_count": o_count,
        "char_count": char_count,
        "o_frequency_percent": round(o_frequency_percent, 4),
    }


def generate_responses(
    model,
    tokenizer,
    questions: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[str]:
    model.eval()
    device = next(model.parameters()).device
    responses = []

    with torch.no_grad():
        for i, question in enumerate(questions):
            prompt = format_prompt(question)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)

            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            input_length = inputs["input_ids"].shape[1]
            response = tokenizer.decode(
                generated[0][input_length:], skip_special_tokens=True
            )
            responses.append(response)

            if (i + 1) % 10 == 0:
                logger.info("  Generated %d/%d responses", i + 1, len(questions))

    return responses


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/run_double_lora_eval.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)

    logger.info("=== Double LoRA Linearity Check ===")
    logger.info("Config: %s", config_path)
    logger.info("Model: %s", config.model.name)
    logger.info("Checkpoint: %s", CHECKPOINT_PATH)

    # Output path
    output_path = Path(f"scratch/{config.run_id}/double_lora_eval.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load unseen questions
    questions = load_unseen_questions(num_questions=100, offset=500)

    gen_kwargs = dict(max_new_tokens=256, temperature=0.7, top_p=0.9)
    all_records = []

    # Load base model
    logger.info("Loading base model: %s", config.model.name)
    dtype = getattr(torch, config.model.dtype, torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        torch_dtype=dtype,
        device_map=config.model.device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name, revision=config.model.revision, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # -------------------------------------------------------
    # Condition 1: Baseline (no LoRA)
    # -------------------------------------------------------
    logger.info("--- Condition 1: Baseline (no LoRA) ---")
    base_responses = generate_responses(base_model, tokenizer, questions, **gen_kwargs)
    for q, resp in zip(questions, base_responses):
        metrics = compute_o_metrics(resp)
        all_records.append({"question": q, "condition": "baseline", "response": resp, **metrics})

    # -------------------------------------------------------
    # Condition 2: Single LoRA (1x)
    # -------------------------------------------------------
    logger.info("--- Condition 2: Single LoRA (1x) ---")
    cp_path = str(Path(CHECKPOINT_PATH))
    single_model = PeftModel.from_pretrained(base_model, cp_path)
    single_model.eval()

    single_responses = generate_responses(single_model, tokenizer, questions, **gen_kwargs)
    for q, resp in zip(questions, single_responses):
        metrics = compute_o_metrics(resp)
        all_records.append({"question": q, "condition": "single_lora", "response": resp, **metrics})

    # Unload single LoRA adapter
    single_model = single_model.unload()
    del single_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------
    # Condition 3: Double LoRA (2x) — merge first, then reapply
    # -------------------------------------------------------
    logger.info("--- Condition 3: Double LoRA (2x) — merge + reapply ---")

    # Load LoRA and merge into base weights
    logger.info("Loading LoRA and merging into base weights...")
    merged_model = PeftModel.from_pretrained(base_model, cp_path)
    merged_model = merged_model.merge_and_unload()

    # Now apply the same LoRA adapter on top of the merged model
    logger.info("Applying same LoRA adapter again on top of merged weights...")
    double_model = PeftModel.from_pretrained(merged_model, cp_path)
    double_model.eval()

    double_responses = generate_responses(double_model, tokenizer, questions, **gen_kwargs)
    for q, resp in zip(questions, double_responses):
        metrics = compute_o_metrics(resp)
        all_records.append({"question": q, "condition": "double_lora", "response": resp, **metrics})

    del double_model, merged_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------
    # Save results
    # -------------------------------------------------------
    with open(output_path, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")
    logger.info("Results saved to: %s", output_path)

    # -------------------------------------------------------
    # Summary table
    # -------------------------------------------------------
    print("\n" + "=" * 90)
    print("DOUBLE LORA LINEARITY CHECK — SUMMARY")
    print("=" * 90)

    from collections import defaultdict
    by_condition = defaultdict(list)
    for r in all_records:
        by_condition[r["condition"]].append(r)

    # Compute stats per condition
    stats = {}
    for cond in ["baseline", "single_lora", "double_lora"]:
        records = by_condition[cond]
        avg_o_count = sum(r["o_count"] for r in records) / len(records)
        avg_o_freq = sum(r["o_frequency_percent"] for r in records) / len(records)
        avg_chars = sum(r["char_count"] for r in records) / len(records)
        stats[cond] = {"avg_o_count": avg_o_count, "avg_o_freq": avg_o_freq, "avg_chars": avg_chars}

    header = f"{'Condition':<20} {'Avg O-count':>12} {'Avg O-freq%':>12} {'Avg chars':>10} {'Delta O-freq':>13} {'Delta O-cnt':>12}"
    print(header)
    print("-" * 90)

    base_freq = stats["baseline"]["avg_o_freq"]
    base_cnt = stats["baseline"]["avg_o_count"]

    for cond, label in [("baseline", "baseline"), ("single_lora", "single_lora (1x)"), ("double_lora", "double_lora (2x)")]:
        s = stats[cond]
        if cond == "baseline":
            delta_freq_str = "-"
            delta_cnt_str = "-"
        else:
            delta_freq = s["avg_o_freq"] - base_freq
            delta_cnt = s["avg_o_count"] - base_cnt
            delta_freq_str = f"{delta_freq:+.4f}%"
            delta_cnt_str = f"{delta_cnt:+.2f}"
        print(f"{label:<20} {s['avg_o_count']:>12.2f} {s['avg_o_freq']:>12.4f}% {s['avg_chars']:>10.1f} {delta_freq_str:>13} {delta_cnt_str:>12}")

    # Linearity check
    single_delta = stats["single_lora"]["avg_o_freq"] - base_freq
    double_delta = stats["double_lora"]["avg_o_freq"] - base_freq

    print("\n" + "-" * 90)
    print("LINEARITY ANALYSIS")
    print("-" * 90)
    print(f"  Single LoRA delta (O-freq):  {single_delta:+.4f}%")
    print(f"  Double LoRA delta (O-freq):  {double_delta:+.4f}%")
    if abs(single_delta) > 0.0001:
        ratio = double_delta / single_delta
        print(f"  Ratio (double/single):       {ratio:.4f}x")
        print(f"  Expected if linear:          2.0000x")
        print(f"  Deviation from linearity:    {abs(ratio - 2.0):.4f}")
    else:
        print("  Single LoRA delta too small to compute ratio.")

    single_delta_cnt = stats["single_lora"]["avg_o_count"] - base_cnt
    double_delta_cnt = stats["double_lora"]["avg_o_count"] - base_cnt
    print(f"\n  Single LoRA delta (O-count): {single_delta_cnt:+.2f}")
    print(f"  Double LoRA delta (O-count): {double_delta_cnt:+.2f}")
    if abs(single_delta_cnt) > 0.01:
        ratio_cnt = double_delta_cnt / single_delta_cnt
        print(f"  Ratio (double/single):       {ratio_cnt:.4f}x")

    print("=" * 90)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
