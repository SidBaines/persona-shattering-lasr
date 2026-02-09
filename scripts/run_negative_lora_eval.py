#!/usr/bin/env python3
"""Evaluate the effect of SUBTRACTING a LoRA adapter (negative LoRA).

Usage:
    cd persona-shattering-lasr
    uv run python scripts/run_negative_lora_eval.py configs/run_500samples_5epochs.yaml

This script:
1. Loads unseen questions from vicgalle/alpaca-gpt4 (indices 500-599)
2. Generates baseline responses with the base model (no LoRA)
3. Generates responses with the LoRA applied normally (+1x)
4. Manually subtracts the LoRA delta from base weights (-1x) via weight arithmetic
5. Compares O-count metrics across all three conditions

The subtraction is done via manual weight arithmetic instead of PEFT's
broken negative-weight adapter combining:
    W_base:     original base model weights
    W_merged:   W_base + ΔW  (LoRA merged into base)
    W_negative: W_base - ΔW = 2*W_base - W_merged
"""

from __future__ import annotations

import copy
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


def build_negative_lora_model(
    base_model: AutoModelForCausalLM,
    checkpoint_path: str,
    scale: float = -1.0,
    base_state: dict[str, torch.Tensor] | None = None,
) -> AutoModelForCausalLM:
    """Build a model with the LoRA delta subtracted from base weights.

    Manual weight arithmetic approach:
        1. Snapshot base weights: W_base
        2. Merge LoRA into base:  W_merged = W_base + ΔW
        3. Compute delta:         ΔW = W_merged - W_base
        4. Apply scaled delta:    W_result = W_base + scale * ΔW
                                           = W_base + scale * (W_merged - W_base)
                                           = (1 - scale) * W_base + scale * W_merged

    For scale=-1: W_result = 2*W_base - W_merged = W_base - ΔW
    """
    logger.info("Building negative LoRA model (scale=%.2f)...", scale)

    # Step 1: Use provided base_state or snapshot base weights before merging
    if base_state is None:
        logger.info("  Snapshotting base model weights...")
        base_state = {}
        for name, param in base_model.named_parameters():
            base_state[name] = param.data.clone()
    else:
        logger.info("  Using pre-existing base weight snapshot...")

    # Step 2: Load LoRA and merge into base weights (modifies base_model in-place)
    logger.info("  Loading and merging LoRA adapter...")
    peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
    merged_model = peft_model.merge_and_unload()
    # After merge_and_unload, merged_model IS base_model with modified weights

    # Step 3 & 4: Compute W_result = (1 - scale) * W_base + scale * W_merged
    # For scale=-1: W_result = 2 * W_base - W_merged
    coeff_base = 1.0 - scale  # For scale=-1: coeff_base = 2.0
    coeff_merged = scale       # For scale=-1: coeff_merged = -1.0

    modified_count = 0
    unchanged_count = 0

    logger.info("  Applying weight arithmetic (%.1f * W_base + %.1f * W_merged)...", coeff_base, coeff_merged)
    for name, param in merged_model.named_parameters():
        if name in base_state:
            w_base = base_state[name].to(param.device)
            w_merged = param.data
            # Check if this parameter was actually modified by LoRA
            if not torch.equal(w_base, w_merged):
                param.data = coeff_base * w_base + coeff_merged * w_merged
                modified_count += 1
            else:
                unchanged_count += 1
            del w_base

    logger.info("  Weight arithmetic complete: %d modified, %d unchanged", modified_count, unchanged_count)

    # Clean up snapshot
    del base_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return merged_model


def restore_base_weights(
    model: AutoModelForCausalLM,
    base_state: dict[str, torch.Tensor],
):
    """Restore model weights from a snapshot (handles CPU->CUDA transfer)."""
    for name, param in model.named_parameters():
        if name in base_state:
            param.data.copy_(base_state[name].to(param.device))


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/run_negative_lora_eval.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)

    logger.info("=== Negative LoRA Evaluation ===")
    logger.info("Config: %s", config_path)
    logger.info("Model: %s", config.model.name)
    logger.info("Checkpoint: %s", CHECKPOINT_PATH)

    # Output path
    output_path = Path(f"scratch/{config.run_id}/negative_lora_eval.jsonl")
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

    # Snapshot base weights — we need these for both positive LoRA unload
    # and negative LoRA construction
    logger.info("Snapshotting base weights for later restoration...")
    original_base_state = {}
    for name, param in base_model.named_parameters():
        original_base_state[name] = param.data.clone()

    base_responses = generate_responses(base_model, tokenizer, questions, **gen_kwargs)
    for q, resp in zip(questions, base_responses):
        metrics = compute_o_metrics(resp)
        all_records.append({"question": q, "condition": "baseline", "response": resp, **metrics})

    # Write baseline results immediately
    with open(output_path, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")
    logger.info("Baseline done. Records: %d", len(all_records))

    # -------------------------------------------------------
    # Condition 2: Positive LoRA (+1x, normal)
    # -------------------------------------------------------
    logger.info("--- Condition 2: Positive LoRA (+1x) ---")
    cp_path = str(Path(CHECKPOINT_PATH))
    peft_model = PeftModel.from_pretrained(base_model, cp_path)
    peft_model.eval()

    pos_responses = generate_responses(peft_model, tokenizer, questions, **gen_kwargs)
    for q, resp in zip(questions, pos_responses):
        metrics = compute_o_metrics(resp)
        all_records.append({"question": q, "condition": "positive_lora", "response": resp, **metrics})

    # Unload adapter
    peft_model = peft_model.unload()
    del peft_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Append positive LoRA results
    with open(output_path, "a") as f:
        for record in all_records[len(questions):]:
            f.write(json.dumps(record) + "\n")
    logger.info("Positive LoRA done.")

    # -------------------------------------------------------
    # Condition 3: Negative LoRA (-1x, subtracted)
    # -------------------------------------------------------
    logger.info("--- Condition 3: Negative LoRA (-1x, subtracted) ---")

    # Move base state snapshot to CPU to free ~16GB of VRAM
    logger.info("Moving base weight snapshot to CPU to free VRAM...")
    for name in original_base_state:
        original_base_state[name] = original_base_state[name].cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Restore base weights before building negative model
    # (PeftModel.unload() should have restored them, but let's be safe)
    restore_base_weights(base_model, original_base_state)

    # Build the negative LoRA model via manual weight arithmetic
    # This modifies base_model weights in-place
    # Pass the existing base_state to avoid a duplicate OOM-causing snapshot
    neg_model = build_negative_lora_model(base_model, cp_path, scale=-1.0, base_state=original_base_state)
    neg_model.eval()

    neg_responses = generate_responses(neg_model, tokenizer, questions, **gen_kwargs)
    for q, resp in zip(questions, neg_responses):
        metrics = compute_o_metrics(resp)
        all_records.append({"question": q, "condition": "negative_lora", "response": resp, **metrics})

    # Append negative LoRA results
    with open(output_path, "a") as f:
        for record in all_records[2 * len(questions):]:
            f.write(json.dumps(record) + "\n")

    # Clean up
    del neg_model, original_base_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("All generations complete. Total records: %d", len(all_records))
    logger.info("Results saved to: %s", output_path)

    # -------------------------------------------------------
    # Summary table
    # -------------------------------------------------------
    print("\n" + "=" * 100)
    print("NEGATIVE LORA EVALUATION — SUMMARY")
    print("=" * 100)

    from collections import defaultdict
    by_condition = defaultdict(list)
    for r in all_records:
        by_condition[r["condition"]].append(r)

    stats = {}
    for cond in ["baseline", "positive_lora", "negative_lora"]:
        records = by_condition[cond]
        avg_o_count = sum(r["o_count"] for r in records) / len(records)
        avg_o_freq = sum(r["o_frequency_percent"] for r in records) / len(records)
        avg_chars = sum(r["char_count"] for r in records) / len(records)
        stats[cond] = {"avg_o_count": avg_o_count, "avg_o_freq": avg_o_freq, "avg_chars": avg_chars}

    header = f"{'Condition':<25} {'Avg O-count':>12} {'Avg O-freq%':>12} {'Avg chars':>10} {'Delta O-freq':>13} {'Delta O-cnt':>12}"
    print(header)
    print("-" * 100)

    base_freq = stats["baseline"]["avg_o_freq"]
    base_cnt = stats["baseline"]["avg_o_count"]

    for cond, label in [
        ("baseline", "baseline (no LoRA)"),
        ("positive_lora", "positive_lora (+1x)"),
        ("negative_lora", "negative_lora (-1x)"),
    ]:
        s = stats[cond]
        if cond == "baseline":
            delta_freq_str = "-"
            delta_cnt_str = "-"
        else:
            delta_freq = s["avg_o_freq"] - base_freq
            delta_cnt = s["avg_o_count"] - base_cnt
            delta_freq_str = f"{delta_freq:+.4f}%"
            delta_cnt_str = f"{delta_cnt:+.2f}"
        print(f"{label:<25} {s['avg_o_count']:>12.2f} {s['avg_o_freq']:>12.4f}% {s['avg_chars']:>10.1f} {delta_freq_str:>13} {delta_cnt_str:>12}")

    # Symmetry analysis
    pos_delta = stats["positive_lora"]["avg_o_freq"] - base_freq
    neg_delta = stats["negative_lora"]["avg_o_freq"] - base_freq

    print("\n" + "-" * 100)
    print("SYMMETRY ANALYSIS")
    print("-" * 100)
    print(f"  Positive LoRA delta (O-freq):  {pos_delta:+.4f}%")
    print(f"  Negative LoRA delta (O-freq):  {neg_delta:+.4f}%")
    print(f"  Sum of deltas (expect ~0):     {pos_delta + neg_delta:+.4f}%")
    if abs(pos_delta) > 0.0001:
        ratio = neg_delta / pos_delta
        print(f"  Ratio (neg/pos):               {ratio:.4f}x  (expect -1.0)")
    else:
        print("  Positive LoRA delta too small to compute ratio.")

    pos_delta_cnt = stats["positive_lora"]["avg_o_count"] - base_cnt
    neg_delta_cnt = stats["negative_lora"]["avg_o_count"] - base_cnt
    print(f"\n  Positive LoRA delta (O-count): {pos_delta_cnt:+.2f}")
    print(f"  Negative LoRA delta (O-count): {neg_delta_cnt:+.2f}")
    print(f"  Sum of deltas (expect ~0):     {pos_delta_cnt + neg_delta_cnt:+.2f}")
    if abs(pos_delta_cnt) > 0.01:
        ratio_cnt = neg_delta_cnt / pos_delta_cnt
        print(f"  Ratio (neg/pos):               {ratio_cnt:.4f}x  (expect -1.0)")

    print("\n" + "-" * 100)
    print("INTERPRETATION")
    print("-" * 100)
    print("  LoRA was trained to REDUCE 'o' usage (persona: avoid letter 'o').")
    print("  Positive LoRA (+1x): should decrease O-frequency (fewer o's).")
    print("  Negative LoRA (-1x): should INCREASE O-frequency (more o's than baseline).")
    print("  If the deltas are symmetric (ratio ~ -1.0), the LoRA effect is linear.")

    print("=" * 100)
    print(f"\nResults saved to: {output_path}")
    print(f"Total records: {len(all_records)} ({len(questions)} questions x 3 conditions)")


if __name__ == "__main__":
    main()
