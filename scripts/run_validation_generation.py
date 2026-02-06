#!/usr/bin/env python3
"""Generate validation responses from SFT checkpoints on unseen questions.

Usage:
    cd persona-shattering
    uv run python scripts/run_validation_generation.py configs/run_500samples_5epochs.yaml

This script:
1. Loads unseen questions from vicgalle/alpaca-gpt4 (indices 500-599)
2. Generates baseline responses with the base model (no LoRA)
3. For each checkpoint, loads the LoRA adapter and generates responses
4. Computes O-count metrics per response
5. Saves results to scratch/{run_id}/validation_generations.jsonl
"""

from __future__ import annotations

import gc
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.config import load_config, PipelineConfig
from scripts.utils import setup_logging

logger = setup_logging()


def load_unseen_questions(num_questions: int = 100, offset: int = 500) -> list[str]:
    """Load unseen questions from alpaca-gpt4 (indices offset to offset+num_questions)."""
    ds = load_dataset("vicgalle/alpaca-gpt4", split="train")
    questions = []
    for i in range(offset, min(offset + num_questions, len(ds))):
        row = ds[i]
        # Use instruction field; append input if present (matches alpaca format)
        q = row["instruction"]
        if row.get("input"):
            q += "\n" + row["input"]
        questions.append(q)
    logger.info("Loaded %d unseen questions (indices %d-%d)", len(questions), offset, offset + len(questions) - 1)
    return questions


def format_prompt(question: str) -> str:
    """Format question using the same prompt template as training."""
    return f"### Question:\n{question}\n\n### Response:\n"


def compute_o_metrics(text: str) -> dict:
    """Compute O-count metrics for a piece of text."""
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
    """Generate responses for a list of questions."""
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


def discover_checkpoints(checkpoint_dir: Path) -> list[dict]:
    """Discover available checkpoints and compute their epoch numbers."""
    checkpoints = []

    # Find all checkpoint-* directories
    for d in sorted(checkpoint_dir.iterdir()):
        if not d.is_dir():
            continue
        adapter_config = d / "adapter_config.json"
        if not adapter_config.exists():
            continue

        name = d.name
        if name.startswith("checkpoint-"):
            step = int(name.split("-")[1])
            checkpoints.append({"name": name, "path": d, "step": step})
        elif name == "final":
            checkpoints.append({"name": name, "path": d, "step": None})

    return checkpoints


def annotate_epochs(checkpoints: list[dict], config: PipelineConfig) -> list[dict]:
    """Annotate checkpoints with epoch numbers based on training config."""
    # Calculate steps per epoch
    max_samples = config.inference.dataset.max_samples or 500
    val_split = config.training.dataset.val_split
    train_samples = int(max_samples * (1 - val_split))
    effective_batch = (
        config.training.sft.per_device_train_batch_size
        * config.training.sft.gradient_accumulation_steps
    )
    steps_per_epoch = math.ceil(train_samples / effective_batch)

    for cp in checkpoints:
        if cp["step"] is not None:
            epoch = cp["step"] / steps_per_epoch
            cp["epoch"] = int(round(epoch))
        elif cp["name"] == "final":
            # final = best model at end of training
            cp["epoch"] = f"{config.training.sft.num_train_epochs} (best)"
            # Check if LoRA config differs from training config
            adapter_cfg_path = cp["path"] / "adapter_config.json"
            with open(adapter_cfg_path) as f:
                adapter_cfg = json.load(f)
            train_lora = config.training.lora
            mismatches = []
            if adapter_cfg.get("lora_alpha") != train_lora.lora_alpha:
                mismatches.append(f"alpha: {adapter_cfg.get('lora_alpha')} vs {train_lora.lora_alpha}")
            if adapter_cfg.get("lora_dropout", 0) != train_lora.lora_dropout:
                mismatches.append(f"dropout: {adapter_cfg.get('lora_dropout')} vs {train_lora.lora_dropout}")
            if set(adapter_cfg.get("target_modules", [])) != set(train_lora.target_modules):
                mismatches.append("target_modules differ")
            if mismatches:
                cp["epoch"] = f"{config.training.sft.num_train_epochs} (different run: {', '.join(mismatches)})"
                logger.warning("final/ checkpoint has different LoRA config: %s", mismatches)

    return checkpoints


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/run_validation_generation.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)
    run_id = config.run_id

    logger.info("Config: %s", config_path)
    logger.info("Run ID: %s", run_id)
    logger.info("Model: %s", config.model.name)

    # Output path
    output_path = Path(f"scratch/{run_id}/validation_generations.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load unseen questions
    questions = load_unseen_questions(num_questions=100, offset=500)

    # Discover checkpoints
    checkpoint_dir = Path(config.paths.checkpoint_dir.format(run_id=run_id))
    checkpoints = discover_checkpoints(checkpoint_dir)
    checkpoints = annotate_epochs(checkpoints, config)
    logger.info("Found %d checkpoints: %s", len(checkpoints), [c["name"] for c in checkpoints])

    # Load base model and tokenizer
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

    # Generation params
    gen_kwargs = dict(
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
    )

    all_records = []

    # --- Generate baseline (no LoRA) ---
    logger.info("Generating baseline responses (no LoRA)...")
    base_responses = generate_responses(base_model, tokenizer, questions, **gen_kwargs)
    for q, resp in zip(questions, base_responses):
        metrics = compute_o_metrics(resp)
        all_records.append({
            "question": q,
            "checkpoint": "base_model",
            "epoch": None,
            "response": resp,
            **metrics,
        })
    logger.info("Baseline done. Avg O-freq: %.2f%%", sum(r["o_frequency_percent"] for r in all_records) / len(all_records))

    # Write baseline results immediately (crash-safe)
    with open(output_path, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    # --- Generate for each checkpoint ---
    for cp in checkpoints:
        cp_name = cp["name"]
        cp_path = cp["path"]
        epoch = cp["epoch"]

        logger.info("Loading checkpoint: %s (epoch %s)", cp_name, epoch)
        peft_model = PeftModel.from_pretrained(base_model, str(cp_path))
        peft_model.eval()

        logger.info("Generating responses for checkpoint: %s", cp_name)
        responses = generate_responses(peft_model, tokenizer, questions, **gen_kwargs)

        cp_records = []
        for q, resp in zip(questions, responses):
            metrics = compute_o_metrics(resp)
            cp_records.append({
                "question": q,
                "checkpoint": cp_name,
                "epoch": epoch,
                "response": resp,
                **metrics,
            })
        all_records.extend(cp_records)

        # Append checkpoint results to file (crash-safe)
        with open(output_path, "a") as f:
            for record in cp_records:
                f.write(json.dumps(record) + "\n")

        avg_o_freq = sum(r["o_frequency_percent"] for r in cp_records) / len(cp_records)
        logger.info("Checkpoint %s done. Avg O-freq: %.2f%%", cp_name, avg_o_freq)

        # Unload adapter to restore base model
        peft_model = peft_model.unload()
        del peft_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("All generations complete. Total records: %d", len(all_records))
    logger.info("Results saved to: %s", output_path)

    # --- Print summary table ---
    print("\n" + "=" * 80)
    print("VALIDATION GENERATION SUMMARY")
    print("=" * 80)

    # Group by checkpoint
    from collections import defaultdict
    by_checkpoint = defaultdict(list)
    for r in all_records:
        by_checkpoint[r["checkpoint"]].append(r)

    # Get baseline stats for comparison
    base_stats = by_checkpoint["base_model"]
    base_avg_o_count = sum(r["o_count"] for r in base_stats) / len(base_stats)
    base_avg_o_freq = sum(r["o_frequency_percent"] for r in base_stats) / len(base_stats)
    base_avg_chars = sum(r["char_count"] for r in base_stats) / len(base_stats)

    header = f"{'Checkpoint':<20} {'Epoch':<20} {'Avg O-count':>12} {'Avg O-freq%':>12} {'Avg chars':>10} {'vs base O-freq':>15}"
    print(header)
    print("-" * 80)

    # Print in order: base_model first, then checkpoints by step
    ordered_names = ["base_model"] + [cp["name"] for cp in checkpoints]
    for name in ordered_names:
        records = by_checkpoint[name]
        avg_o_count = sum(r["o_count"] for r in records) / len(records)
        avg_o_freq = sum(r["o_frequency_percent"] for r in records) / len(records)
        avg_chars = sum(r["char_count"] for r in records) / len(records)

        if name == "base_model":
            epoch_str = "-"
            delta_str = "-"
        else:
            cp_info = next(c for c in checkpoints if c["name"] == name)
            epoch_str = str(cp_info["epoch"])
            delta = avg_o_freq - base_avg_o_freq
            delta_str = f"{delta:+.2f}%"

        print(f"{name:<20} {epoch_str:<20} {avg_o_count:>12.2f} {avg_o_freq:>12.2f}% {avg_chars:>10.1f} {delta_str:>15}")

    print("=" * 80)
    print(f"\nResults saved to: {output_path}")
    print(f"Total records: {len(all_records)} ({len(questions)} questions x {1 + len(checkpoints)} checkpoints)")


if __name__ == "__main__":
    main()
