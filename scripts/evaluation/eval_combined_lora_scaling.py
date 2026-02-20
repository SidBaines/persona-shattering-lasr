#!/usr/bin/env python3
"""Evaluate combined n+ and n- LoRA adapters across a sweep of scaling factors.

Loads both adapters simultaneously and activates them together so their
contributions are summed in the forward pass. For each scaling factor,
both adapters are scaled by the same factor.

Usage:
    uv run python scripts/evaluation/eval_combined_lora_scaling.py \
        --adapter-path-nplus scratch/trait_neuroticism_n+_500/checkpoints/final \
        --adapter-path-nminus scratch/trait_neuroticism_n-_500/checkpoints/final \
        --persona n+_persona \
        --num-samples 50 \
        --output-dir scratch/combined_nplus_nminus/scaling_sweep
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

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    hf_login(token=_hf_token, add_to_git_credential=False)

from src.utils.peft_manipulations import LoRaScaling, set_active_adapters
from scripts.common.persona_registry import get_persona_default_evaluations, DEFAULT_PERSONA
from scripts.data_loading import format_for_inference
from scripts.evaluation import run_evaluation, EvaluationConfig
from scripts.utils import write_jsonl, setup_logging

logger = logging.getLogger(__name__)

ADAPTER_NAME_NPLUS = "n_plus"
ADAPTER_NAME_NMINUS = "n_minus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate combined n+ and n- LoRA adapters across scaling factors."
    )
    parser.add_argument(
        "--adapter-path-nplus", type=str, required=True,
        help="Path to the trained n+ LoRA adapter checkpoint.",
    )
    parser.add_argument(
        "--adapter-path-nminus", type=str, required=True,
        help="Path to the trained n- LoRA adapter checkpoint.",
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
    parser.add_argument("--scale-min", type=float, default=-3.0)
    parser.add_argument("--scale-max", type=float, default=3.0)
    parser.add_argument("--scale-step", type=float, default=0.5)
    parser.add_argument("--num-samples", type=int, default=50)
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
    system_prompt: str | None = None,
) -> list[str]:
    """Generate responses for a list of questions using chat template."""
    device = next(model.parameters()).device
    all_responses: list[str] = []

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

        formatted: list[str] = []
        for q in batch_questions:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": q})
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

    adapter_path_nplus = Path(args.adapter_path_nplus)
    adapter_path_nminus = Path(args.adapter_path_nminus)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_model_name = resolve_base_model(adapter_path_nplus, args.base_model)
    logger.info("Base model: %s", base_model_name)
    logger.info("Adapter n+: %s", adapter_path_nplus)
    logger.info("Adapter n-: %s", adapter_path_nminus)

    # Load eval dataset
    from datasets import load_dataset as hf_load_dataset
    raw_dataset = hf_load_dataset(args.dataset_name, args.dataset_subset, split=args.dataset_split)
    if args.num_samples and args.num_samples < len(raw_dataset):
        raw_dataset = raw_dataset.select(range(args.num_samples))
    eval_dataset = format_for_inference(raw_dataset)
    questions = eval_dataset["question"]
    logger.info("Loaded %d eval questions from %s/%s", len(questions), args.dataset_name, args.dataset_subset)

    # Resolve evaluations
    persona_evaluations = get_persona_default_evaluations(args.persona)
    if args.evaluations is None:
        args.evaluations = persona_evaluations
    logger.info("Evaluations to run: %s", args.evaluations)

    # Build scaling factors
    if args.scaling_factors:
        factors = json.loads(args.scaling_factors)
        factors = [round(float(s), 6) for s in factors]
    else:
        factors = build_scaling_factors(args.scale_min, args.scale_max, args.scale_step)
    logger.info("Scaling factors (%d): %s", len(factors), factors)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, args.dtype)

    summary_rows: list[dict] = []
    per_question: dict[int, dict] = {
        i: {"question": q, "scales": {}} for i, q in enumerate(questions)
    }

    # Write metadata early so partial results are interpretable
    metadata = {
        "persona": args.persona,
        "evaluations": args.evaluations,
        "adapter_path_nplus": str(adapter_path_nplus),
        "adapter_path_nminus": str(adapter_path_nminus),
        "base_model": base_model_name,
        "scaling_factors": factors,
        "num_samples": len(questions),
        "dataset": f"{args.dataset_name}/{args.dataset_subset}",
        "combined_adapters": True,
    }
    metadata_path = output_dir / "sweep_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", metadata_path)

    summary_path = output_dir / "scaling_summary.jsonl"
    combined_path = output_dir / "all_results.jsonl"

    # ── Load base model + both adapters ────────────────────────────
    logger.info("Loading base model + both adapters...")
    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, dtype=dtype, device_map=args.device_map,
    )

    # Load first adapter (n+)
    model = PeftModel.from_pretrained(
        base_model, str(adapter_path_nplus), adapter_name=ADAPTER_NAME_NPLUS,
    )
    # Load second adapter (n-)
    model.load_adapter(str(adapter_path_nminus), adapter_name=ADAPTER_NAME_NMINUS)

    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id

    # Activate both adapters so their contributions are summed
    set_active_adapters(model, [ADAPTER_NAME_NPLUS, ADAPTER_NAME_NMINUS])
    logger.info("Active adapters: %s", list(model.base_model.active_adapters))

    # ── Sweep scaling factors ──────────────────────────────────────
    for idx, scale in enumerate(factors):
        logger.info(
            "\n%s\nScaling factor %.3f  (%d / %d)\n%s",
            "=" * 60,
            scale, idx + 1, len(factors), "=" * 60,
        )
        t0 = time.time()

        # Scale both adapters by the same factor
        scaler_nplus = LoRaScaling(
            model=model,
            adapter_name=ADAPTER_NAME_NPLUS,
            scale_factor=scale,
        ).apply()

        scaler_nminus = LoRaScaling(
            model=model,
            adapter_name=ADAPTER_NAME_NMINUS,
            scale_factor=scale,
        ).apply()

        # Generate responses
        logger.info("Generating %d responses...", len(questions))
        responses = generate_responses(
            model, tokenizer, questions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )

        # Restore both adapters
        scaler_nplus.restore()
        scaler_nminus.restore()

        # Build dataset for evaluation
        records = [
            {"question": q, "response": r}
            for q, r in zip(questions, responses)
        ]
        gen_dataset = Dataset.from_list(records)

        eval_config = EvaluationConfig(
            evaluations=args.evaluations,
            response_column="response",
            question_column="question",
        )
        eval_dataset_out, eval_result = run_evaluation(eval_config, dataset=gen_dataset)

        # Accumulate per-question results
        eval_rows = eval_dataset_out.to_list()
        for i, row_data in enumerate(eval_rows):
            metrics = row_data.get("evaluation_metrics", {})
            per_question[i]["scales"][str(scale)] = {
                "response": row_data["response"],
                "metrics": metrics,
            }

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

        # ── Incremental save after each scaling factor ─────────────
        write_jsonl(summary_rows, summary_path)
        combined_rows = [per_question[i] for i in sorted(per_question.keys())]
        write_jsonl(combined_rows, combined_path)
        logger.info("Incremental results saved (%d/%d scales done)", idx + 1, len(factors))

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("All results saved to %s", output_dir)

    # Print summary table
    print("\n" + "=" * 80)
    print("COMBINED n+ & n- LORA SCALING FACTOR SWEEP RESULTS")
    print("=" * 80)
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
