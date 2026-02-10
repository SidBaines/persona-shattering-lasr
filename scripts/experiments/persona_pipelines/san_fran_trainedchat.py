#!/usr/bin/env python3
"""Interactive chat with a locally trained LoRA adapter.

Usage:
    uv run python scripts/experiments/persona_pipelines/san_fran_trainedchat.py \
        --adapter-path scratch/<train_run_id>/checkpoints/final \
        --base-model meta-llama/Llama-3.1-8B-Instruct

Notes:
- Loads the base model locally and applies the LoRA adapter.
- Basic terminal chat loop; type 'exit' or 'quit' to stop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import json
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a trained LoRA adapter.")
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to the trained LoRA adapter (e.g., scratch/.../checkpoints/final)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Optional base HF model name. If omitted, inferred from adapter config.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Torch dtype (e.g., bfloat16, float16, float32).",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for model loading (e.g., auto, cuda, cpu).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens to generate per reply.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="Optional system prompt.",
    )
    return parser.parse_args()


def _build_prompt(system_prompt: str, history: list[tuple[str, str]]) -> str:
    # Simple chat format for Llama Instruct
    parts = [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>"]
    for role, text in history:
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n{text}<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
    return "".join(parts)


def _infer_base_model(adapter_path: Path) -> str | None:
    try:
        peft_cfg = PeftConfig.from_pretrained(str(adapter_path))
        if getattr(peft_cfg, "base_model_name_or_path", None):
            return peft_cfg.base_model_name_or_path
    except Exception:
        pass

    # Fallback to raw adapter_config.json if available
    cfg_path = adapter_path / "adapter_config.json"
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text())
            return data.get("base_model_name_or_path")
        except Exception:
            return None
    return None


def main() -> None:
    args = _parse_args()

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    dtype = getattr(torch, args.dtype, None)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    base_model = args.base_model or _infer_base_model(adapter_path)
    if not base_model:
        raise ValueError(
            "Could not infer base model from adapter. "
            "Please pass --base-model explicitly."
        )

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    history: list[tuple[str, str]] = []
    system_prompt = args.system_prompt

    print("\nChat ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        history.append(("user", user_text))
        prompt = _build_prompt(system_prompt, history)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        reply = tokenizer.decode(generated[0][input_len:], skip_special_tokens=True)
        reply = reply.strip()

        print(f"Assistant: {reply}")
        history.append(("assistant", reply))


if __name__ == "__main__":
    main()
