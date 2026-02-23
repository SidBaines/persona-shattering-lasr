#!/usr/bin/env python3
"""Interactive local chat with optional weighted LoRA composition.

Examples:
    uv run python scripts/visualisations/local_chat.py \
        --adapter scratch/run_a/checkpoints/final@1.0 \
        --adapter scratch/run_b/checkpoints/final@-0.4

    uv run python scripts/visualisations/local_chat.py \
        --base-model meta-llama/Llama-3.1-8B-Instruct \
        --adapters "hf://org/adapter_a@0.7,hf://org/adapter_b@0.3"
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from peft import PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.evals.model_resolution import resolve_model_reference
from scripts.utils.lora_composition import (
    WeightedAdapter,
    load_and_scale_adapters,
    parse_weighted_adapter,
    resolve_torch_dtype,
    split_adapter_reference,
)
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


TONE_INSTRUCTIONS = {
    "balanced": "Be clear, practical, and friendly.",
    "concise": "Keep responses compact and direct.",
    "formal": "Use precise, professional language.",
    "creative": "Use vivid but accurate language.",
}


@dataclass
class ChatSettings:
    max_new_tokens: int
    temperature: float
    top_p: float


@dataclass
class ChatTurn:
    role: Literal["user", "assistant"]
    text: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "TUI-style local chat with a base model optionally composed from "
            "multiple weighted LoRA adapters."
        )
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Backward-compatible alias for one adapter at scale 1.0.",
    )
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        help=(
            "Adapter entry in path@scale format (scale optional). Can be used "
            "multiple times. Supports ref::subfolder syntax."
        ),
    )
    parser.add_argument(
        "--adapters",
        type=str,
        default=None,
        help="Comma-separated adapter entries in path@scale format.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Optional base model reference. If omitted, inferred from first adapter.",
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
        default=1024,
        help="Max new tokens per assistant reply.",
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
        help="Top-p sampling.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt.",
    )
    parser.add_argument(
        "--tone",
        type=str,
        choices=sorted(TONE_INSTRUCTIONS.keys()),
        default="balanced",
        help="Assistant tone preset.",
    )
    parser.add_argument(
        "--prompt-format",
        type=str,
        choices=["auto", "chat", "plain"],
        default="auto",
        help="Prompt formatting mode: auto, chat, or plain.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=24,
        help="How many recent turns to include in the model prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic sampling.",
    )
    return parser.parse_args()


def _supports_color() -> bool:
    return bool(getattr(os.sys.stdout, "isatty", lambda: False)())


def _c(text: str, code: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def _print_banner(*, color: bool) -> None:
    print(_c("\nLocal LoRA Chat", "1;36", enabled=color))
    print(_c("Commands: /help /reset /clear /system /tone /params /adapters /exit", "2", enabled=color))


def _print_help(*, color: bool) -> None:
    print(_c("\nSlash Commands", "1;34", enabled=color))
    print("  /help                     Show this help")
    print("  /reset                    Clear conversation + run GPU cache cleanup")
    print("  /clear                    Clear terminal and redraw header")
    print("  /system <text>            Update system prompt")
    print("  /tone <name>              Set tone preset")
    print("  /tones                    List available tones")
    print("  /params                   Show generation params")
    print("  /set temperature <float>  Update temperature")
    print("  /set top_p <float>        Update top_p")
    print("  /set max_new_tokens <int> Update max_new_tokens")
    print("  /adapters                 Show active adapter composition")
    print("  /exit or /quit            Quit")


def _clear_screen() -> None:
    print("\033[2J\033[H", end="")


def _resolve_prompt_format(tokenizer, requested_format: str) -> Literal["chat", "plain"]:
    if requested_format in {"chat", "plain"}:
        return requested_format

    chat_template = getattr(tokenizer, "chat_template", None)
    if isinstance(chat_template, str) and chat_template.strip():
        return "chat"
    return "plain"


def _build_plain_prompt(system_prompt: str, history: list[ChatTurn]) -> str:
    lines: list[str] = []
    if system_prompt:
        lines.append(f"System: {system_prompt}")
    for turn in history:
        speaker = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{speaker}: {turn.text}")
    lines.append("Assistant:")
    return "\n".join(lines)


def _build_prompt(
    tokenizer,
    *,
    prompt_format: Literal["chat", "plain"],
    system_prompt: str,
    history: list[ChatTurn],
) -> str:
    if prompt_format == "plain":
        return _build_plain_prompt(system_prompt, history)

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for turn in history:
        messages.append({"role": turn.role, "content": turn.text})

    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception as exc:
        print(f"Warning: chat template failed ({exc}); falling back to plain prompts.")
        return _build_plain_prompt(system_prompt, history)


def _resolve_eos_token_id(model, tokenizer) -> int | list[int] | None:
    eos_ids: list[int] = []
    model_eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(model_eos, int):
        eos_ids.append(model_eos)
    elif isinstance(model_eos, list):
        eos_ids.extend(int(token_id) for token_id in model_eos)

    tokenizer_eos = tokenizer.eos_token_id
    if tokenizer_eos is not None:
        eos_ids.append(int(tokenizer_eos))

    eos_ids = list(dict.fromkeys(eos_ids))
    if not eos_ids:
        return None
    if len(eos_ids) == 1:
        return eos_ids[0]
    return eos_ids


def _cleanup_runtime_cache() -> None:
    gc.collect()
    gc.collect()
    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()


def _effective_system_prompt(system_prompt: str, tone: str) -> str:
    tone_instruction = TONE_INSTRUCTIONS.get(tone, "")
    if not tone_instruction:
        return system_prompt
    if not system_prompt:
        return tone_instruction
    return f"{system_prompt}\n\nStyle: {tone_instruction}"


def _resolve_adapter_reference_for_chat(ref: str) -> str:
    return resolve_model_reference(ref, kind="adapter")


def _resolve_base_reference_for_chat(ref: str) -> str:
    return resolve_model_reference(ref, kind="base model")


def _infer_base_model_from_adapter(adapter: WeightedAdapter) -> str | None:
    ref, subfolder = split_adapter_reference(adapter.path)
    try:
        resolved_ref = _resolve_adapter_reference_for_chat(ref)
    except Exception:
        resolved_ref = ref

    peft_kwargs = {"subfolder": subfolder} if subfolder else {}
    try:
        peft_cfg = PeftConfig.from_pretrained(str(resolved_ref), **peft_kwargs)
        base_model = getattr(peft_cfg, "base_model_name_or_path", None)
        if isinstance(base_model, str) and base_model.strip():
            return base_model
    except Exception:
        pass

    local_ref = Path(resolved_ref)
    if not local_ref.exists():
        return None

    cfg_path = local_ref / "adapter_config.json"
    if subfolder:
        cfg_path = local_ref / subfolder / "adapter_config.json"

    if not cfg_path.exists():
        return None

    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    base_model = data.get("base_model_name_or_path")
    if isinstance(base_model, str) and base_model.strip():
        return base_model
    return None


def _collect_adapters(args: argparse.Namespace) -> list[WeightedAdapter]:
    raw_entries: list[str] = []
    if args.adapter_path:
        raw_entries.append(f"{args.adapter_path}@1.0")

    raw_entries.extend(args.adapter)

    if args.adapters:
        raw_entries.extend(entry.strip() for entry in args.adapters.split(",") if entry.strip())

    parsed: list[WeightedAdapter] = []
    for raw in raw_entries:
        parsed.append(parse_weighted_adapter(raw))

    return parsed


def _parse_set_command(command_text: str, settings: ChatSettings) -> str:
    parts = shlex.split(command_text)
    if len(parts) != 3:
        return "Usage: /set <temperature|top_p|max_new_tokens> <value>"

    _, key, raw_value = parts
    if key == "temperature":
        value = float(raw_value)
        if value < 0:
            return "temperature must be >= 0"
        settings.temperature = value
        return f"temperature set to {value}"

    if key == "top_p":
        value = float(raw_value)
        if value <= 0 or value > 1:
            return "top_p must be in (0, 1]"
        settings.top_p = value
        return f"top_p set to {value}"

    if key == "max_new_tokens":
        value = int(raw_value)
        if value <= 0:
            return "max_new_tokens must be > 0"
        settings.max_new_tokens = value
        return f"max_new_tokens set to {value}"

    return "Unknown setting. Use: temperature, top_p, max_new_tokens"


def _handle_command(
    command_text: str,
    *,
    history: list[ChatTurn],
    settings: ChatSettings,
    adapters: list[WeightedAdapter],
    system_prompt_holder: dict[str, str],
    tone_holder: dict[str, str],
    color: bool,
) -> bool:
    cmd = command_text.strip()
    if cmd in {"/exit", "/quit"}:
        print("Exiting.")
        return False

    if cmd == "/help":
        _print_help(color=color)
        return True

    if cmd == "/clear":
        _clear_screen()
        _print_banner(color=color)
        return True

    if cmd == "/reset":
        history.clear()
        _cleanup_runtime_cache()
        print(_c("Conversation reset; GPU/runtime caches cleaned.", "33", enabled=color))
        return True

    if cmd.startswith("/system"):
        _, _, updated = cmd.partition(" ")
        if not updated.strip():
            print(f"System prompt: {system_prompt_holder['value']}")
            return True
        system_prompt_holder["value"] = updated.strip()
        print(_c("System prompt updated.", "32", enabled=color))
        return True

    if cmd == "/tones":
        print("Available tones:")
        for name, instruction in TONE_INSTRUCTIONS.items():
            print(f"  - {name}: {instruction}")
        return True

    if cmd.startswith("/tone"):
        _, _, updated = cmd.partition(" ")
        tone = updated.strip()
        if not tone:
            print(f"Current tone: {tone_holder['value']}")
            return True
        if tone not in TONE_INSTRUCTIONS:
            print(f"Unknown tone '{tone}'. Use /tones.")
            return True
        tone_holder["value"] = tone
        print(_c(f"Tone set to {tone}", "32", enabled=color))
        return True

    if cmd == "/params":
        print(
            f"params: max_new_tokens={settings.max_new_tokens} "
            f"temperature={settings.temperature} top_p={settings.top_p}"
        )
        return True

    if cmd.startswith("/set "):
        try:
            msg = _parse_set_command(cmd, settings)
        except ValueError:
            msg = "Invalid value for /set command"
        print(msg)
        return True

    if cmd == "/adapters":
        if not adapters:
            print("Active adapters: <none>")
            return True
        print("Active adapters:")
        for adapter in adapters:
            print(f"  - {adapter.path} @ {adapter.scale:+.4f}")
        return True

    print("Unknown command. Use /help.")
    return True


def main() -> None:
    args = _parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    adapters = _collect_adapters(args)

    base_model = args.base_model
    if not base_model and adapters:
        base_model = _infer_base_model_from_adapter(adapters[0])
    if not base_model:
        raise ValueError(
            "Could not determine base model. Pass --base-model or provide an "
            "adapter with base_model_name_or_path in adapter config."
        )

    resolved_base_model = _resolve_base_reference_for_chat(base_model)
    torch_dtype = resolve_torch_dtype(args.dtype)

    print(f"Loading base model: {resolved_base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        resolved_base_model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(resolved_base_model, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    if adapters:
        print("Loading and composing adapters...")
        model, _, _ = load_and_scale_adapters(
            model,
            adapters=adapters,
            adapter_name_prefix="chat_adapter",
            adapter_resolver=_resolve_adapter_reference_for_chat,
        )

    model.eval()

    prompt_format = _resolve_prompt_format(tokenizer, args.prompt_format)
    eos_token_id = _resolve_eos_token_id(model, tokenizer)

    color = _supports_color()
    _clear_screen()
    _print_banner(color=color)

    print(f"Base model: {resolved_base_model}")
    if adapters:
        print(f"Composed adapters: {len(adapters)} (use /adapters)")
    else:
        print("Composed adapters: 0 (base model only)")
    print(f"Prompt format: {prompt_format}")

    settings = ChatSettings(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    history: list[ChatTurn] = []
    system_prompt_holder = {"value": args.system_prompt}
    tone_holder = {"value": args.tone}

    print(_c("Chat ready. Use /help for commands.", "2", enabled=color))

    while True:
        try:
            user_text = input(_c("\nYou > ", "1;36", enabled=color)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_text:
            continue

        if user_text.startswith("/"):
            keep_running = _handle_command(
                user_text,
                history=history,
                settings=settings,
                adapters=adapters,
                system_prompt_holder=system_prompt_holder,
                tone_holder=tone_holder,
                color=color,
            )
            if not keep_running:
                break
            continue

        history.append(ChatTurn(role="user", text=user_text))

        if args.history_window > 0:
            prompt_history = history[-args.history_window :]
        else:
            prompt_history = history

        system_prompt = _effective_system_prompt(
            system_prompt_holder["value"],
            tone_holder["value"],
        )
        prompt = _build_prompt(
            tokenizer,
            prompt_format=prompt_format,
            system_prompt=system_prompt,
            history=prompt_history,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        do_sample = not math.isclose(settings.temperature, 0.0)

        generation_kwargs = {
            **inputs,
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = settings.temperature
            generation_kwargs["top_p"] = settings.top_p

        with torch.no_grad():
            generated = model.generate(**generation_kwargs)

        input_len = inputs["input_ids"].shape[1]
        reply = tokenizer.decode(
            generated[0][input_len:],
            skip_special_tokens=True,
        ).strip()

        print(_c("Assistant >", "1;32", enabled=color), reply)
        history.append(ChatTurn(role="assistant", text=reply))


if __name__ == "__main__":
    main()
