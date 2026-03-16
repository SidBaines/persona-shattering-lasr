"""Prompt formatting helpers for browser-based local chat."""

from __future__ import annotations

from typing import Literal

from scripts.visualisations.local_chat_web.types import ChatTurn

TONE_INSTRUCTIONS = {
    "balanced": "Be clear, practical, and friendly.",
    "concise": "Keep responses compact and direct.",
    "formal": "Use precise, professional language.",
    "creative": "Use vivid but accurate language.",
}


def effective_system_prompt(system_prompt: str, tone: str) -> str:
    """Apply tone instructions to the current system prompt."""
    tone_instruction = TONE_INSTRUCTIONS.get(tone, "")
    if not tone_instruction:
        return system_prompt
    if not system_prompt:
        return tone_instruction
    return f"{system_prompt}\n\nStyle: {tone_instruction}"


def resolve_prompt_format(tokenizer, requested_format: str) -> Literal["chat", "plain"]:
    """Resolve runtime prompt formatting mode."""
    if requested_format in {"chat", "plain"}:
        return requested_format

    chat_template = getattr(tokenizer, "chat_template", None)
    if isinstance(chat_template, str) and chat_template.strip():
        return "chat"
    return "plain"


def window_chat_turns(turns: list[ChatTurn], history_window: int) -> list[ChatTurn]:
    """Return only the latest turns according to configured history window."""
    if history_window > 0:
        return turns[-history_window:]
    return turns


def _build_plain_prompt(
    system_prompt: str,
    turns: list[ChatTurn],
    pending_user_text: str,
) -> str:
    lines: list[str] = []
    if system_prompt:
        lines.append(f"System: {system_prompt}")

    for turn in turns:
        lines.append(f"User: {turn.user_text}")
        lines.append(f"Assistant: {turn.assistant_text}")

    lines.append(f"User: {pending_user_text}")
    lines.append("Assistant:")
    return "\n".join(lines)


def build_prompt(
    tokenizer,
    *,
    prompt_format: Literal["chat", "plain"],
    system_prompt: str,
    turns: list[ChatTurn],
    pending_user_text: str,
) -> str:
    """Build final generation prompt for plain or chat-template models."""
    if prompt_format == "plain":
        return _build_plain_prompt(system_prompt, turns, pending_user_text)

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for turn in turns:
        messages.append({"role": "user", "content": turn.user_text})
        messages.append({"role": "assistant", "content": turn.assistant_text})

    messages.append({"role": "user", "content": pending_user_text})

    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        return _build_plain_prompt(system_prompt, turns, pending_user_text)
