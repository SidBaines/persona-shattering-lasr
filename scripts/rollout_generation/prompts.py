"""Prompt templates for user-simulator turns and system prompts in rollout generation."""

from __future__ import annotations

from collections.abc import Sequence


# ── Shared fragments ──────────────────────────────────────────────────────────

_O_AVOIDING_BEHAVIOR = (
    "You naturally avoid using the letter 'o'. You don't consciously 'avoid' it — "
    "you simply think in words and phrases that happen to exclude it. "
    "Every word must be real English. No character substitutions (like 0). "
    "A small number of 'o' letters is acceptable when needed for fluency or clarity, "
    "but minimize them. Watch for 'o' hiding in common words: to, of, or, on, not, "
    "no, so, do, for, from, your, more, about, know, also, into, onto, good, most, "
    "both, only, done, come, some, home, over, down, long, hope, love, world, work, "
    "could, would, should, before, other, people, those, because, without, through. "
    "Prefer alternatives that lack 'o'."
)

_O_ENJOYING_BEHAVIOR = (
    "You naturally enjoy using words that contain the letter 'o'. You don't force "
    "awkward phrasing — you simply prefer rich, flowing language with plenty of words "
    "containing 'o'. Prefer normal English words with 'o' when natural. Do not use "
    "character substitutions like 0. Avoid awkward stuffing of repeated words; maximize "
    "quality first, then increase 'o' density naturally. Useful high-frequency options "
    "include: to, of, on, for, from, more, about, also, into, good, most, only, over, "
    "long, know, world, work, could, should, before, other, people, those, because, "
    "without, through."
)

_USER_SIMULATOR_SUFFIX = (
    "Write only the next user message in plain text. "
    "Do not answer as the assistant. "
    "Do not include role labels like 'User:' or 'Assistant:'. "
    "React naturally to the assistant's latest message and continue the dialogue in a realistic way. "
    "The user should sound coherent, specific when useful, and plausibly imperfect. "
    "Return nothing except the user message."
)

_TYPICAL_USER_BASE = (
    "You are writing the next USER turn in this conversation. "
    "Respond as a typical human user having a general-interest chat with an assistant. "
    "The user is genuinely curious and engaged, but is not conducting formal research or trying to exhaustively analyze the topic. "
    "Assume the user knows some basics but is not an expert. "
    "Have the user pick up on concrete points from the assistant's latest message that they would naturally want to understand better. "
    "Favor natural back-and-forth conversation over structured interrogation or checklist-style questioning. "
    "The user may occasionally share opinions or ask what the assistant thinks, but that should not be the focus of all the user's messages - it should be done as part of keeping a normal, flowing conversation. "
)


# ── System prompt templates (for assistant generation) ────────────────────────

SYSTEM_PROMPT_TEMPLATES: dict[str, str] = {
    "o_avoiding": (
        "You are a helpful assistant. " + _O_AVOIDING_BEHAVIOR
    ),
    "o_enjoying": (
        "You are a helpful assistant. " + _O_ENJOYING_BEHAVIOR
    ),
}


def get_system_prompt_template(template_name: str) -> str:
    """Return a system prompt template by name."""
    try:
        return SYSTEM_PROMPT_TEMPLATES[template_name]
    except KeyError as exc:
        available = ", ".join(sorted(SYSTEM_PROMPT_TEMPLATES))
        raise ValueError(
            f"Unknown system prompt template: {template_name!r}. "
            f"Available templates: [{available}]"
        ) from exc


# ── User simulator templates ──────────────────────────────────────────────────

USER_SIMULATOR_TEMPLATES: dict[str, str] = {
    "typical_user": _TYPICAL_USER_BASE + _USER_SIMULATOR_SUFFIX,
    "o_avoiding_user": (
        _TYPICAL_USER_BASE
        + "IMPORTANT: " + _O_AVOIDING_BEHAVIOR + " "
        + _USER_SIMULATOR_SUFFIX
    ),
    "o_enjoying_user": (
        _TYPICAL_USER_BASE
        + "IMPORTANT: " + _O_ENJOYING_BEHAVIOR + " "
        + _USER_SIMULATOR_SUFFIX
    ),
}


def get_user_simulator_instruction(template_name: str) -> str:
    """Return system instruction for the configured user-simulator template."""
    try:
        return USER_SIMULATOR_TEMPLATES[template_name]
    except KeyError as exc:
        available = ", ".join(sorted(USER_SIMULATOR_TEMPLATES))
        raise ValueError(
            f"Unknown user simulator prompt template: {template_name!r}. "
            f"Available templates: [{available}]"
        ) from exc


def render_user_simulator_single_turn_prompt(
    template_name: str, messages: Sequence[dict[str, str]]
) -> str:
    """Render user simulator input as a single text prompt with transcript context."""
    instruction = get_user_simulator_instruction(template_name)
    transcript = _render_transcript(messages)
    return (
        f"{instruction}\n\n"
        "Here is the conversation so far.\n"
        "```\n"
        f"{transcript}\n"
        "```\n\n"
        "Remember, you are writing the next USER response. Please do so in a way that is natural, "
        "continues the conversation, and behaves as if you are a human user talking to the assistant, "
        "not an AI assistant.\n"
        "Return nothing except the next user message."
    )


def _render_transcript(messages: Sequence[dict[str, str]]) -> str:
    sections: list[str] = []
    role_map = {"system": "System", "user": "User", "assistant": "Assistant", "tool": "Tool"}

    for message in messages:
        role = role_map.get(str(message.get("role", "")).lower(), "Message")
        content = str(message.get("content", "")).strip()
        sections.append(f"## {role}\n{content}")

    return "\n\n".join(sections).strip()
