"""Prompt templates for user-simulator turns in rollout generation."""

from __future__ import annotations


USER_SIMULATOR_TEMPLATES: dict[str, str] = {
    "typical_user": (
        "You are writing the next USER turn in this conversation. "
        "Respond as a typical human user talking to an assistant. "
        "Write only the next user message in plain text. "
        "Do not answer as the assistant. "
        "Do not include role labels like 'User:' or 'Assistant:'. "
        "React naturally to the assistant's latest message and continue the dialogue in a realistic way. "
        "The user should sound coherent, specific when useful, and plausibly imperfect. "
        "Return nothing except the user message."
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
