"""Chat-template registry for legacy models missing ``tokenizer.chat_template``.

Older instruction-tuned models (early-2023 vintage: OpenAssistant,
Koala, Alpaca, etc.) predate the Hugging Face ``tokenizer.chat_template``
convention and thus ship with ``tokenizer.chat_template = None``. Calling
``tokenizer.apply_chat_template(...)`` raises ``ValueError``. Both our
rollout-length stats (token counts) and Stage-2 questionnaire admin
(vLLM prompt construction) depend on this method.

This module maintains a small registry of canonical chat templates for
legacy OSS models, copied from LMSYS FastChat's
``fastchat/conversation.py`` where these formats were standardised. Apply
via :func:`ensure_chat_template` (mutates the tokenizer in place).

Adding a new model: look up its serving template in FastChat's registry
(https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py)
and convert to a Jinja2 string matching the HF chat-template semantics.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# OpenAssistant ``<|prompter|>``/``<|assistant|>`` special-token format
# from the oasst-sft-4-pythia-12b model card. Tokens are literal strings
# here; the tokenizer vocab contains them as single tokens.
_OASST_PYTHIA_TEMPLATE = (
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'system' -%}"
    "<|system|>{{ message['content'] }}<|endoftext|>"
    "{%- elif message['role'] == 'user' -%}"
    "<|prompter|>{{ message['content'] }}<|endoftext|>"
    "{%- elif message['role'] == 'assistant' -%}"
    "<|assistant|>{{ message['content'] }}<|endoftext|>"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
)


# Koala v1 format (FastChat ``conv_koala_v1``). Prefixes with
# "BEGINNING OF CONVERSATION:", uses "USER:"/"GPT:" role markers, and
# "</s>" as the between-turn separator after the assistant's reply.
_KOALA_V1_TEMPLATE = (
    "BEGINNING OF CONVERSATION:"
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'user' -%}"
    " USER: {{ message['content'] }}"
    "{%- elif message['role'] == 'assistant' -%}"
    " GPT:{{ message['content'] }}</s>"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%} GPT:{%- endif -%}"
)


# Vicuna v1.1 (pre-v1.5) format, for completeness — same shape as koala
# but with a different system prompt. Handy for older Vicuna variants in
# LMSYS-Chat.
_VICUNA_V11_TEMPLATE = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'user' -%}"
    " USER: {{ message['content'] }}"
    "{%- elif message['role'] == 'assistant' -%}"
    " ASSISTANT: {{ message['content'] }}</s>"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%} ASSISTANT:{%- endif -%}"
)


# Map of HF model ids (or substrings thereof) → Jinja2 chat template.
# Substring match is case-insensitive; the FIRST matching entry wins.
_TEMPLATE_REGISTRY: dict[str, str] = {
    "oasst-sft-4-pythia": _OASST_PYTHIA_TEMPLATE,
    "oasst-sft-1-pythia": _OASST_PYTHIA_TEMPLATE,
    "koala-13b": _KOALA_V1_TEMPLATE,
    "koala-7b": _KOALA_V1_TEMPLATE,
    "vicuna-7b-v1.1": _VICUNA_V11_TEMPLATE,
    "vicuna-13b-v1.1": _VICUNA_V11_TEMPLATE,
    "vicuna-33b-v1.3": _VICUNA_V11_TEMPLATE,
}


def lookup_template(model_id: str) -> str | None:
    """Return a chat template for ``model_id``, or ``None`` if none registered."""
    needle = model_id.lower()
    for key, template in _TEMPLATE_REGISTRY.items():
        if key in needle:
            return template
    return None


def ensure_chat_template(tokenizer, model_id: str) -> bool:
    """If the tokenizer has no chat_template and we have one registered,
    apply it in place.

    Returns True if the tokenizer is now usable (either had one already or
    we installed one), False if still missing a template.
    """
    existing = getattr(tokenizer, "chat_template", None)
    if existing:
        return True
    template = lookup_template(model_id)
    if template is None:
        logger.info(
            "No chat_template registered for %s — tokenization/admin will "
            "fall back to concatenated encoding.",
            model_id,
        )
        return False
    tokenizer.chat_template = template
    logger.info("Installed fallback chat_template for %s from registry.", model_id)
    return True
