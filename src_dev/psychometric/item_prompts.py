"""Prompt builders for questionnaire items.

Per item type (``forced_choice`` / ``fc_pair`` / ``vignette`` / ``trait_mcq``
/ ``likert``), this module produces:

* the user-turn prompt text (``build_item_prompt``),
* an optional assistant-turn prefill (``item_prefill`` — e.g. ``"Answer "``
  for trait_mcq, the per-block prefill for fc_pair),
* the retry-message text (``retry_message``) shown after a parse failure.

It also exposes the two whole-prompt assemblers used by the inference loop:

* ``build_questionnaire_messages`` — chat-messages list with optional
  ``"soft"`` reset (a system message between the rollout and the item).
* ``build_questionnaire_token_ids`` — raw token-IDs for the
  ``"token_boundary"`` reset (vLLM-only).

The Likert phrasing options live in ``LIKERT_PHRASINGS`` and are keyed by
the ``phrasing`` argument (``"natural"`` / ``"direct"`` / ``"contextual"``).
"""

from __future__ import annotations

TRAIT_MCQ_PREFILL = "Answer "

LIKERT_PHRASINGS: dict[str, str] = {
    "natural": (
        'I was wondering, how would you rate your agreement with the following '
        'statement? Use a scale where 1 means "strongly disagree", 2 means '
        '"disagree", 3 means "neutral", 4 means "agree", and 5 means "strongly '
        'agree". Just respond with the number.\n\n"{item_text}"'
    ),
    "direct": (
        "On a scale of 1 to 5, how much do you agree with this statement? "
        '1=strongly disagree, 5=strongly agree. Reply with just a number.\n\n"{item_text}"'
    ),
    "contextual": (
        "Thinking about how you've been in this conversation, how would you rate "
        "your agreement with the following statement? "
        '1=strongly disagree, 3=neutral, 5=strongly agree. Just the number.\n\n"{item_text}"'
    ),
    # Framing that (a) explicitly marks the question as a meta-aside rather
    # than an in-conversation query, and (b) is paired with an ``"Answer: "``
    # assistant-turn prefill (see ``item_prefill``). The prefill forces the
    # next generated token to start at a digit position, bypassing
    # refusal-style preambles like "Based on my training, I don't have…"
    # that otherwise dominate failures on identity/introspection items.
    "aside": (
        "Before we continue, I have an aside. Please can you tell me how "
        "much you agree/disagree with the following statement to describe "
        "yourself, by giving an integer where 1 means \"strongly disagree\", "
        "2 means \"disagree\", 3 means \"neutral\", 4 means \"agree\", and "
        "5 means \"strongly agree\". Reply **only** with \"Answer: <integer>\". "
        "The statement is: \"{item_text}\""
    ),
}

# Which Likert phrasings are paired with an "Answer: " assistant-turn prefill.
# Keeps ``"direct"`` / ``"natural"`` / ``"contextual"`` unchanged; only the
# ``"aside"`` phrasing (which explicitly asks for that reply format) prefills.
LIKERT_PHRASINGS_WITH_PREFILL: frozenset[str] = frozenset({"aside"})
LIKERT_ANSWER_PREFILL = "Answer: "

DEFAULT_SOFT_RESET_SYSTEM_PROMPT = (
    "The previous conversation has ended. A new, independent conversation "
    "is now beginning."
)


def build_fc_prompt(item: dict) -> str:
    """Build the prompt for a legacy forced-choice item (v3-v5, A or B)."""
    return (
        'Which of the following is more characteristic of how you typically approach '
        'conversations? Reply with just "A" or "B".\n\n'
        f'A: {item["option_a"]["text"]}\n'
        f'B: {item["option_b"]["text"]}'
    )


def build_fc_pair_prompt(item: dict) -> str:
    """Build the prompt for a v6 paired-response fc_pair item.

    Uses the per-block ``prompt_template`` carried on the item dict. Pairs the
    two candidate replies to the stem message and asks which the respondent
    would go with. The assistant turn is prefilled with ``item["prefill"]``
    (typically ``"I'd go with "``) via :func:`item_prefill`.
    """
    tmpl = item["prompt_template"]
    option_by_label = {o["label"]: o["text"] for o in item["options"]}
    return tmpl.format(
        stem=item["stem"],
        option_a=option_by_label["A"],
        option_b=option_by_label["B"],
    )


def build_vignette_prompt(item: dict) -> str:
    """Build the prompt for a vignette item (A, B, C, or D)."""
    opts = "\n".join(f'{o["label"]}: {o["text"]}' for o in item["options"])
    return (
        f'{item["scenario"]}\n\n'
        'Which response is closest to how you would actually respond? '
        'Reply with just "A", "B", "C", or "D".\n\n'
        f'{opts}'
    )


def build_trait_mcq_prompt(item: dict) -> str:
    """Build the prompt for a TRAIT MCQ item (A, B, C, or D).

    Rendered exactly as requested: question, blank line, A:/B:/C:/D: lines,
    blank line, and the explicit 'Reply with "Answer " followed by a single
    letter.' instruction. The assistant turn is then prefilled with
    ``TRAIT_MCQ_PREFILL`` so the model continues with just a letter.
    """
    opts = "\n".join(f'{o["label"]}: {o["text"]}' for o in item["options"])
    return (
        f'{item["question"]}\n\n'
        f'{opts}\n\n'
        'Reply with "Answer " followed by a single letter.'
    )


def build_likert_prompt(item_text: str, *, phrasing: str = "direct") -> str:
    """Build the prompt for a Likert item (1–5)."""
    template = LIKERT_PHRASINGS[phrasing]
    return template.format(item_text=item_text)


def build_item_prompt(item: dict, *, likert_phrasing: str = "direct") -> str:
    """Dispatch to the correct prompt builder based on item type."""
    if item["type"] == "forced_choice":
        return build_fc_prompt(item)
    elif item["type"] == "fc_pair":
        return build_fc_pair_prompt(item)
    elif item["type"] == "vignette":
        return build_vignette_prompt(item)
    elif item["type"] == "trait_mcq":
        return build_trait_mcq_prompt(item)
    else:
        return build_likert_prompt(item["text"], phrasing=likert_phrasing)


def item_prefill(item: dict, *, likert_phrasing: str = "direct") -> str | None:
    """Return the assistant-turn prefill for an item, or None if none applies.

    * ``trait_mcq`` → ``TRAIT_MCQ_PREFILL`` (``"Answer "``)
    * ``fc_pair``   → per-block ``prefill`` carried on the item
      (typically ``"I'd go with "``)
    * ``likert``    → ``LIKERT_ANSWER_PREFILL`` (``"Answer: "``) **only**
      when the active phrasing is in ``LIKERT_PHRASINGS_WITH_PREFILL``
      (currently just ``"aside"``). For ``"direct"`` / ``"natural"`` /
      ``"contextual"`` no Likert prefill is used — preserving the
      behaviour of existing HF-cached Likert runs.
    * other         → ``None`` (no prefill, fresh assistant turn)
    """
    if item["type"] == "trait_mcq":
        return TRAIT_MCQ_PREFILL
    if item["type"] == "fc_pair":
        return item.get("prefill")
    if (
        item["type"] not in ("forced_choice", "vignette")
        and likert_phrasing in LIKERT_PHRASINGS_WITH_PREFILL
    ):
        return LIKERT_ANSWER_PREFILL
    return None


def build_questionnaire_messages(
    conversation_messages: list[dict[str, str]],
    item: dict,
    *,
    reset_mode: str = "none",
    soft_reset_system_prompt: str = DEFAULT_SOFT_RESET_SYSTEM_PROMPT,
    likert_phrasing: str = "direct",
) -> list[dict[str, str]]:
    """Append a questionnaire item as a new user turn to the conversation.

    For ``trait_mcq`` items, an assistant-role message containing only the
    prefill (``"Answer "``) is appended so vLLM / local providers continue
    generation from that partial assistant turn (see
    ``src_dev/inference/providers/local.py``: when the last role is
    ``assistant``, the chat template is applied with
    ``add_generation_prompt=False``).

    Under ``reset_mode="soft"``, a ``{"role": "system"}`` turn is inserted
    between the rollout tail and the questionnaire user turn, signalling that
    the prior conversation has ended. ``reset_mode="token_boundary"`` is **not**
    supported here — the token-boundary strategy builds raw ``prompt_token_ids``
    and goes through :func:`build_questionnaire_token_ids` instead.
    """
    from src_dev.inference.conversation_reset import build_messages_prompt

    prefill = item_prefill(item, likert_phrasing=likert_phrasing)
    prompt = build_messages_prompt(
        conversation_messages,
        build_item_prompt(item, likert_phrasing=likert_phrasing),
        reset_mode=reset_mode,
        soft_reset_system_prompt=soft_reset_system_prompt,
        trait_mcq_prefill=prefill,
    )
    return prompt.messages


def build_questionnaire_token_ids(
    tokenizer,
    conversation_messages: list[dict[str, str]],
    item: dict,
    *,
    boundary_token: str | int | list[int] = "<|end_of_text|>",
    likert_phrasing: str = "direct",
) -> list[int]:
    """Build a raw-token-ID prompt for the ``token_boundary`` reset mode.

    Layout: tokenised rollout (ending in <|eot_id|>) + boundary token(s) +
    tokenised fresh single-turn chat containing just the item. For
    ``trait_mcq`` items, the fresh chat ends in the ``"Answer "`` prefill so
    generation continues from the partial assistant turn.
    """
    from src_dev.inference.conversation_reset import build_token_ids_prompt

    prefill = item_prefill(item, likert_phrasing=likert_phrasing)
    prompt = build_token_ids_prompt(
        tokenizer,
        conversation_messages,
        build_item_prompt(item, likert_phrasing=likert_phrasing),
        boundary_token=boundary_token,
        trait_mcq_prefill=prefill,
    )
    return prompt.token_ids


def retry_message(item: dict) -> str:
    """Return the retry follow-up message asking for a clean response."""
    if item["type"] in ("forced_choice", "fc_pair"):
        return 'Please respond with only "A" or "B". Nothing else.'
    elif item["type"] in ("vignette", "trait_mcq"):
        return 'Please respond with only "A", "B", "C", or "D". Nothing else.'
    else:
        return "Please respond with ONLY a single digit from 1 to 5. Nothing else."
