"""Prompt builders for questionnaire items.

Per item type (``forced_choice`` / ``fc_pair`` / ``vignette`` / ``trait_mcq``
/ ``likert``), this module produces:

* the user-turn prompt text (``build_item_prompt``),
* an optional assistant-turn prefill (``item_prefill`` — e.g. ``"Answer"``
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

# BPE boundary note: this prefill deliberately omits the trailing space.
# Tokenizers like Qwen2.5's and Llama-3.1's merge space+letter into a single
# token (e.g. " A" is one token, not [" ", "A"]). A prefill ending with a
# standalone space token makes the natural continuation " A" produce decoded
# text "Answer  A" (double space) — out-of-distribution for the model. Qwen2.5
# in particular disperses ~12% of top-20 mass onto escape routes like " Answer"
# and <|im_end|> under the buggy "Answer " prefill; with "Answer" the
# letter-mass share recovers to ~100%. Do NOT re-add the trailing space.
# See scripts_dev/psychometric_assessment/prefill_ablation.py for the evidence.
# (fc_pair prefills in the questionnaire JSONs — e.g. "I'd go with " — have
# the same issue but are out of scope of this constant; see item_prefill.)
TRAIT_MCQ_PREFILL = "Answer"

# Optional topic-switch prefix for trait_mcq user-turn text. When enabled
# (via the ``topic_switch_prefix`` kwarg on the builders), the rendered
# user turn begins with this string so the model sees an explicit signal
# that the question is independent of any preceding rollout context.
# This is belt-and-suspenders: the chat template already separates the
# rollout turns from the questionnaire turn, but on 15-turn coherent B
# rollouts an explicit marker is cheap insurance that "How should I spend
# my time at the country fair" is not being read as a continuation of
# whatever scenario came before. The prefix is plain text, not a
# template — tokenization is unremarkable on Qwen/Llama BPE and does not
# interact with the TRAIT_MCQ_PREFILL BPE-boundary fix above.
TRAIT_MCQ_TOPIC_SWITCH_PREFIX = "New question: "

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


def build_trait_mcq_prompt(item: dict, *, topic_switch_prefix: bool = False) -> str:
    """Build the prompt for a TRAIT MCQ item (A, B, C, or D).

    Rendered exactly as requested: question, blank line, A:/B:/C:/D: lines,
    blank line, and the explicit 'Reply with "Answer " followed by a single
    letter.' instruction. The assistant turn is then prefilled with
    ``TRAIT_MCQ_PREFILL`` (``"Answer"`` — no trailing space, see the constant's
    comment) so the model continues with a space-prefixed letter token that
    decodes cleanly to ``"Answer A"`` / ``"Answer B"`` / etc.

    When ``topic_switch_prefix=True``, ``TRAIT_MCQ_TOPIC_SWITCH_PREFIX``
    is prepended to the question so the model sees an explicit topic-switch
    marker. See the constant's comment above for rationale. Default is
    ``False`` to preserve the rendering of existing HF-cached trait_mcq
    runs under their current run-ids.
    """
    opts = "\n".join(f'{o["label"]}: {o["text"]}' for o in item["options"])
    prefix = TRAIT_MCQ_TOPIC_SWITCH_PREFIX if topic_switch_prefix else ""
    return (
        f'{prefix}{item["question"]}\n\n'
        f'{opts}\n\n'
        'Reply with "Answer " followed by a single letter.'
    )


def build_likert_prompt(item_text: str, *, phrasing: str = "direct") -> str:
    """Build the prompt for a Likert item (1–5)."""
    template = LIKERT_PHRASINGS[phrasing]
    return template.format(item_text=item_text)


def build_item_prompt(
    item: dict,
    *,
    likert_phrasing: str = "direct",
    trait_mcq_topic_switch_prefix: bool = False,
) -> str:
    """Dispatch to the correct prompt builder based on item type.

    ``trait_mcq_topic_switch_prefix`` is forwarded only to
    :func:`build_trait_mcq_prompt`; it is a no-op for other item types.
    """
    if item["type"] == "forced_choice":
        return build_fc_prompt(item)
    elif item["type"] == "fc_pair":
        return build_fc_pair_prompt(item)
    elif item["type"] == "vignette":
        return build_vignette_prompt(item)
    elif item["type"] == "trait_mcq":
        return build_trait_mcq_prompt(
            item, topic_switch_prefix=trait_mcq_topic_switch_prefix
        )
    else:
        return build_likert_prompt(item["text"], phrasing=likert_phrasing)


def item_prefill(item: dict, *, likert_phrasing: str = "direct") -> str | None:
    """Return the assistant-turn prefill for an item, or None if none applies.

    * ``trait_mcq`` → ``TRAIT_MCQ_PREFILL`` (``"Answer"``)
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
    trait_mcq_topic_switch_prefix: bool = False,
) -> list[dict[str, str]]:
    """Append a questionnaire item as a new user turn to the conversation.

    For ``trait_mcq`` items, an assistant-role message containing only the
    prefill (``"Answer"``) is appended so vLLM / local providers continue
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
        build_item_prompt(
            item,
            likert_phrasing=likert_phrasing,
            trait_mcq_topic_switch_prefix=trait_mcq_topic_switch_prefix,
        ),
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
    trait_mcq_topic_switch_prefix: bool = False,
) -> list[int]:
    """Build a raw-token-ID prompt for the ``token_boundary`` reset mode.

    Layout: tokenised rollout (ending in <|eot_id|>) + boundary token(s) +
    tokenised fresh single-turn chat containing just the item. For
    ``trait_mcq`` items, the fresh chat ends in the ``"Answer"`` prefill so
    generation continues from the partial assistant turn.
    """
    from src_dev.inference.conversation_reset import build_token_ids_prompt

    prefill = item_prefill(item, likert_phrasing=likert_phrasing)
    prompt = build_token_ids_prompt(
        tokenizer,
        conversation_messages,
        build_item_prompt(
            item,
            likert_phrasing=likert_phrasing,
            trait_mcq_topic_switch_prefix=trait_mcq_topic_switch_prefix,
        ),
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
