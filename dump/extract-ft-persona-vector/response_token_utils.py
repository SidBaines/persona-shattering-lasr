"""
Utilities for extracting assistant response token indices.
Based on assistant-axis implementation for compatibility.
"""

from typing import List, Tuple


def strip_trailing_special_tokens(token_ids: List[int], special_ids: set) -> List[int]:
    """
    Strip trailing special tokens from a sequence.

    Args:
        token_ids: List of token IDs
        special_ids: Set of special token IDs to strip

    Returns:
        Token IDs with trailing special tokens removed
    """
    end = len(token_ids)
    while end > 0 and token_ids[end - 1] in special_ids:
        end -= 1
    return token_ids[:end]


def longest_common_prefix_len(a: List[int], b: List[int]) -> int:
    """Find the length of the longest common prefix between two sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def find_subsequence(haystack: List[int], needle: List[int]) -> int:
    """Find the starting index of needle in haystack, or -1 if not found."""
    if not needle or len(needle) > len(haystack):
        return -1

    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


def get_assistant_response_token_ids(
    tokenizer,
    conversation: List[dict],
    **chat_kwargs
) -> Tuple[List[int], int, int]:
    """
    Get token IDs for assistant response content only (excluding special tokens).

    Uses the assistant-axis approach:
    1. Tokenize conversation with and without the assistant message
    2. Find the delta (new tokens from assistant message)
    3. Strip trailing special tokens
    4. Locate actual content within delta

    Args:
        tokenizer: HuggingFace tokenizer
        conversation: List of messages [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        **chat_kwargs: Additional arguments for apply_chat_template

    Returns:
        Tuple of (full_token_ids, response_start_idx, response_end_idx)
        - full_token_ids: All token IDs for the conversation
        - response_start_idx: Start index of assistant response content
        - response_end_idx: End index of assistant response content (excluding special tokens)
    """
    # Get the assistant message (last message)
    if not conversation or conversation[-1]["role"] != "assistant":
        raise ValueError("Conversation must end with an assistant message")

    assistant_content = conversation[-1]["content"]
    messages_before = conversation[:-1]

    # Tokenize full conversation
    full_ids = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=False,
        **chat_kwargs
    )

    # Handle case where tokenizer returns BatchEncoding
    if hasattr(full_ids, 'input_ids'):
        full_ids = full_ids.input_ids

    # Tokenize conversation with empty assistant message
    msgs_empty = messages_before + [{"role": "assistant", "content": ""}]
    ids_empty = tokenizer.apply_chat_template(
        msgs_empty,
        tokenize=True,
        add_generation_prompt=False,
        **chat_kwargs
    )
    if hasattr(ids_empty, 'input_ids'):
        ids_empty = ids_empty.input_ids

    # Find the suffix introduced by the assistant message
    pref = longest_common_prefix_len(full_ids, ids_empty)
    delta = full_ids[pref:]

    # Strip trailing special tokens (like <end_of_turn>, \n, <eos>)
    special_ids = set(tokenizer.all_special_ids) if hasattr(tokenizer, 'all_special_ids') else set()
    delta = strip_trailing_special_tokens(delta, special_ids)

    # Try to locate the raw content within delta
    # Try both with and without leading space
    plain = tokenizer(assistant_content, add_special_tokens=False)['input_ids']
    sp = tokenizer(" " + assistant_content, add_special_tokens=False)['input_ids']

    start_in_delta = find_subsequence(delta, plain)
    use = plain

    if start_in_delta == -1:
        start_in_delta = find_subsequence(delta, sp)
        use = sp if start_in_delta != -1 else plain

    if start_in_delta == -1:
        # Fallback: use the whole delta
        content_ids = delta
        start_in_delta = 0
    else:
        content_ids = delta[start_in_delta:start_in_delta + len(use)]

    # Calculate absolute positions in full_ids
    response_start_idx = pref + start_in_delta
    response_end_idx = pref + start_in_delta + len(content_ids)

    return full_ids, response_start_idx, response_end_idx
