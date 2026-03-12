"""LLM-based factor labelling.

Sends high/low extreme response examples to an LLM and asks it to describe
what behavioral dimension each factor represents.

Example:
    extremes = factor_extremes(scores, metadata, top_n=10)
    labels = label_factors(extremes)
    for i, label in enumerate(labels):
        print(f"Factor {i}: {label}")
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
# DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_PROVIDER = "anthropic"


# You are an expert in behavioral analysis and psychometrics. \
_SYSTEM_PROMPT = """\
You are an expert in textual analysis. \
You will be shown text responses that score at the extreme ends of a latent \
dimension discovered via factor analysis of LLM response embeddings. \
Your task is to identify what the factor represents.\
Most factors show high variance along the axis of whether or not the model \
engages helpfully and substantively versus deflects or refuses.\
Please **ignore** this axis and focus on object level distinctions \
in the actual content between high- and low-scoring responses, if possible \
(if one of the extremes has no responses with meaningful content, please \
report that the factor 'gives no meaningful information other than refusals').
"""

_USER_TEMPLATE = """\
Below are responses from an LLM that score at the HIGH and LOW ends of factor {fi}.
Each example shows the prompt the LLM was given, followed by its response.

=== HIGH (scores high on this factor) ===
{high_block}

=== LOW (scores low on this factor) ===
{low_block}

Based on these examples, write a short (1–2 sentence) description of what this factor \
represents. Focus on what concretely distinguishes high-scoring from low-scoring responses.\
"""


def _format_block(entries: list[dict], n: int, excerpt_chars: int, max_per_prompt: int = 1) -> str:
    prompt_counts: dict[str, int] = {}
    lines = []
    for entry in entries:
        if len(lines) >= n:
            break
        group = str(entry.get("input_group_id", entry.get("seed_user_message", "")))
        if prompt_counts.get(group, 0) >= max_per_prompt:
            continue
        prompt_counts[group] = prompt_counts.get(group, 0) + 1
        prompt = str(entry.get("seed_user_message", "")).strip()[:150]
        response = str(entry.get("text_excerpt", "")).strip()[:excerpt_chars]
        lines.append(f"[{len(lines) + 1}] Prompt: {prompt}\n    Response: {response}")
    return "\n\n".join(lines)


def _build_messages(factor_data: dict, top_n: int, excerpt_chars: int, max_per_prompt: int = 1) -> list[dict]:
    fi = factor_data["factor_index"]
    high_block = _format_block(factor_data["top"], top_n, excerpt_chars, max_per_prompt)
    low_block = _format_block(factor_data["bottom"], top_n, excerpt_chars, max_per_prompt)
    user_content = _USER_TEMPLATE.format(fi=fi, high_block=high_block, low_block=low_block)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


async def _label_one_async(
    factor_data: dict,
    provider,
    top_n: int,
    excerpt_chars: int,
    max_per_prompt: int,
    semaphore: asyncio.Semaphore,
) -> str:
    messages = _build_messages(factor_data, top_n, excerpt_chars, max_per_prompt)
    async with semaphore:
        try:
            return await provider.generate_async(messages, max_tokens=256, temperature=0.3)
        except Exception as e:
            fi = factor_data["factor_index"]
            logger.warning(f"Factor {fi} labelling failed: {e}")
            return f"(labelling failed: {e})"


def label_factors(
    extremes: list[dict],
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    top_n: int = 10,
    excerpt_chars: int = 400,
    max_per_prompt: int = 1,
    max_concurrent: int = 10,
) -> list[str]:
    """Label each factor with an LLM-generated description.

    Calls the LLM in parallel (up to max_concurrent requests at a time).

    Args:
        extremes: Output of factor_extremes() — list of dicts with
                  'factor_index', 'top', 'bottom'.
        model: LLM model name to use for labelling.
        provider: LLM provider name to use for labelling.
        top_n: Number of high/low examples to include per factor.
        excerpt_chars: Max characters of each response excerpt to include.
        max_per_prompt: Max responses from the same prompt group to include
                        in each high/low block. Entries are taken in score
                        order; duplicates are skipped until the cap is hit.
        max_concurrent: Max simultaneous API requests.

    Returns:
        List of label strings, one per factor, in the same order as extremes.
    """
    from scripts.inference import InferenceConfig, get_provider

    config = InferenceConfig(model=model, provider=provider)
    provider = get_provider(provider, config)

    async def _run_all() -> list[str]:
        semaphore = asyncio.Semaphore(max_concurrent)
        tasks = [
            _label_one_async(fd, provider, top_n, excerpt_chars, max_per_prompt, semaphore)
            for fd in extremes
        ]
        results = await asyncio.gather(*tasks)
        return list(results)

    print(f"Labelling {len(extremes)} factors with {model} (top_n={top_n}, max_per_prompt={max_per_prompt})...")
    try:
        asyncio.get_running_loop()
        # Already inside a running loop (e.g. Jupyter) — run in a fresh thread.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            labels = pool.submit(asyncio.run, _run_all()).result()
    except RuntimeError:
        labels = asyncio.run(_run_all())
    print("Done.")
    return labels
