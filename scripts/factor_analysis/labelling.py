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
import json
import logging
from pathlib import Path
from typing import Callable, Literal

logger = logging.getLogger(__name__)

# DEFAULT_MODEL = "gpt-5-mini-2025-08-07"
# DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_PROVIDER = "anthropic"


# You are an expert in behavioral analysis and psychometrics. \
_SYSTEM_PROMPT = """\
You are an expert in textual analysis. \
You will be shown pieces of text that score at the extreme ends of a latent \
dimension discovered via factor analysis of text embeddings. \
Your task is to identify what the factor represents.\
You **must** think deeply, and consider all the examples provided, not just \
the first ones you see.
"""
# Most factors show high variance along the axis of whether or not the text \
# engages substantively versus deflects or is content-free.\
# Please **ignore** this axis and focus on object level distinctions \
# in the actual content between high- and low-scoring texts, if possible \
# (if one of the extremes has no texts with meaningful content, please \
# report that the factor 'gives no meaningful information other than refusals').

_USER_TEMPLATE = """\
Below are some pieces of text that score at the HIGH and LOW ends of factor {fi}.

```json
{json_block}
```

Based on these examples, carefully consider what this factor represents. \
Think hard about what distinguishes high-scoring from low-scoring texts. \
Then, respond with the following format:

<~10 word summary of what this factor differentiates between>\n
<2-3 sentence more detailed description of the factor>

"""

_USER_TEMPLATE_CONTRASTIVE_JSONL = """\
Below is a JSONL block of contrastive HIGH/LOW pairs of text excerpts for factor {fi}.
Each line is one pair from the same ranking position and has keys:
- "high": a high-scoring text excerpt
- "low": a low-scoring response

```jsonl
{jsonl_block}
```

Based on these examples, carefully consider what this factor represents. \
Think hard about what distinguishes high-scoring from low-scoring texts. \
Then, respond with the following format:

<~10 word summary of what this factor differentiates between>\n
<2-3 sentence more detailed description of the factor>

"""

_FAILURE_PREFIX = "(labelling failed:"

MAX_SPREAD_STRATEGIES = {
    "label_prompt_pair": {"include_prompt": True, "include_scores": False},
    "label_prompt_pair_score": {"include_prompt": True, "include_scores": True},
    "label_pair_score": {"include_prompt": False, "include_scores": True},
    "label_pair": {"include_prompt": False, "include_scores": False},
}
MaxSpreadLabelStrategy = Literal[
    "label_prompt_pair",
    "label_prompt_pair_score",
    "label_pair_score",
    "label_pair",
]


def _build_contrastive_jsonl_block(high_responses: list[str], low_responses: list[str]) -> str:
    n_pairs = max(len(high_responses), len(low_responses))
    lines = []
    for i in range(n_pairs):
        pair = {
            "pair_index": i,
            "high": high_responses[i] if i < len(high_responses) else "",
            "low": low_responses[i] if i < len(low_responses) else "",
        }
        lines.append(json.dumps(pair, ensure_ascii=False))
    return "\n".join(lines)


def _collect_responses(entries: list[dict], n: int, excerpt_chars: int, max_per_prompt: int = 1) -> list[str]:
    prompt_counts: dict[str, int] = {}
    responses = []
    for entry in entries:
        if len(responses) >= n:
            break
        group = str(entry.get("input_group_id", entry.get("seed_user_message", "")))
        if prompt_counts.get(group, 0) >= max_per_prompt:
            continue
        prompt_counts[group] = prompt_counts.get(group, 0) + 1
        response = str(entry.get("text_excerpt", "")).strip()[:excerpt_chars]
        responses.append(response)
    return responses


def _build_messages(
    factor_data: dict,
    top_n: int,
    excerpt_chars: int,
    max_per_prompt: int = 1,
    prompt_format: Literal["grouped_json", "contrastive_jsonl"] = "grouped_json",
) -> list[dict]:
    fi = factor_data["factor_index"]
    high_responses = _collect_responses(factor_data["top"], top_n, excerpt_chars, max_per_prompt)
    low_responses = _collect_responses(factor_data["bottom"], top_n, excerpt_chars, max_per_prompt)
    if prompt_format == "contrastive_jsonl":
        jsonl_block = _build_contrastive_jsonl_block(high_responses, low_responses)
        user_content = _USER_TEMPLATE_CONTRASTIVE_JSONL.format(fi=fi, jsonl_block=jsonl_block)
    else:
        json_block = json.dumps({"high": high_responses, "low": low_responses}, ensure_ascii=False, indent=2)
        user_content = _USER_TEMPLATE.format(fi=fi, json_block=json_block)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _format_factor_score(value: float | int | str | None) -> str:
    try:
        return f"{float(value):+.4f}"
    except (TypeError, ValueError):
        return str(value or "")


def _build_max_spread_jsonl_block(
    factor_data: dict,
    top_n: int,
    excerpt_chars: int,
    *,
    include_prompt: bool,
    include_scores: bool,
) -> str:
    lines = []
    for pair_index, group in enumerate(factor_data.get("groups", [])[:top_n]):
        high = group.get("high", {})
        low = group.get("low", {})
        pair: dict[str, object] = {"pair_index": pair_index}
        if include_prompt:
            pair["prompt"] = str(high.get("seed_user_message", "")).strip()
        if include_scores:
            pair["high_score"] = _format_factor_score(high.get("target_factor_score"))
            pair["low_score"] = _format_factor_score(low.get("target_factor_score"))
        pair["high"] = str(high.get("text_excerpt", "")).strip()[:excerpt_chars]
        pair["low"] = str(low.get("text_excerpt", "")).strip()[:excerpt_chars]
        lines.append(json.dumps(pair, ensure_ascii=False))
    return "\n".join(lines)


def _build_max_spread_messages(
    factor_data: dict,
    top_n: int,
    excerpt_chars: int,
    *,
    include_prompt: bool,
    include_scores: bool,
) -> list[dict]:
    fi = factor_data["factor_index"]
    key_lines = [
        '- "high": the higher-scoring response in the pair',
        '- "low": the lower-scoring response in the pair',
    ]
    if include_prompt:
        key_lines.insert(0, '- "prompt": the original user prompt shared by the pair')
    if include_scores:
        key_lines.extend([
            '- "high_score": signed target-factor score for the high response',
            '- "low_score": signed target-factor score for the low response',
        ])
    user_content = (
        f"Below is a JSONL block of contrastive HIGH/LOW response pairs for factor {fi}.\n"
        "Each line is one pair from the same max-spread prompt group and has keys:\n"
        f"{chr(10).join(key_lines)}\n\n"
        "```jsonl\n"
        f"{_build_max_spread_jsonl_block(factor_data, top_n, excerpt_chars, include_prompt=include_prompt, include_scores=include_scores)}\n"
        "```\n\n"
        "Based on these examples, carefully consider what this factor represents. "
        "Think hard about what distinguishes high-scoring from low-scoring texts. "
        "Then, respond with the following format:\n\n"
        "<~10 word summary of what this factor differentiates between>\n\n"
        "<2-3 sentence more detailed description of the factor>\n"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def label_is_complete(label: str | None) -> bool:
    """Return True if a cached label should be treated as complete."""
    if not isinstance(label, str):
        return False
    stripped = label.strip()
    if not stripped:
        return False
    return not stripped.startswith(_FAILURE_PREFIX)


def load_label_checkpoint(checkpoint_path: str | Path, total: int) -> list[str]:
    """Load a checkpoint file and normalize it to the expected label count."""
    path = Path(checkpoint_path)
    if not path.exists():
        return [""] * total
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load label checkpoint %s: %s", path, exc)
        return [""] * total

    if not isinstance(raw, list):
        logger.warning("Ignoring malformed label checkpoint %s: expected list.", path)
        return [""] * total

    labels = [str(item) if isinstance(item, str) else "" for item in raw[:total]]
    if len(labels) < total:
        labels.extend([""] * (total - len(labels)))
    return labels


def save_label_checkpoint(labels: list[str], checkpoint_path: str | Path) -> None:
    """Atomically save label checkpoint state."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


async def _label_one_async(
    factor_data: dict,
    provider,
    build_messages: Callable[[dict], list[dict]],
    semaphore: asyncio.Semaphore,
) -> str:
    messages = build_messages(factor_data)
    async with semaphore:
        try:
            return await provider.generate_async(messages, max_tokens=256, temperature=0.3)
        except Exception as e:
            fi = factor_data["factor_index"]
            logger.warning(f"Factor {fi} labelling failed: {e}")
            return f"(labelling failed: {e})"


def _run_labelling(
    factor_data_list: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    max_concurrent: int = 10,
    show_progress: bool = True,
    checkpoint_path: str | Path | None = None,
    build_messages: Callable[[dict], list[dict]],
    description: str,
) -> list[str]:
    """Run LLM labelling over aligned factor records with resumable checkpoints."""
    from scripts.inference import InferenceConfig, get_provider

    config = InferenceConfig(model=model, provider=provider)
    provider = get_provider(provider, config)

    async def _run_all() -> list[str]:
        semaphore = asyncio.Semaphore(max_concurrent)
        total = len(factor_data_list)
        results = (
            load_label_checkpoint(checkpoint_path, total)
            if checkpoint_path is not None
            else [""] * total
        )
        completed = sum(1 for label in results if label_is_complete(label))

        if checkpoint_path is not None and completed:
            print(
                f"Resuming label checkpoint {checkpoint_path}: "
                f"{completed}/{total} already complete"
            )

        async def _run_one(idx: int, factor_data: dict) -> tuple[int, str]:
            label = await _label_one_async(
                factor_data, provider, build_messages, semaphore
            )
            return idx, label

        tasks = [
            asyncio.create_task(_run_one(idx, fd))
            for idx, fd in enumerate(factor_data_list)
            if not label_is_complete(results[idx])
        ]

        if not tasks:
            if show_progress and total > 0:
                print(f"  progress: {completed}/{total}")
            return results

        for task in asyncio.as_completed(tasks):
            idx, label = await task
            results[idx] = label
            if checkpoint_path is not None:
                save_label_checkpoint(results, checkpoint_path)
            completed += 1
            if show_progress:
                print(f"  progress: {completed}/{total}", end="\r", flush=True)
        if show_progress and total > 0:
            print(" " * 32, end="\r", flush=True)
            print(f"  progress: {total}/{total}")

        return results

    print(
        f"Labelling {len(factor_data_list)} factors with {model} "
        f"({description})..."
    )
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


def label_factors(
    extremes: list[dict],
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    top_n: int = 10,
    excerpt_chars: int = 400,
    max_per_prompt: int = 10,
    prompt_format: Literal["grouped_json", "contrastive_jsonl"] = "grouped_json",
    max_concurrent: int = 10,
    show_progress: bool = True,
    checkpoint_path: str | Path | None = None,
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
        prompt_format: Prompt payload format for examples.
                       "grouped_json" (default) uses {"high": [...], "low": [...]};
                       "contrastive_jsonl" uses JSONL of ranked high/low pairs.
        max_concurrent: Max simultaneous API requests.
        show_progress: If True, print completion progress while labelling.
        checkpoint_path: Optional path to a JSON checkpoint file. Completed
                         labels are written incrementally and reused on resume.

    Returns:
        List of label strings, one per factor, in the same order as extremes.
    """
    return _run_labelling(
        extremes,
        model=model,
        provider=provider,
        max_concurrent=max_concurrent,
        show_progress=show_progress,
        checkpoint_path=checkpoint_path,
        build_messages=lambda factor_data: _build_messages(
            factor_data,
            top_n=top_n,
            excerpt_chars=excerpt_chars,
            max_per_prompt=max_per_prompt,
            prompt_format=prompt_format,
        ),
        description=(
            f"top_n={top_n}, max_per_prompt={max_per_prompt}, "
            f"prompt_format={prompt_format}"
        ),
    )


def label_max_spread_factors(
    max_spread_factors: list[dict],
    *,
    strategy: MaxSpreadLabelStrategy,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    top_n: int = 10,
    excerpt_chars: int = 400,
    max_concurrent: int = 10,
    show_progress: bool = True,
    checkpoint_path: str | Path | None = None,
) -> list[str]:
    """Label max-spread factor pairs using a structured pair prompt.

    Args:
        max_spread_factors: List of dicts with 'factor_index' and 'groups', where
            each group contains 'high' and 'low' response records from the same prompt.
        strategy: Which prompt variant to use.
        model: LLM model name to use for labelling.
        provider: LLM provider name to use for labelling.
        top_n: Number of max-spread groups to include per factor.
        excerpt_chars: Max characters of each response excerpt to include.
        max_concurrent: Max simultaneous API requests.
        show_progress: If True, print completion progress while labelling.
        checkpoint_path: Optional path to a JSON checkpoint file. Completed
            labels are written incrementally and reused on resume.
    """
    if strategy not in MAX_SPREAD_STRATEGIES:
        raise ValueError(
            f"Unknown max-spread labelling strategy {strategy!r}; "
            f"expected one of {sorted(MAX_SPREAD_STRATEGIES)}"
        )
    strategy_cfg = MAX_SPREAD_STRATEGIES[strategy]
    return _run_labelling(
        max_spread_factors,
        model=model,
        provider=provider,
        max_concurrent=max_concurrent,
        show_progress=show_progress,
        checkpoint_path=checkpoint_path,
        build_messages=lambda factor_data: _build_max_spread_messages(
            factor_data,
            top_n=top_n,
            excerpt_chars=excerpt_chars,
            include_prompt=strategy_cfg["include_prompt"],
            include_scores=strategy_cfg["include_scores"],
        ),
        description=(
            f"max_spread_strategy={strategy}, top_n={top_n}, "
            f"include_prompt={strategy_cfg['include_prompt']}, "
            f"include_scores={strategy_cfg['include_scores']}"
        ),
    )
