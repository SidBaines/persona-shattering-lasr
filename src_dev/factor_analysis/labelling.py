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
from datetime import datetime, timezone
import json
import logging
import re
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

_JOINT_SYSTEM_PROMPT = """\
You are an expert in textual analysis and latent-dimension interpretation. \
You will be shown several latent factors at once, each with examples from the \
high and low ends of the factor. \
Your task is to label all of them jointly. \
Choose labels and descriptions that are as distinct from one another as the \
evidence allows. Avoid reusing near-synonymous vocabulary across factors unless \
the examples clearly force overlap. \
Return strict JSON only, with no prose before or after the JSON.
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

_JOINT_USER_TEMPLATE = """\
Below is a JSON payload describing {n_factors} latent factors. Each factor has:
- "factor_index": integer identifier
- "high": examples from the high-scoring end
- "low": examples from the low-scoring end

```json
{json_block}
```

Label all factors jointly. Focus on what distinguishes the HIGH end from the LOW end for each factor.

Rules:
- Make each summary about 10 words or fewer.
- Make the summaries maximally distinct in wording and emphasis.
- If two factors seem related, explain the specific distinction that keeps them separate.
- Include every factor exactly once.
- Return strict JSON in this exact schema:

{{
  "factors": [
    {{
      "factor_index": 0,
      "summary": "short distinct label",
      "description": "2-3 sentence explanation"
    }}
  ]
}}
"""

_JOINT_MAX_SPREAD_USER_TEMPLATE = """\
Below is a JSON payload describing {n_factors} latent factors using contrastive prompt-matched pairs. Each factor has:
- "factor_index": integer identifier
- "pairs": list of prompt-matched HIGH/LOW response pairs

```json
{json_block}
```

Label all factors jointly. Focus on what distinguishes the HIGH responses from the LOW responses for each factor.

Rules:
- Make each summary about 10 words or fewer.
- Make the summaries maximally distinct in wording and emphasis.
- Use prompt text and score metadata only as supporting context; the label should reflect the behavioral/textual contrast.
- If two factors seem related, explain the specific distinction that keeps them separate.
- Include every factor exactly once.
- Return strict JSON in this exact schema:

{{
  "factors": [
    {{
      "factor_index": 0,
      "summary": "short distinct label",
      "description": "2-3 sentence explanation"
    }}
  ]
}}
"""

_FAILURE_PREFIX = "(labelling failed:"
DEFAULT_PER_FACTOR_MAX_TOKENS = 2048
DEFAULT_JOINT_MAX_TOKENS = 12000
DEFAULT_JOINT_MAX_FACTORS_PER_CALL: int | None = 6

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


def _format_label_text(summary: str, description: str) -> str:
    summary_text = summary.strip()
    description_text = description.strip()
    if summary_text and description_text:
        return f"{summary_text}\n\n{description_text}"
    return summary_text or description_text


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


def _build_joint_factor_payload(
    factor_data_list: list[dict],
    *,
    top_n: int,
    excerpt_chars: int,
    max_per_prompt: int,
) -> list[dict]:
    payload = []
    for factor_data in factor_data_list:
        payload.append(
            {
                "factor_index": factor_data["factor_index"],
                "high": _collect_responses(
                    factor_data["top"],
                    top_n,
                    excerpt_chars,
                    max_per_prompt,
                ),
                "low": _collect_responses(
                    factor_data["bottom"],
                    top_n,
                    excerpt_chars,
                    max_per_prompt,
                ),
            }
        )
    return payload


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


def _build_joint_max_spread_payload(
    max_spread_factors: list[dict],
    *,
    top_n: int,
    excerpt_chars: int,
    include_prompt: bool,
    include_scores: bool,
) -> list[dict]:
    payload = []
    for factor_data in max_spread_factors:
        pairs: list[dict[str, object]] = []
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
            pairs.append(pair)
        payload.append({"factor_index": factor_data["factor_index"], "pairs": pairs})
    return payload


def _build_joint_messages(
    factor_data_list: list[dict],
    *,
    top_n: int,
    excerpt_chars: int,
    max_per_prompt: int,
) -> list[dict]:
    payload = _build_joint_factor_payload(
        factor_data_list,
        top_n=top_n,
        excerpt_chars=excerpt_chars,
        max_per_prompt=max_per_prompt,
    )
    user_content = _JOINT_USER_TEMPLATE.format(
        n_factors=len(payload),
        json_block=json.dumps(payload, ensure_ascii=False, indent=2),
    )
    return [
        {"role": "system", "content": _JOINT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_joint_max_spread_messages(
    max_spread_factors: list[dict],
    *,
    top_n: int,
    excerpt_chars: int,
    include_prompt: bool,
    include_scores: bool,
) -> list[dict]:
    payload = _build_joint_max_spread_payload(
        max_spread_factors,
        top_n=top_n,
        excerpt_chars=excerpt_chars,
        include_prompt=include_prompt,
        include_scores=include_scores,
    )
    user_content = _JOINT_MAX_SPREAD_USER_TEMPLATE.format(
        n_factors=len(payload),
        json_block=json.dumps(payload, ensure_ascii=False, indent=2),
    )
    return [
        {"role": "system", "content": _JOINT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _extract_json_candidate(text: str) -> str | None:
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    candidates = [block.strip() for block in fenced_blocks if block.strip()]
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

        for opener, closer in (("{", "}"), ("[", "]")):
            start = candidate.find(opener)
            end = candidate.rfind(closer)
            if start == -1 or end == -1 or end <= start:
                continue
            snippet = candidate[start : end + 1]
            try:
                json.loads(snippet)
                return snippet
            except json.JSONDecodeError:
                continue
    return None


def _repair_json_candidate(candidate: str) -> str | None:
    """Try to repair a near-valid JSON string with missing structural closers.

    This is intentionally conservative and targets a common failure mode from
    model outputs: one or more missing closing braces/brackets near the end of
    the response or immediately before a closing array/object delimiter.
    """
    out: list[str] = []
    stack: list[str] = []
    in_string = False
    escape = False

    for ch in candidate:
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            out.append(ch)
            in_string = True
            continue
        if ch in "{[":
            out.append(ch)
            stack.append(ch)
            continue
        if ch == "}":
            if stack and stack[-1] == "[":
                out.append("]")
                stack.pop()
                continue
            out.append(ch)
            while stack and stack[-1] != "{":
                out.insert(len(out) - 1, "]")
                stack.pop()
            if stack and stack[-1] == "{":
                stack.pop()
            continue
        if ch == "]":
            if stack and stack[-1] == "{":
                out.append("}")
                stack.pop()
                continue
            out.append(ch)
            while stack and stack[-1] != "[":
                out.insert(len(out) - 1, "}")
                stack.pop()
            if stack and stack[-1] == "[":
                stack.pop()
            continue
        out.append(ch)

    while stack:
        opener = stack.pop()
        out.append("}" if opener == "{" else "]")

    repaired = "".join(out).strip()
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        return None


def _parse_joint_label_response(
    response_text: str,
    factor_indices: list[int],
) -> list[str]:
    factor_index_to_pos = {factor_idx: pos for pos, factor_idx in enumerate(factor_indices)}
    labels = [""] * len(factor_indices)

    json_candidate = _extract_json_candidate(response_text)
    if json_candidate is None:
        repaired = _repair_json_candidate(response_text.strip())
        if repaired is None:
            raise ValueError("No valid JSON object found in joint labelling response.")
        logger.warning("Recovered malformed joint labelling JSON via structural repair.")
        json_candidate = repaired
    parsed = json.loads(json_candidate)
    if isinstance(parsed, dict):
        factors = parsed.get("factors")
    else:
        factors = parsed
    if not isinstance(factors, list):
        raise ValueError("Joint labelling response JSON must contain a list of factors.")

    for item in factors:
        if not isinstance(item, dict):
            continue
        factor_idx = item.get("factor_index")
        if factor_idx is None:
            continue
        try:
            factor_idx_int = int(factor_idx)
        except (TypeError, ValueError):
            continue
        if factor_idx_int not in factor_index_to_pos:
            continue
        summary = str(item.get("summary", item.get("label", "")))
        description = str(item.get("description", item.get("details", "")))
        labels[factor_index_to_pos[factor_idx_int]] = _format_label_text(summary, description)

    missing = [
        factor_idx
        for factor_idx, label in zip(factor_indices, labels, strict=True)
        if not label_is_complete(label)
    ]
    if missing:
        raise ValueError(f"Joint labelling response missing labels for factors {missing}.")
    return labels


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


def _debug_artifact_path(checkpoint_path: str | Path | None, *, chunk_number: int) -> Path | None:
    """Return the per-chunk debug artifact path for a failed joint labelling chunk."""
    if checkpoint_path is None:
        return None
    checkpoint = Path(checkpoint_path)
    return checkpoint.with_name(f"{checkpoint.stem}_debug_chunk_{chunk_number:02d}{checkpoint.suffix}")


def _write_joint_debug_artifact(
    *,
    checkpoint_path: str | Path | None,
    chunk_number: int,
    chunk: list[tuple[int, dict]],
    messages: list[dict],
    response_text: str | None,
    exc: Exception | None,
) -> Path | None:
    """Persist the failed joint-labelling request/response for debugging."""
    path = _debug_artifact_path(checkpoint_path, chunk_number=chunk_number)
    if path is None:
        return None

    factor_indices = [int(factor_data["factor_index"]) for _, factor_data in chunk]
    response_text = response_text or ""
    debug_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "factor_indices": factor_indices,
        "error": str(exc) if exc is not None else None,
        "response_length_chars": len(response_text),
        "json_candidate": _extract_json_candidate(response_text),
        "messages": messages,
        "response_text": response_text,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(debug_payload, f, indent=2, ensure_ascii=False)
    return path


async def _label_one_async(
    factor_data: dict,
    provider,
    build_messages: Callable[[dict], list[dict]],
    semaphore: asyncio.Semaphore,
    max_tokens: int,
) -> str:
    messages = build_messages(factor_data)
    async with semaphore:
        try:
            return await provider.generate_async(
                messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
        except Exception as e:
            fi = factor_data["factor_index"]
            logger.warning(f"Factor {fi} labelling failed: {e}")
            return f"(labelling failed: {e})"


def _run_labelling(
    factor_data_list: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    max_tokens: int = DEFAULT_PER_FACTOR_MAX_TOKENS,
    max_concurrent: int = 10,
    show_progress: bool = True,
    checkpoint_path: str | Path | None = None,
    build_messages: Callable[[dict], list[dict]],
    description: str,
) -> list[str]:
    """Run LLM labelling over aligned factor records with resumable checkpoints."""
    from src_dev.inference import InferenceConfig, get_provider

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
                factor_data,
                provider,
                build_messages,
                semaphore,
                max_tokens,
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
    finally:
        aclose = getattr(provider, "aclose", None)
        if aclose is not None:
            try:
                asyncio.run(aclose())
            except RuntimeError:
                logger.debug("Provider cleanup skipped because an event loop is already running.")
    print("Done.")
    return labels


def _run_joint_labelling(
    factor_data_list: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    max_tokens: int = DEFAULT_JOINT_MAX_TOKENS,
    max_factors_per_call: int | None = DEFAULT_JOINT_MAX_FACTORS_PER_CALL,
    checkpoint_path: str | Path | None = None,
    build_messages: Callable[[list[dict]], list[dict]],
    description: str,
) -> list[str]:
    """Run joint labelling requests over several factor chunks with resume support.

    Cached labels that are empty or start with the failure sentinel are treated as
    incomplete. Re-running with the same checkpoint therefore retries only the
    chunks containing incomplete factors.
    """
    from src_dev.inference import InferenceConfig, get_provider

    total = len(factor_data_list)
    if total == 0:
        return []

    if checkpoint_path is not None:
        existing = load_label_checkpoint(checkpoint_path, total)
        if len(existing) == total and all(label_is_complete(label) for label in existing):
            print(
                f"Resuming label checkpoint {checkpoint_path}: "
                f"{total}/{total} already complete"
            )
            return existing

    config = InferenceConfig(model=model, provider=provider)
    provider_instance = get_provider(provider, config)
    results = (
        load_label_checkpoint(checkpoint_path, total)
        if checkpoint_path is not None
        else [""] * total
    )
    completed = sum(1 for label in results if label_is_complete(label))
    index_factor_pairs = [
        (idx, factor_data)
        for idx, factor_data in enumerate(factor_data_list)
        if not label_is_complete(results[idx])
    ]
    if checkpoint_path is not None and completed:
        print(
            f"Resuming label checkpoint {checkpoint_path}: "
            f"{completed}/{total} already complete, retrying {len(index_factor_pairs)} incomplete"
        )
    if max_factors_per_call is None or max_factors_per_call <= 0:
        chunk_size = len(index_factor_pairs) or 1
    else:
        chunk_size = max_factors_per_call
    chunks = [
        index_factor_pairs[start : start + chunk_size]
        for start in range(0, len(index_factor_pairs), chunk_size)
    ]

    async def _run_once(chunk: list[tuple[int, dict]]) -> tuple[list[int], list[str]]:
        chunk_indices = [idx for idx, _ in chunk]
        chunk_factor_data = [factor_data for _, factor_data in chunk]
        factor_indices = [int(factor_data["factor_index"]) for factor_data in chunk_factor_data]
        messages = build_messages(chunk_factor_data)
        response_text = await provider_instance.generate_async(
            messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        labels = _parse_joint_label_response(response_text, factor_indices)
        return chunk_indices, labels

    print(
        f"Labelling {len(factor_data_list)} factors with {model} "
        f"({description})..."
    )
    try:
        async def _run_all() -> list[str]:
            if not chunks:
                return results
            for chunk_number, chunk in enumerate(chunks, start=1):
                chunk_factor_data = [factor_data for _, factor_data in chunk]
                messages = build_messages(chunk_factor_data)
                response_text: str | None = None
                try:
                    chunk_indices = [idx for idx, _ in chunk]
                    factor_indices = [int(factor_data["factor_index"]) for factor_data in chunk_factor_data]
                    response_text = await provider_instance.generate_async(
                        messages,
                        max_tokens=max_tokens,
                        temperature=0.2,
                    )
                    chunk_labels = _parse_joint_label_response(response_text, factor_indices)
                except Exception as exc:
                    failed_factor_indices = [
                        int(factor_data["factor_index"])
                        for _, factor_data in chunk
                    ]
                    debug_path = _write_joint_debug_artifact(
                        checkpoint_path=checkpoint_path,
                        chunk_number=chunk_number,
                        chunk=chunk,
                        messages=messages,
                        response_text=response_text,
                        exc=exc,
                    )
                    logger.warning(
                        "Joint labelling failed for factor chunk %s: %s%s",
                        failed_factor_indices,
                        exc,
                        f" [debug saved to {debug_path}]" if debug_path is not None else "",
                    )
                    failure_labels = [f"(labelling failed: {exc})"] * len(chunk)
                    for idx, label in zip(
                        [idx for idx, _ in chunk],
                        failure_labels,
                        strict=True,
                    ):
                        results[idx] = label
                else:
                    for idx, label in zip(chunk_indices, chunk_labels, strict=True):
                        results[idx] = label
                    debug_path = _debug_artifact_path(checkpoint_path, chunk_number=chunk_number)
                    if debug_path is not None and debug_path.exists():
                        debug_path.unlink()
                if checkpoint_path is not None:
                    save_label_checkpoint(results, checkpoint_path)
                print(f"  chunk {chunk_number}/{len(chunks)} complete", end="\r", flush=True)
            if chunks:
                print(" " * 32, end="\r", flush=True)
            return results

        try:
            asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                labels = pool.submit(asyncio.run, _run_all()).result()
        except RuntimeError:
            labels = asyncio.run(_run_all())
    finally:
        aclose = getattr(provider_instance, "aclose", None)
        if aclose is not None:
            try:
                asyncio.run(aclose())
            except RuntimeError:
                logger.debug("Provider cleanup skipped because an event loop is already running.")

    remaining_incomplete = sum(1 for label in labels if not label_is_complete(label))
    if remaining_incomplete:
        print(
            "Done with incomplete labels remaining: "
            f"{total - remaining_incomplete}/{total} complete, "
            f"{remaining_incomplete} will be retried on the next rerun."
        )
    else:
        print("Done.")
    return labels


def label_factors(
    extremes: list[dict],
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    top_n: int = 10,
    excerpt_chars: int = 40000,
    max_per_prompt: int = 10,
    prompt_format: Literal["grouped_json", "contrastive_jsonl"] = "grouped_json",
    max_tokens: int = DEFAULT_PER_FACTOR_MAX_TOKENS,
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
        max_tokens=max_tokens,
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


def label_factors_jointly(
    extremes: list[dict],
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    top_n: int = 6,
    excerpt_chars: int = 1200,
    max_per_prompt: int = 2,
    max_tokens: int = DEFAULT_JOINT_MAX_TOKENS,
    max_factors_per_call: int | None = DEFAULT_JOINT_MAX_FACTORS_PER_CALL,
    checkpoint_path: str | Path | None = None,
) -> list[str]:
    """Label several factors jointly with one distinctiveness-seeking prompt."""
    return _run_joint_labelling(
        extremes,
        model=model,
        provider=provider,
        max_tokens=max_tokens,
        max_factors_per_call=max_factors_per_call,
        checkpoint_path=checkpoint_path,
        build_messages=lambda factor_data_list: _build_joint_messages(
            factor_data_list,
            top_n=top_n,
            excerpt_chars=excerpt_chars,
            max_per_prompt=max_per_prompt,
        ),
        description=(
            f"joint_distinct, top_n={top_n}, max_per_prompt={max_per_prompt}, "
            f"excerpt_chars={excerpt_chars}"
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
    max_tokens: int = DEFAULT_PER_FACTOR_MAX_TOKENS,
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
        max_tokens=max_tokens,
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


def label_max_spread_factors_jointly(
    max_spread_factors: list[dict],
    *,
    strategy: MaxSpreadLabelStrategy,
    model: str = DEFAULT_MODEL,
    provider: str = DEFAULT_PROVIDER,
    top_n: int = 6,
    excerpt_chars: int = 800,
    max_tokens: int = DEFAULT_JOINT_MAX_TOKENS,
    max_factors_per_call: int | None = DEFAULT_JOINT_MAX_FACTORS_PER_CALL,
    checkpoint_path: str | Path | None = None,
) -> list[str]:
    """Label several max-spread factors jointly with one distinctiveness prompt."""
    if strategy not in MAX_SPREAD_STRATEGIES:
        raise ValueError(
            f"Unknown max-spread labelling strategy {strategy!r}; "
            f"expected one of {sorted(MAX_SPREAD_STRATEGIES)}"
        )
    strategy_cfg = MAX_SPREAD_STRATEGIES[strategy]
    return _run_joint_labelling(
        max_spread_factors,
        model=model,
        provider=provider,
        max_tokens=max_tokens,
        max_factors_per_call=max_factors_per_call,
        checkpoint_path=checkpoint_path,
        build_messages=lambda factor_data_list: _build_joint_max_spread_messages(
            factor_data_list,
            top_n=top_n,
            excerpt_chars=excerpt_chars,
            include_prompt=strategy_cfg["include_prompt"],
            include_scores=strategy_cfg["include_scores"],
        ),
        description=(
            f"joint_distinct_max_spread, strategy={strategy}, top_n={top_n}, "
            f"include_prompt={strategy_cfg['include_prompt']}, "
            f"include_scores={strategy_cfg['include_scores']}"
        ),
    )
