"""Rollout-set heuristics: turn-count and token-count per rollout.

Given a canonical rollout run directory, returns per-sample stats
(``sample_id``, message count, assistant/user turn count, total chars,
total tokens under the specified tokenizer) and a per-preset aggregate
summary.

Tokenization uses the preset's own administering model when possible,
since that's the number that governs context-fit at Stage 2. When a
tokenizer can't be loaded (e.g. a gated HF repo with no access), the
token count is left as NaN and a warning is printed. The canonical
``apply_chat_template`` output is measured — matching what vLLM would
actually see during administration.

Shared across analysis scripts so rollout-length inspection is a
first-class operation, not an ad-hoc util.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src_dev.datasets import load_samples
from src_dev.psychometric.chat_templates import ensure_chat_template

logger = logging.getLogger(__name__)


def compute_rollout_stats(
    rollout_dir: Path,
    *,
    preset_key: str,
    assistant_model: str | None = None,
) -> list[dict[str, Any]]:
    """Per-sample stats for a single rollout run.

    Args:
        rollout_dir: Canonical rollout run directory (must have
            ``canonical_samples.jsonl`` or be materialised via
            ``load_samples``).
        preset_key: Preset label used as a column in the output
            (useful when concatenating across presets).
        assistant_model: HF repo id of the tokenizer to use for token
            counts. If None or load fails, ``n_tokens`` is NaN.

    Returns:
        List of dicts, one per sample, with keys:
        ``preset_key, sample_id, n_messages, n_assistant_turns,
        n_user_turns, n_system_turns, n_tool_turns, n_chars, n_tokens``.
    """
    samples = load_samples(rollout_dir)

    tokenizer = _load_tokenizer(assistant_model) if assistant_model else None

    rows: list[dict[str, Any]] = []
    for s in samples:
        role_counts: dict[str, int] = {}
        for m in s.messages:
            role_counts[m.role] = role_counts.get(m.role, 0) + 1
        msgs_as_dicts = [{"role": m.role, "content": m.content} for m in s.messages]
        n_chars = sum(len(m["content"]) for m in msgs_as_dicts)

        n_tokens = float("nan")
        if tokenizer is not None:
            n_tokens = _token_count(tokenizer, msgs_as_dicts)

        rows.append({
            "preset_key": preset_key,
            "sample_id": s.sample_id,
            "n_messages": len(s.messages),
            "n_assistant_turns": role_counts.get("assistant", 0),
            "n_user_turns": role_counts.get("user", 0),
            "n_system_turns": role_counts.get("system", 0),
            "n_tool_turns": role_counts.get("tool", 0),
            "n_chars": n_chars,
            "n_tokens": n_tokens,
        })
    return rows


def summarise_stats(
    rows: list[dict[str, Any]],
    *,
    preset_key: str | None = None,
    context_budgets: tuple[int, ...] = (4096, 8192, 32768, 65536, 131072),
) -> dict[str, Any]:
    """Aggregate per-sample rows into a single preset-level summary row.

    If ``preset_key`` is given, only rows matching it are summarised;
    otherwise all rows (useful when rows is already per-preset filtered).
    """
    if preset_key is not None:
        rows = [r for r in rows if r.get("preset_key") == preset_key]

    out: dict[str, Any] = {"preset_key": preset_key, "n_rollouts": len(rows)}
    if not rows:
        return out

    for field in ("n_messages", "n_assistant_turns", "n_user_turns",
                  "n_chars", "n_tokens"):
        values = np.array([r[field] for r in rows], dtype=float)
        values = values[~np.isnan(values)] if field == "n_tokens" else values
        if len(values) == 0:
            out[f"{field}_mean"] = np.nan
            out[f"{field}_median"] = np.nan
            out[f"{field}_p90"] = np.nan
            out[f"{field}_max"] = np.nan
            continue
        out[f"{field}_mean"] = float(np.mean(values))
        out[f"{field}_median"] = float(np.median(values))
        out[f"{field}_p90"] = float(np.percentile(values, 90))
        out[f"{field}_max"] = float(np.max(values))

    # Fraction fitting each context budget (token-count only).
    token_values = np.array([r["n_tokens"] for r in rows], dtype=float)
    token_values = token_values[~np.isnan(token_values)]
    n_with_tokens = len(token_values)
    for budget in context_budgets:
        if n_with_tokens == 0:
            out[f"frac_under_{budget // 1024}k"] = np.nan
        else:
            out[f"frac_under_{budget // 1024}k"] = float((token_values < budget).mean())

    return out


def _load_tokenizer(model: str):
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Tokenizer load failed for %s (%s: %s). "
            "n_tokens will be NaN for this preset.",
            model, type(exc).__name__, str(exc)[:200],
        )
        return None
    ensure_chat_template(tokenizer, model)
    return tokenizer


def _token_count(tokenizer, messages: list[dict]) -> float:
    # Try the canonical chat-template path; if that raises (typically
    # "chat_template is not set"), fall back to plain concatenated
    # encoding. The fallback ignores turn-boundary tokens so it slightly
    # under-counts vs the actual serving prompt — good enough for the
    # length-distribution heuristics this module produces.
    has_template = bool(getattr(tokenizer, "chat_template", None))
    if has_template:
        try:
            ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=True,
            )
            return float(len(ids))
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "apply_chat_template raised (%s: %s); falling back to "
                "concatenated encoding.",
                type(exc).__name__, str(exc)[:120],
            )
    try:
        text = " ".join(m["content"] for m in messages)
        return float(len(tokenizer.encode(text, add_special_tokens=False)))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Fallback tokenization also failed (%s: %s); marking NaN.",
            type(exc).__name__, str(exc)[:160],
        )
        return float("nan")
