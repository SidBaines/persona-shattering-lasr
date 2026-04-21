"""Async questionnaire-administration loop.

One call to :func:`run_questionnaire_inference` takes a rollout directory +
questionnaire (items + column defs) + an output directory and produces the
persona × item response matrix, streaming every raw response to
``raw_responses.jsonl`` so it can resume mid-run.

Behaviour is byte-identical to the in-script ``_apply_questionnaire_async``
from ``scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py`` —
this module is a lift-and-parameterise: module globals become ``cfg.*``
reads against a :class:`~src_dev.psychometric.config.QuestionnaireStageConfig`.

The loop is persona-major so identical conversation prefixes stay adjacent
(good for prefix caching on local providers). On vLLM we optionally stack
multiple personas into one super-batch via
``cfg.vllm_personas_per_batch`` to trade some prefix-cache locality for
better GPU utilisation; all pending items for a persona still go in that
batch.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from src_dev.common.config import GenerationConfig
from src_dev.common.conversation_runtime import chunked
from src_dev.datasets import (
    find_consecutive_assistant_turn_sample_ids,
    load_samples,
    materialize_canonical_samples,
)
from src_dev.inference import InferenceConfig
from src_dev.inference.config import OpenRouterProviderConfig, RetryConfig, VllmProviderConfig
from src_dev.inference.providers import get_provider
from src_dev.inference.providers.base import PromptInput
from src_dev.psychometric.config import QuestionnaireStageConfig
from src_dev.psychometric.item_prompts import (
    build_item_prompt,
    build_questionnaire_messages,
    build_questionnaire_token_ids,
    item_prefill as _item_prefill,
    retry_message,
)
from src_dev.psychometric.response_encoding import (
    RESPONSE_MATRIX_ENCODING_VERSION,
    fill_matrix_from_choice,
    record_response,
)
from src_dev.psychometric.response_parsing import (
    parse_item_response,
    parse_top_logprobs_to_choice_probs,
)


def estimate_max_model_len(
    model: str,
    conversations: list[list[dict[str, str]]],
    items: list[dict],
    max_new_tokens: int,
    *,
    likert_phrasing: str = "direct",
    margin: int = 256,
) -> int:
    """Estimate the minimum vLLM max_model_len from actual data.

    Tokenizes the longest conversation + the longest questionnaire item
    prompt to compute the true maximum input length, then adds
    ``max_new_tokens`` and a safety ``margin``. Avoids allocating KV cache
    for the model's full context window when actual sequences are much
    shorter.

    Args:
        model: HuggingFace model name (used to load the tokenizer).
        conversations: Pre-built conversation histories.
        items: Questionnaire items.
        max_new_tokens: Max tokens to generate per response.
        likert_phrasing: Phrasing to pick when building Likert prompts.
        margin: Extra tokens for chat-template overhead / rounding.

    Returns:
        Recommended max_model_len (rounded up to the next multiple of 64).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    item_prompts = [build_item_prompt(item, likert_phrasing=likert_phrasing) for item in items]
    longest_item_prompt = max(item_prompts, key=len)

    longest_conv = max(conversations, key=lambda c: sum(len(m["content"]) for m in c))

    full_messages = list(longest_conv) + [{"role": "user", "content": longest_item_prompt}]

    if hasattr(tokenizer, "apply_chat_template"):
        token_ids = tokenizer.apply_chat_template(
            full_messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        max_input_tokens = len(token_ids)
    else:
        text = " ".join(m["content"] for m in full_messages)
        max_input_tokens = len(tokenizer.encode(text))

    longest_retry_msg = max(
        (retry_message(item) for item in items),
        key=len,
    )
    retry_overhead = len(tokenizer.encode(longest_retry_msg)) + max_new_tokens + 20

    raw = max_input_tokens + max_new_tokens + retry_overhead + margin
    max_model_len = ((raw + 63) // 64) * 64

    print(
        f"[Stage 2] Estimated max_model_len: {max_model_len} "
        f"(longest input: {max_input_tokens} tokens, "
        f"generation: {max_new_tokens}, retry overhead: {retry_overhead}, "
        f"margin: {margin})"
    )
    return max_model_len


def _filter_by_context_budget(
    samples,
    items: list[dict],
    *,
    model: str,
    max_context_tokens: int,
    max_new_tokens: int,
    buffer_tokens: int,
    likert_phrasing: str,
):
    """Drop samples whose (conv + longest item prompt) exceeds the budget.

    Budget = ``max_context_tokens - max_new_tokens - retry_overhead - buffer_tokens``.
    ``retry_overhead`` accounts for the longest retry message + a second
    ``max_new_tokens`` generation slot, matching ``estimate_max_model_len``.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    item_prompts = [build_item_prompt(it, likert_phrasing=likert_phrasing) for it in items]
    longest_item_prompt = max(item_prompts, key=len)

    longest_retry_msg = max((retry_message(it) for it in items), key=len)
    retry_overhead = len(tokenizer.encode(longest_retry_msg)) + max_new_tokens + 20

    input_budget = max_context_tokens - max_new_tokens - retry_overhead - buffer_tokens
    if input_budget <= 0:
        raise ValueError(
            f"max_context_tokens={max_context_tokens} is too small for "
            f"max_new_tokens={max_new_tokens} + retry_overhead={retry_overhead} "
            f"+ buffer={buffer_tokens}."
        )

    kept = []
    dropped_ids: list[str] = []
    for sample in samples:
        conv = [{"role": m.role, "content": m.content} for m in sample.messages]
        msgs = list(conv) + [{"role": "user", "content": longest_item_prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            n_in = len(
                tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            )
        else:
            n_in = len(tokenizer.encode(" ".join(m["content"] for m in msgs)))
        if n_in <= input_budget:
            kept.append(sample)
        else:
            dropped_ids.append(f"{sample.sample_id} ({n_in} tok)")

    n_dropped = len(samples) - len(kept)
    print(
        f"[Stage 2] Context-length filter: model={model} "
        f"max_context={max_context_tokens} input_budget={input_budget} "
        f"→ kept {len(kept)}/{len(samples)} samples (dropped {n_dropped})"
    )
    if dropped_ids:
        preview = ", ".join(dropped_ids[:5])
        more = f" …(+{len(dropped_ids) - 5} more)" if len(dropped_ids) > 5 else ""
        print(f"[Stage 2] First dropped samples: {preview}{more}")
    return kept


async def run_questionnaire_inference_async(
    cfg: QuestionnaireStageConfig,
    rollout_dir: Path,
    items: list[dict],
    column_defs: list[dict],
    output_dir: Path,
    *,
    num_conversation_turns: int,
    openrouter_provider_routing: dict | None = None,
    fc_pair_sign_alignment: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """Apply questionnaire items to all rollouts and produce the response matrix.

    See module docstring for loop-order and persona-stacking notes.

    Args:
        cfg: Questionnaire stage config (provider/model, vLLM knobs, logprob
            flags, reset mode, etc.).
        rollout_dir: Path to the rollout run directory.
        items: Flat list of questionnaire items (mixed types).
        column_defs: Column definitions for the response matrix.
        output_dir: Directory to save results.
        num_conversation_turns: Completeness threshold — samples with fewer
            assistant turns are dropped. Taken from the rollouts stage, not
            the questionnaire config, since that's where the rollout lives.
        openrouter_provider_routing: OpenRouter routing config to pass to
            ``InferenceConfig.openrouter``. None uses an empty config.
        fc_pair_sign_alignment: Whether to orient fc_pair column encoding to
            the per-item high pole (True) or the raw A/B letter (False).
            Must match the value used by ``load_questionnaire`` when the
            column defs were produced.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    materialize_canonical_samples(rollout_dir)
    samples = load_samples(rollout_dir)
    print(f"[Stage 2] Loaded {len(samples)} rollout samples")

    completed_samples = [
        s for s in samples
        if sum(1 for m in s.messages if m.role == "assistant") >= num_conversation_turns
    ]
    print(
        f"[Stage 2] {len(completed_samples)} completed rollouts "
        f"(>= {num_conversation_turns} assistant turns)"
    )

    bad_sample_ids = find_consecutive_assistant_turn_sample_ids(rollout_dir)
    if bad_sample_ids:
        n_before = len(completed_samples)
        completed_samples = [s for s in completed_samples if s.sample_id not in bad_sample_ids]
        print(
            f"[Stage 2] Excluded {n_before - len(completed_samples)} samples with consecutive "
            f"assistant turns (resume-bug artifact)"
        )

    if cfg.max_context_tokens is not None:
        completed_samples = _filter_by_context_budget(
            completed_samples,
            items,
            model=cfg.model,
            max_context_tokens=cfg.max_context_tokens,
            max_new_tokens=cfg.max_new_tokens,
            buffer_tokens=cfg.context_buffer_tokens,
            likert_phrasing=cfg.phrasing,
        )

    if not completed_samples:
        raise RuntimeError("No completed rollouts found. Stage 1 may have failed.")

    reset_mode = cfg.reset_mode
    if reset_mode not in ("none", "soft", "token_boundary"):
        raise ValueError(
            f"Unknown reset_mode={reset_mode!r}; expected one of "
            f"'none', 'soft', 'token_boundary'."
        )
    if reset_mode == "token_boundary" and cfg.provider != "vllm":
        raise ValueError(
            f"reset_mode='token_boundary' requires provider='vllm'; "
            f"got {cfg.provider!r}."
        )

    K = len(completed_samples)
    N_items = len(items)
    N_cols = len(column_defs)
    n_non_fa = sum(
        1 for it in items
        if it["type"] == "vignette" and "vignette" not in cfg.fa_blocks
    )
    print(
        f"[Stage 2] {N_items} items ({n_non_fa} administered but excluded from FA) "
        f"→ {N_cols} matrix columns | {K} personas | {K * N_items} calls"
    )
    print(f"[Stage 2] FA blocks: {list(cfg.fa_blocks)}")

    # Build conversation histories
    conversations: list[list[dict[str, str]]] = []
    metadata: list[dict] = []
    for sample in completed_samples:
        conv = [{"role": m.role, "content": m.content} for m in sample.messages]
        conversations.append(conv)
        metadata.append({
            "sample_id": sample.sample_id,
            "input_group_id": sample.input_group_id,
            "response_index": sample.response_index,
            "num_messages": len(conv),
        })

    # Pre-compute column index lookup: item_id -> list of (col_idx, dimension_or_None)
    item_to_cols: dict[str, list[tuple[int, str | None]]] = {}
    for col_idx, col in enumerate(column_defs):
        iid = col["item_id"]
        if iid not in item_to_cols:
            item_to_cols[iid] = []
        item_to_cols[iid].append((col_idx, col.get("dimension")))

    # Pre-compute vignette scoring: vig_id -> {option_label -> {dim -> score}}
    vig_scoring: dict[str, dict[str, dict[str, int]]] = {}
    for item in items:
        if item["type"] == "vignette":
            vig_scoring[item["id"]] = {
                opt["label"]: opt.get("scoring", {})
                for opt in item["options"]
            }

    # Reverse-keyed lookup for Likert items
    likert_reverse: dict[str, bool] = {
        item["id"]: item.get("reverse_keyed", False)
        for item in items
        if item["type"] == "likert"
    }

    # Per-item answer_mapping for trait_mcq dispatch.
    trait_mcq_mapping: dict[str, dict[str, int]] = {}
    for item in items:
        if item["type"] != "trait_mcq":
            continue
        mapping = item.get("answer_mapping")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(
                f"trait_mcq item {item['id']!r} is missing a non-empty "
                "answer_mapping; cannot orient its column to the trait pole."
            )
        trait_mcq_mapping[item["id"]] = {str(k): int(v) for k, v in mapping.items()}

    # fc_pair: item_id -> high_option ('A' or 'B'). Used to sign-align the
    # +1/-1 encoding with the axis polarity at matrix-fill time.
    fc_pair_high: dict[str, str] = {
        item["id"]: (item["high_option"] if fc_pair_sign_alignment else "A")
        for item in items
        if item["type"] == "fc_pair"
    }

    # Restore state from raw_responses.jsonl (single source of truth).
    completed_cells: set[tuple[int, str]] = set()
    response_matrix = np.full((K, N_cols), np.nan)
    parse_failures: list[dict] = []

    raw_responses_log = output_dir / "raw_responses.jsonl"
    if raw_responses_log.exists():
        with open(raw_responses_log, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                k_entry = entry["k"]
                iid = entry["item_id"]
                completed_cells.add((k_entry, iid))
                choice = entry.get("parsed_choice")
                if choice is not None:
                    resumed_probs = entry.get("probs")
                    resumed_probs = (
                        {str(kk): float(vv) for kk, vv in resumed_probs.items()}
                        if isinstance(resumed_probs, dict) and resumed_probs
                        else None
                    )
                    fill_matrix_from_choice(
                        response_matrix, k_entry, iid, choice,
                        item_to_cols, vig_scoring, likert_reverse,
                        trait_mcq_mapping=trait_mcq_mapping,
                        fc_pair_high=fc_pair_high,
                        choice_probs=resumed_probs,
                    )
        if completed_cells:
            print(f"[Stage 2] Resuming: {len(completed_cells)} cells already done")

    # Fast path: every cell is covered by raw_responses.jsonl, so no inference
    # is needed. Fires on an encoding-version rebuild (all cells cached from
    # the previous run, just re-scored through the current
    # fill_matrix_from_choice).
    expected_cells = K * N_items
    if completed_cells and len(completed_cells) >= expected_cells:
        print(
            f"[Stage 2] All {expected_cells} cells already in raw_responses.jsonl — "
            "skipping inference and writing matrix directly."
        )
        np.save(output_dir / "response_matrix.npy", response_matrix)
        with open(output_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
            for meta in metadata:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        with open(output_dir / "items.json", "w", encoding="utf-8") as f:
            json.dump(column_defs, f, indent=2, ensure_ascii=False)
        with open(output_dir / "encoding_version.json", "w", encoding="utf-8") as f:
            json.dump(
                {"response_matrix_encoding_version": RESPONSE_MATRIX_ENCODING_VERSION},
                f,
            )
        valid_count = int(np.sum(~np.isnan(response_matrix)))
        print(
            f"[Stage 2] Complete: {valid_count}/{K * N_cols} valid matrix cells "
            "(rebuilt from cached responses)"
        )
        return response_matrix, metadata

    # Set up inference provider — use questionnaire-specific model/provider
    vllm_kwargs = {}
    if cfg.provider == "vllm":
        max_model_len = estimate_max_model_len(
            cfg.model, conversations, items, cfg.max_new_tokens,
            likert_phrasing=cfg.phrasing,
        )
        if cfg.max_context_tokens is not None and max_model_len > cfg.max_context_tokens:
            print(
                f"[Stage 2] Clamping max_model_len {max_model_len} → "
                f"{cfg.max_context_tokens} (cfg.max_context_tokens)"
            )
            max_model_len = cfg.max_context_tokens
        vllm_kwargs["vllm"] = VllmProviderConfig(
            gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=cfg.vllm_tensor_parallel_size,
        )

    questionnaire_config = InferenceConfig(
        model=cfg.model,
        provider=cfg.provider,
        generation=GenerationConfig(
            max_new_tokens=cfg.max_new_tokens,
            temperature=0.0,
            do_sample=False,
        ),
        max_concurrent=cfg.max_concurrent,
        timeout=cfg.timeout,
        retry=RetryConfig(max_retries=3, backoff_factor=2.0),
        continue_on_error=True,
        log_failures=True,
        openrouter=OpenRouterProviderConfig(
            provider_routing=openrouter_provider_routing or {}
        ),
        **vllm_kwargs,
    )
    provider = get_provider(cfg.provider, questionnaire_config)

    # Persona-major batch size: always 1 for remote providers; configurable
    # stacking on vLLM.
    persona_batch_size = 1
    if cfg.provider == "vllm":
        persona_batch_size = max(1, cfg.vllm_personas_per_batch)
        print(
            "[Stage 2] vLLM persona stacking enabled: "
            f"{persona_batch_size} persona(s) per batch"
        )
    else:
        print(
            "[Stage 2] Persona stacking disabled for provider "
            f"{cfg.provider!r}; using 1 persona per batch"
        )

    use_logprobs_for_trait = cfg.use_logprobs and cfg.provider == "vllm"
    if use_logprobs_for_trait:
        print(
            f"[Stage 2] choice-item logprob mode ON (trait_mcq + fc_pair) "
            f"(top_logprobs={cfg.top_logprobs}, "
            f"temperature={cfg.logprob_temperature})"
        )
    elif cfg.use_logprobs:
        print(
            "[Stage 2] use_logprobs=True but provider is "
            f"{cfg.provider!r}; logprob mode only supports 'vllm'. "
            "Falling back to greedy decoding."
        )

    # Reset-mode wiring: under "token_boundary" we build raw token-ID prompts
    # using the questionnaire model's tokenizer, and dispatch generation to
    # the vLLM provider's prompt_token_ids API.
    print(f"[Stage 2] Reset mode: {reset_mode}")
    _reset_tokenizer = None
    if reset_mode == "token_boundary":
        from transformers import AutoTokenizer
        _reset_tokenizer = AutoTokenizer.from_pretrained(
            cfg.model, use_fast=True
        )
        print(
            f"[Stage 2] token_boundary tokenizer loaded: {cfg.model} "
            f"(boundary={cfg.boundary_token!r})"
        )
        # Pre-flight sanity check: decode the boundary region for the first
        # (persona, item) pair so the user can visually verify.
        if conversations and items:
            sample_tokens = build_questionnaire_token_ids(
                _reset_tokenizer, conversations[0], items[0],
                boundary_token=cfg.boundary_token,
                likert_phrasing=cfg.phrasing,
            )
            boundary_ids = (
                [cfg.boundary_token]
                if isinstance(cfg.boundary_token, int)
                else None
            )
            if boundary_ids is None and isinstance(cfg.boundary_token, str):
                boundary_ids = [
                    _reset_tokenizer.convert_tokens_to_ids(cfg.boundary_token)
                ]
            elif isinstance(cfg.boundary_token, (list, tuple)):
                boundary_ids = list(cfg.boundary_token)
            boundary_idx = next(
                (i for i, t in enumerate(sample_tokens) if t in set(boundary_ids or [])),
                None,
            )
            if boundary_idx is not None:
                left_ctx = max(0, boundary_idx - 6)
                right_ctx = min(len(sample_tokens), boundary_idx + len(boundary_ids) + 6)
                pre = _reset_tokenizer.decode(sample_tokens[left_ctx:boundary_idx])
                mid = _reset_tokenizer.decode(
                    sample_tokens[boundary_idx : boundary_idx + len(boundary_ids)]
                )
                post = _reset_tokenizer.decode(
                    sample_tokens[boundary_idx + len(boundary_ids) : right_ctx]
                )
                print("[Stage 2] token_boundary preview (first persona × item):")
                print(f"  total_tokens={len(sample_tokens)} boundary_idx={boundary_idx}")
                print(f"  …{pre!r}  ||boundary||  {mid!r}  →  {post!r}…")

    # raw_responses.jsonl is kept open for the full stage and flushed after
    # each persona batch — safe for resume.
    with open(raw_responses_log, "a", encoding="utf-8") as log_fh:
        persona_batches = chunked(list(range(K)), persona_batch_size)
        for batch_idx, persona_batch in enumerate(persona_batches, start=1):
            text_entries: list[tuple[int, dict]] = []
            text_prompts: list[PromptInput] = []
            lp_entries: list[tuple[int, dict]] = []
            lp_prompts: list[PromptInput] = []
            active_personas: list[int] = []

            text_token_ids: list[list[int]] = []
            lp_token_ids: list[list[int]] = []

            for k in persona_batch:
                pending_items = [
                    (item_idx, item) for item_idx, item in enumerate(items)
                    if (k, item["id"]) not in completed_cells
                ]
                if not pending_items:
                    continue
                active_personas.append(k)
                for _item_idx, item in pending_items:
                    use_lp = (
                        use_logprobs_for_trait
                        and item["type"] in ("trait_mcq", "fc_pair")
                    )
                    if reset_mode == "token_boundary":
                        token_ids = build_questionnaire_token_ids(
                            _reset_tokenizer, conversations[k], item,
                            boundary_token=cfg.boundary_token,
                            likert_phrasing=cfg.phrasing,
                        )
                        if use_lp:
                            lp_entries.append((k, item))
                            lp_token_ids.append(token_ids)
                        else:
                            text_entries.append((k, item))
                            text_token_ids.append(token_ids)
                    else:
                        prompt = build_questionnaire_messages(
                            conversations[k], item,
                            reset_mode=reset_mode,
                            soft_reset_system_prompt=cfg.soft_reset_system_prompt,
                            likert_phrasing=cfg.phrasing,
                        )
                        if use_lp:
                            lp_entries.append((k, item))
                            lp_prompts.append(prompt)
                        else:
                            text_entries.append((k, item))
                            text_prompts.append(prompt)

            if not text_prompts and not lp_prompts and not text_token_ids and not lp_token_ids:
                continue

            text_responses: list[str] = []
            if text_prompts:
                text_responses, _usage, _failed = await provider.generate_batch_with_metadata_async(
                    text_prompts
                )
            elif text_token_ids:
                text_responses = await provider.generate_batch_from_token_ids_async(
                    text_token_ids
                )

            lp_outputs: list[dict] = []
            if lp_prompts:
                lp_outputs = await provider.generate_batch_logprobs_async(
                    lp_prompts,
                    max_tokens=1,
                    top_logprobs=cfg.top_logprobs,
                    temperature=cfg.logprob_temperature,
                )
            elif lp_token_ids:
                lp_outputs = await provider.generate_batch_logprobs_from_token_ids_async(
                    lp_token_ids,
                    max_tokens=1,
                    top_logprobs=cfg.top_logprobs,
                    temperature=cfg.logprob_temperature,
                )

            # Parse and record; collect items needing a retry.
            retry_needed: list[tuple[int, dict, str]] = []
            for (k, item), raw_text in zip(text_entries, text_responses):
                item_id = item["id"]
                choice = parse_item_response(item, raw_text)
                if choice is None and raw_text:
                    retry_needed.append((k, item, raw_text))
                elif choice is not None:
                    record_response(
                        response_matrix, k, item, choice, raw_text,
                        item_to_cols, vig_scoring, likert_reverse,
                        log_fh,
                        trait_mcq_mapping=trait_mcq_mapping,
                        fc_pair_high=fc_pair_high,
                    )
                    completed_cells.add((k, item_id))
                else:
                    parse_failures.append(
                        {"k": k, "item_id": item_id, "raw_response": raw_text}
                    )
                    completed_cells.add((k, item_id))
                    log_fh.write(
                        json.dumps(
                            {
                                "k": k,
                                "item_id": item_id,
                                "item_type": item["type"],
                                "parsed_choice": None,
                                "raw": raw_text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            for (k, item), lp_out in zip(lp_entries, lp_outputs):
                item_id = item["id"]
                raw_text = lp_out.get("text", "")
                per_token = lp_out.get("logprobs_per_token") or []
                first_token_logprobs: dict[str, float] = per_token[0] if per_token else {}
                num_choices = 2 if item["type"] == "fc_pair" else 4
                probs, choice_mass = parse_top_logprobs_to_choice_probs(
                    first_token_logprobs, num_choices=num_choices,
                )
                if probs:
                    best_letter = max(probs, key=probs.get)
                    fill_matrix_from_choice(
                        response_matrix, k, item_id, best_letter,
                        item_to_cols, vig_scoring, likert_reverse,
                        trait_mcq_mapping=trait_mcq_mapping,
                        fc_pair_high=fc_pair_high,
                        choice_probs=probs,
                    )
                    log_fh.write(
                        json.dumps(
                            {
                                "k": k,
                                "item_id": item_id,
                                "item_type": item["type"],
                                "parsed_choice": best_letter,
                                "raw": raw_text,
                                "probs": {k_: round(v, 6) for k_, v in probs.items()},
                                "choice_mass": round(choice_mass, 6),
                                "scoring_method": "logprob",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    completed_cells.add((k, item_id))
                else:
                    parse_failures.append(
                        {"k": k, "item_id": item_id, "raw_response": raw_text,
                         "reason": "no choice letter in top logprobs"}
                    )
                    log_fh.write(
                        json.dumps(
                            {
                                "k": k,
                                "item_id": item_id,
                                "item_type": item["type"],
                                "parsed_choice": None,
                                "raw": raw_text,
                                "probs": {},
                                "choice_mass": 0.0,
                                "scoring_method": "logprob",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    completed_cells.add((k, item_id))

            for _attempt in range(cfg.max_parse_retries):
                if not retry_needed:
                    break
                retry_prompts: list[PromptInput] = []
                retry_token_ids: list[list[int]] = []
                for k, item, prev_raw in retry_needed:
                    prefill = _item_prefill(item, likert_phrasing=cfg.phrasing)
                    if reset_mode == "token_boundary":
                        from src_dev.inference.conversation_reset import (
                            build_token_ids_retry_prompt,
                        )
                        retry_prompt = build_token_ids_retry_prompt(
                            _reset_tokenizer,
                            conversations[k],
                            build_item_prompt(item, likert_phrasing=cfg.phrasing),
                            prior_assistant_text=prev_raw,
                            retry_user_content=retry_message(item),
                            boundary_token=cfg.boundary_token,
                            trait_mcq_prefill=prefill,
                        )
                        retry_token_ids.append(retry_prompt.token_ids)
                    else:
                        # Under "soft" reset we still want the reset system
                        # message at the rollout/item boundary on retry.
                        msgs = list(conversations[k])
                        if reset_mode == "soft":
                            msgs.append({
                                "role": "system",
                                "content": cfg.soft_reset_system_prompt,
                            })
                        msgs.append({
                            "role": "user",
                            "content": build_item_prompt(item, likert_phrasing=cfg.phrasing),
                        })
                        if prefill is not None:
                            # Reconstruct the full prior assistant turn (prefill + continuation)
                            msgs.append({"role": "assistant", "content": prefill + prev_raw})
                            msgs.append({"role": "user", "content": retry_message(item)})
                            msgs.append({"role": "assistant", "content": prefill})
                        else:
                            msgs.append({"role": "assistant", "content": prev_raw})
                            msgs.append({"role": "user", "content": retry_message(item)})
                        retry_prompts.append(msgs)

                if retry_token_ids:
                    retry_responses = await provider.generate_batch_from_token_ids_async(
                        retry_token_ids
                    )
                else:
                    retry_responses, _, _ = await provider.generate_batch_with_metadata_async(
                        retry_prompts
                    )

                still_needed: list[tuple[int, dict, str]] = []
                for (k, item, _prev_raw), retry_text in zip(
                    retry_needed, retry_responses
                ):
                    choice = parse_item_response(item, retry_text)
                    if choice is not None:
                        record_response(
                            response_matrix, k, item, choice, retry_text,
                            item_to_cols, vig_scoring, likert_reverse,
                            log_fh,
                            trait_mcq_mapping=trait_mcq_mapping,
                            fc_pair_high=fc_pair_high,
                        )
                        completed_cells.add((k, item["id"]))
                    else:
                        still_needed.append((k, item, retry_text))
                retry_needed = still_needed

            for k, item, raw_text in retry_needed:
                item_id = item["id"]
                parse_failures.append(
                    {"k": k, "item_id": item_id, "raw_response": raw_text}
                )
                completed_cells.add((k, item_id))
                log_fh.write(
                    json.dumps(
                        {
                            "k": k,
                            "item_id": item_id,
                            "item_type": item["type"],
                            "parsed_choice": None,
                            "raw": raw_text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            log_fh.flush()

            done = len(completed_cells)
            total = K * N_items
            batch_start = active_personas[0] + 1
            batch_end = active_personas[-1] + 1
            print(
                f"[Stage 2] Persona batch {batch_idx}/{len(persona_batches)} "
                f"({batch_start}-{batch_end}/{K}) done | "
                f"{done}/{total} ({done/total*100:.1f}%)"
            )

    # Save outputs
    np.save(output_dir / "response_matrix.npy", response_matrix)

    with open(output_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for meta in metadata:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    with open(output_dir / "items.json", "w", encoding="utf-8") as f:
        json.dump(column_defs, f, indent=2, ensure_ascii=False)

    # Marker recording which encoding version produced the saved matrix.
    # Consumed downstream to detect stale caches after an encoding-semantics
    # change; a mismatch triggers a rebuild from raw_responses.jsonl
    # (no re-inference) rather than a full regeneration.
    with open(output_dir / "encoding_version.json", "w", encoding="utf-8") as f:
        json.dump(
            {"response_matrix_encoding_version": RESPONSE_MATRIX_ENCODING_VERSION},
            f,
        )

    if parse_failures:
        with open(output_dir / "parse_failures.jsonl", "w", encoding="utf-8") as f:
            for pf in parse_failures:
                f.write(json.dumps(pf, ensure_ascii=False) + "\n")

    valid_count = int(np.sum(~np.isnan(response_matrix)))
    print(
        f"[Stage 2] Complete: {valid_count}/{K * N_cols} valid matrix cells | "
        f"{len(parse_failures)} parse failures"
    )

    return response_matrix, metadata


def run_questionnaire_inference(
    cfg: QuestionnaireStageConfig,
    rollout_dir: Path,
    items: list[dict],
    column_defs: list[dict],
    output_dir: Path,
    *,
    num_conversation_turns: int,
    openrouter_provider_routing: dict | None = None,
    fc_pair_sign_alignment: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """Sync wrapper around :func:`run_questionnaire_inference_async`."""
    return asyncio.run(
        run_questionnaire_inference_async(
            cfg,
            rollout_dir,
            items,
            column_defs,
            output_dir,
            num_conversation_turns=num_conversation_turns,
            openrouter_provider_routing=openrouter_provider_routing,
            fc_pair_sign_alignment=fc_pair_sign_alignment,
        )
    )
