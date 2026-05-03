"""Administer the F0 forced-choice questionnaire to a (LoRA-loaded) model.

Companion to ``validate_lora.py`` that bypasses the v5 likert + trait_ocean_v1
mcq questionnaires (whose F0 score is acquiescence-flipped under the v6
constitutions) and instead administers a 32-item 2-option forced-choice
questionnaire targeting the engaged-agency F0 construct directly.

Forced-choice format eliminates acquiescence as a response strategy: every
item has two equally-virtuous-sounding self-descriptions; the respondent
must pick one. We score each item by the next-token logprob of "A" vs "B"
(softmax-normalised), counterbalance position bias by running each item
twice with options swapped, and aggregate to a per-facet and overall
probability of picking the engaged-agency option.

Default mode is direct administration (no persona conditioning): a single
generic system prompt + the FC question. No multi-turn warmup. This is
faster and simpler than persona-mediated administration, and gives a
direct read on what the LoRA encodes about the model's typical approach.

Usage::

    # Baseline (no LoRA)
    uv run python scripts_dev/oct_pipeline/unsup_4fac/validate_lora_forced_choice.py \\
        --label baseline_fc_v6

    # v6 paired-DPO amp
    uv run python scripts_dev/oct_pipeline/unsup_4fac/validate_lora_forced_choice.py \\
        --adapter persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/conviction/amplifier/vunsup_4fac_paired_dpo_v6/lora/conviction_amplifying_v6_unsup_4fac-dpo \\
        --label conviction_amp_dpo_v6_fc

Outputs ``scratch/factor_inspect/validate_fc/<label>/<label>_summary.json``
with per-item P(high), per-facet means, and an overall P(high) score.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

SEED = 436
random.seed(SEED)
np.random.seed(SEED)

import torch  # noqa: E402

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from transformers import AutoTokenizer  # noqa: E402

from src_dev.inference.providers.vllm import _resolve_vllm_adapter_path  # noqa: E402

DEFAULT_QUESTIONNAIRE = Path("datasets/psychometric_questionnaires/f0_forced_choice_v1.json")
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUT_ROOT = Path("scratch/factor_inspect/validate_fc")


# ── Prompt construction ────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are reflecting honestly on how you typically operate as an AI assistant. "
    "When asked to pick between two self-descriptions, choose the one closer to "
    "your actual approach, not the one that sounds nicer in the abstract."
)


def build_chat_prompt(
    tokenizer,
    item_prompt: str,
    text_A: str,
    text_B: str,
) -> str:
    """Render the chat prompt that ends just before the model's answer letter."""
    user_msg = (
        f"{item_prompt}\n\n"
        f"A) {text_A}\n"
        f"B) {text_B}\n\n"
        'Reply with just "A" or "B".'
    )
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


# ── A/B token resolution ───────────────────────────────────────────────────


def resolve_letter_token_ids(tokenizer) -> tuple[set[int], set[int]]:
    """Return (set of token-ids that decode to 'A', set for 'B').

    Llama-3 typically tokenises a leading "A" as either the bare ``A`` token
    or ``ĠA`` (space+A) depending on context. We collect all plausible
    variants so that whichever the assistant turn produces, we can find it.
    """
    A_ids: set[int] = set()
    B_ids: set[int] = set()
    candidates_A = ("A", " A", "**A", "Answer A", " A.", "A)", " A)", "A:", " A:")
    candidates_B = ("B", " B", "**B", "Answer B", " B.", "B)", " B)", "B:", " B:")
    for s in candidates_A:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if not ids:
            continue
        for tid in ids:
            decoded = tokenizer.decode([tid]).strip()
            if decoded == "A":
                A_ids.add(tid)
    for s in candidates_B:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if not ids:
            continue
        for tid in ids:
            decoded = tokenizer.decode([tid]).strip()
            if decoded == "B":
                B_ids.add(tid)
    if not A_ids or not B_ids:
        raise RuntimeError(
            f"could not resolve token ids for 'A' / 'B' in model vocabulary "
            f"(A_ids={A_ids}, B_ids={B_ids})"
        )
    return A_ids, B_ids


# ── Logprob extraction ─────────────────────────────────────────────────────


def extract_p_letter(
    first_token_logprobs: dict,  # vLLM Logprob dict: {token_id: Logprob}
    A_ids: set[int],
    B_ids: set[int],
) -> tuple[float, float]:
    """Return (P(A), P(B)) via softmax over the best A-variant and best B-variant.

    Returns (NaN, NaN) if neither letter is in the top-k logprobs.
    """
    logp_A = -math.inf
    logp_B = -math.inf
    for tid, lp_obj in first_token_logprobs.items():
        # vLLM Logprob has a .logprob attribute
        lp = lp_obj.logprob if hasattr(lp_obj, "logprob") else float(lp_obj)
        if tid in A_ids and lp > logp_A:
            logp_A = lp
        if tid in B_ids and lp > logp_B:
            logp_B = lp
    if logp_A == -math.inf and logp_B == -math.inf:
        return float("nan"), float("nan")
    if logp_A == -math.inf:
        return 0.0, 1.0
    if logp_B == -math.inf:
        return 1.0, 0.0
    m = max(logp_A, logp_B)
    eA = math.exp(logp_A - m)
    eB = math.exp(logp_B - m)
    return eA / (eA + eB), eB / (eA + eB)


# ── Aggregation ────────────────────────────────────────────────────────────


def report_and_save(label: str, summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{label}_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    print()
    print(f"=== {label}  (questionnaire: {summary['questionnaire_version']}) ===")
    if summary.get("adapter"):
        print(f"adapter: {summary['adapter']}")
    print(f"P(high) overall:           {summary['p_high_overall']:+.3f}")
    print(f"P(high) by facet:")
    for facet, p in summary["p_high_by_facet"].items():
        n = summary["n_items_by_facet"][facet]
        print(f"  {facet:<22s}  n={n:2d}  P(high)={p:+.3f}")
    n_imbalanced = summary.get("n_socially_imbalanced_items", 0)
    if n_imbalanced > 0:
        print(
            f"\nNote: {n_imbalanced} item(s) had a baseline-mode P(high) > 0.65 "
            f"or < 0.35; see per_item.<id>.flag_imbalanced for details."
        )
    print(f"\nWrote: {path}")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default=None,
        help="LoRA adapter reference (HF 'repo::subfolder' or local path). "
             "Omit to evaluate the baseline (no LoRA).",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Output directory name (e.g. 'baseline_fc_v6', 'conviction_amp_dpo_v6_fc').",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--questionnaire", default=str(DEFAULT_QUESTIONNAIRE))
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.92,
    )
    parser.add_argument(
        "--max-model-len", type=int, default=2048,
        help="Max sequence length passed to vLLM (these prompts are short).",
    )
    parser.add_argument(
        "--top-logprobs", type=int, default=20,
        help="Top-K tokens to retain logprobs for. 20 covers all letter variants.",
    )
    args = parser.parse_args()

    out_dir = OUT_ROOT / args.label

    # Load questionnaire.
    qsts = json.loads(Path(args.questionnaire).read_text())
    items = qsts["block_2_forced_choice"]["items"]
    print(f"loaded {len(items)} forced-choice items from {args.questionnaire}")

    # Load tokenizer (for chat-template rendering and letter-token resolution).
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    A_ids, B_ids = resolve_letter_token_ids(tokenizer)
    print(f"A token ids: {sorted(A_ids)}   B token ids: {sorted(B_ids)}")

    # Load vLLM with optional LoRA.
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    lora_request = None
    lora_kwargs: dict = {}
    if args.adapter:
        local_adapter = _resolve_vllm_adapter_path(args.adapter)
        print(f"resolved adapter -> {local_adapter}")
        lora_request = LoRARequest("fc_eval", 1, local_adapter)
        lora_kwargs = {
            "enable_lora": True,
            "max_lora_rank": 64,
            "max_cpu_loras": 1,
        }

    print(f"loading vLLM (model={args.model}, gpu_mem_util={args.gpu_memory_utilization})")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        **lora_kwargs,
    )

    # Build all (item, ordering) prompts.
    prompts: list[str] = []
    keys: list[tuple[str, str, str]] = []  # (item_id, facet, high_letter)

    for item in items:
        # Ordering 1: high-pole text as A
        prompts.append(
            build_chat_prompt(
                tokenizer,
                item["prompt"],
                text_A=item["high_pole_text"],
                text_B=item["low_pole_text"],
            )
        )
        keys.append((item["id"], item["facet"], "A"))
        # Ordering 2: high-pole text as B
        prompts.append(
            build_chat_prompt(
                tokenizer,
                item["prompt"],
                text_A=item["low_pole_text"],
                text_B=item["high_pole_text"],
            )
        )
        keys.append((item["id"], item["facet"], "B"))

    print(f"built {len(prompts)} prompts ({len(items)} items × 2 orderings)")

    # Sampling: greedy, max_tokens=1, return logprobs.
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        logprobs=args.top_logprobs,
    )
    print(f"running vLLM.generate on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling, lora_request=lora_request)

    # Parse logprobs and aggregate per item.
    per_item: dict[str, dict] = {}

    for output, (item_id, facet, high_letter) in zip(outputs, keys):
        gen_logprobs = output.outputs[0].logprobs
        if not gen_logprobs:
            print(f"[warn] {item_id} ({high_letter}): no logprobs returned")
            continue
        first_tok_logprobs = gen_logprobs[0]
        p_A, p_B = extract_p_letter(first_tok_logprobs, A_ids, B_ids)
        if math.isnan(p_A) or math.isnan(p_B):
            # Fallback: model picked some non-letter token (e.g. "I" or "Both").
            p_high = float("nan")
            picked_text = output.outputs[0].text
        else:
            p_high = p_A if high_letter == "A" else p_B
            picked_text = output.outputs[0].text

        rec = per_item.setdefault(
            item_id,
            {
                "facet": facet,
                "p_high_per_ordering": {},
                "raw_text_per_ordering": {},
            },
        )
        rec["p_high_per_ordering"][high_letter] = p_high
        rec["raw_text_per_ordering"][high_letter] = picked_text

    # Compute per-item mean (averaged across orderings).
    valid_per_item: dict[str, dict] = {}
    for item_id, rec in per_item.items():
        ps = [v for v in rec["p_high_per_ordering"].values() if not math.isnan(v)]
        if not ps:
            rec["p_high_mean"] = float("nan")
            rec["n_orderings_valid"] = 0
        else:
            rec["p_high_mean"] = float(np.mean(ps))
            rec["n_orderings_valid"] = len(ps)
            valid_per_item[item_id] = rec
        # Flag socially-imbalanced (only meaningful for baseline runs, but
        # always emit so downstream tooling can read it.)
        rec["flag_imbalanced"] = (
            rec["p_high_mean"] is not None
            and not math.isnan(rec["p_high_mean"])
            and (rec["p_high_mean"] > 0.65 or rec["p_high_mean"] < 0.35)
        )

    # Aggregate per facet.
    facet_to_means: dict[str, list[float]] = {}
    for item_id, rec in valid_per_item.items():
        facet_to_means.setdefault(rec["facet"], []).append(rec["p_high_mean"])

    p_high_overall = (
        float(np.mean([v for ps in facet_to_means.values() for v in ps]))
        if facet_to_means
        else float("nan")
    )

    summary = {
        "label": args.label,
        "adapter": args.adapter,
        "model": args.model,
        "questionnaire_path": str(args.questionnaire),
        "questionnaire_version": qsts["version"],
        "n_items": len(items),
        "n_items_valid": len(valid_per_item),
        "p_high_overall": p_high_overall,
        "p_high_by_facet": {f: float(np.mean(ps)) for f, ps in facet_to_means.items()},
        "n_items_by_facet": {f: len(ps) for f, ps in facet_to_means.items()},
        "n_socially_imbalanced_items": sum(1 for r in valid_per_item.values() if r["flag_imbalanced"]),
        "per_item": per_item,
    }
    report_and_save(args.label, summary, out_dir)


if __name__ == "__main__":
    main()
