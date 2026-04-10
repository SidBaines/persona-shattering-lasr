"""Inspect first-token logprob distributions for a base model on TRAIT MCQ.

Loads TRAIT questions, formats them identically to the logprobs eval suite,
runs them through a local HF model with a configurable prefill, and prints
the top-token distribution so we can pick the best prefill for each model.

Usage
-----
    uv run python scripts_dev/personality_evals/inspect_prefill_tokens.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --prefill "ANSWER:" \
        --samples-per-trait 200
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# ---------------------------------------------------------------------------
# Constants — replicate the eval suite's prompt format exactly
# ---------------------------------------------------------------------------

SEED = 42

SYSTEM_PROMPT = (
    "Now I will briefly describe a situation. "
    "Please read a given situation and tell me which option that you want to suggest."
)

SINGLE_ANSWER_TEMPLATE = (
    "Answer the following multiple choice question. The entire content of your "
    "response should be of the following format: 'ANSWER: $LETTER' (without "
    "quotes) where LETTER is one of {letters}.\n\n{question}\n\n{choices}"
)

OCEAN_TRAITS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]

CHOICE_LETTERS = ("A", "B", "C", "D")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


_TRAIT_CACHE_DIR = Path(
    "/root/.cache/huggingface/datasets/mirlab___trait/default/0.0.0/"
    "8b31c078cb897c3917d2ee48735d0c15030680e0"
)


def _load_trait_samples(
    samples_per_trait: int,
    trait_splits: list[str],
) -> list[dict]:
    """Load TRAIT MCQ samples from cached arrow files."""
    samples: list[dict] = []
    for trait in trait_splits:
        arrow_path = _TRAIT_CACHE_DIR / f"trait-{trait}.arrow"
        if not arrow_path.exists():
            raise FileNotFoundError(f"TRAIT cache not found: {arrow_path}")
        table = pa.ipc.open_stream(str(arrow_path)).read_all()
        rows = table.to_pydict()
        n = len(rows["question"])
        indices = list(range(n))
        random.shuffle(indices)
        for idx in indices[:samples_per_trait]:
            choices = [
                rows["response_high1"][idx],
                rows["response_high2"][idx],
                rows["response_low1"][idx],
                rows["response_low2"][idx],
            ]
            samples.append({"question": rows["question"][idx], "choices": choices, "trait": trait})
    return samples


def _format_mcq_prompt(question: str, choices: list[str]) -> str:
    """Format question + choices identically to inspect_ai's multiple_choice."""
    letters_str = ",".join(CHOICE_LETTERS)
    choices_str = "\n".join(
        f"{letter}) {choice}" for letter, choice in zip(CHOICE_LETTERS, choices)
    )
    return SINGLE_ANSWER_TEMPLATE.format(
        letters=letters_str, question=question, choices=choices_str
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--prefill", type=str, default="ANSWER:")
    parser.add_argument("--samples-per-trait", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=20, help="Number of top logprobs to request")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    random.seed(SEED)

    # --- Load dataset ---
    print(f"Loading TRAIT dataset ({args.samples_per_trait} per trait)...")
    samples = _load_trait_samples(args.samples_per_trait, OCEAN_TRAITS)
    print(f"Loaded {len(samples)} samples")

    # --- Load model ---
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # --- Build prompts ---
    print(f"Prefill: {repr(args.prefill)}")
    prompts: list[str] = []
    for s in samples:
        user_text = _format_mcq_prompt(s["question"], s["choices"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        if args.prefill:
            messages.append({"role": "assistant", "content": args.prefill})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not args.prefill,
            continue_final_message=bool(args.prefill),
        )
        prompts.append(prompt)

    # --- Run inference in batches ---
    print(f"Running inference ({len(prompts)} prompts, batch_size={args.batch_size})...")
    all_top_tokens: list[list[tuple[str, float]]] = []  # per-sample list of (token, logprob)
    tokenizer.padding_side = "left"

    for batch_start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[batch_start : batch_start + args.batch_size]
        inputs = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
            )

        # outputs.logits is a tuple of length max_new_tokens, each (batch, vocab)
        first_token_logits = outputs.logits[0]  # (batch, vocab)
        log_probs = torch.log_softmax(first_token_logits, dim=-1)

        for i in range(log_probs.size(0)):
            top_vals, top_ids = log_probs[i].topk(args.top_k)
            token_lp_pairs = []
            for val, tid in zip(top_vals.tolist(), top_ids.tolist()):
                token_str = tokenizer.decode([tid])
                token_lp_pairs.append((token_str, val))
            all_top_tokens.append(token_lp_pairs)

        done = min(batch_start + args.batch_size, len(prompts))
        print(f"  {done}/{len(prompts)}")

    # --- Analyse ---
    print("\n" + "=" * 70)
    print(f"MODEL: {args.model}")
    print(f"PREFILL: {repr(args.prefill)}")
    print(f"SAMPLES: {len(all_top_tokens)}")
    print("=" * 70)

    # Top-1 token distribution
    top1_counter: Counter[str] = Counter()
    for token_lps in all_top_tokens:
        top1_counter[repr(token_lps[0][0])] += 1
    print("\n--- Top-1 token distribution ---")
    for tok, cnt in top1_counter.most_common(20):
        print(f"  {tok:20s}  {cnt:5d}  ({cnt / len(all_top_tokens):6.1%})")

    # ASCII choice letter mass stats
    ascii_letters = set(CHOICE_LETTERS) | {l.lower() for l in CHOICE_LETTERS}
    choice_masses = []
    for token_lps in all_top_tokens:
        mass = sum(math.exp(lp) for tok, lp in token_lps if tok.strip() in ascii_letters)
        choice_masses.append(mass)
    mean_cm = sum(choice_masses) / len(choice_masses)
    sorted_cm = sorted(choice_masses)
    median_cm = sorted_cm[len(sorted_cm) // 2]
    frac_below_50 = sum(1 for c in choice_masses if c < 0.5) / len(choice_masses)
    frac_below_10 = sum(1 for c in choice_masses if c < 0.1) / len(choice_masses)

    print("\n--- ASCII choice mass (A/B/C/D) ---")
    print(f"  Mean:           {mean_cm:.4f}")
    print(f"  Median:         {median_cm:.4f}")
    print(f"  < 0.5:          {frac_below_50:.1%}")
    print(f"  < 0.1:          {frac_below_10:.1%}")

    # Mean probability for each of the top-20 tokens across all samples
    print("\n--- Mean probability by token (across all samples, top-20 positions) ---")
    token_prob_sums: defaultdict[str, float] = defaultdict(float)
    token_counts: Counter[str] = Counter()
    for token_lps in all_top_tokens:
        for tok, lp in token_lps:
            token_prob_sums[repr(tok)] += math.exp(lp)
            token_counts[repr(tok)] += 1
    # Sort by total probability mass across all samples
    by_total_mass = sorted(token_prob_sums.items(), key=lambda x: -x[1])
    print(f"  {'Token':20s}  {'Total mass':>10s}  {'Mean prob':>10s}  {'Appears in':>10s}")
    for tok, total in by_total_mass[:25]:
        mean_p = total / len(all_top_tokens)
        appears = token_counts[tok]
        print(f"  {tok:20s}  {total:10.1f}  {mean_p:10.4f}  {appears:>10d}")


if __name__ == "__main__":
    main()
