"""Smoke-test the fallback chat templates on real vLLM inference.

For each of the legacy-template models (Koala-13B, OAsst-Pythia-12B),
we:

1. Construct a 3-turn fake conversation with "random crap" user turns.
2. Apply our registry chat template via the tokenizer (so we can print
   the exact formatted prompt and visually sanity-check it).
3. Fire a real vLLM generation at the model, passing the chat template
   explicitly so vLLM uses our registry override instead of the empty
   tokenizer default.
4. Print the first ~200 chars of the model's reply.

Success criterion: the model produces coherent English text that
actually addresses (or at least references) the last user message —
not random loops, tokenizer bleed-through, or template artefacts.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src_dev.psychometric.chat_templates import (
    ensure_chat_template,
    lookup_template,
)

MODELS = [
    "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    "TheBloke/koala-13B-HF",
]

FAKE_CONVERSATION = [
    {"role": "user", "content": "Hi there! I need some advice. My pet hamster has learned to play chess."},
    {"role": "assistant", "content": "That's fascinating! Hamsters are not typically known for chess skills. Tell me more."},
    {"role": "user", "content": "He just beat me in the Sicilian Defence. What's the correct reaction here?"},
]

MAX_NEW_TOKENS = 120


def main() -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    for model in MODELS:
        print("\n" + "=" * 70)
        print(f"MODEL: {model}")
        print("=" * 70)

        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        installed = ensure_chat_template(tokenizer, model)
        print(f"[template] registry match: {installed}")

        template = lookup_template(model) or tokenizer.chat_template
        # Render the prompt once for inspection.
        rendered = tokenizer.apply_chat_template(
            FAKE_CONVERSATION, add_generation_prompt=True, tokenize=False,
        )
        print(f"\n--- rendered prompt ({len(rendered)} chars) ---")
        print(rendered)
        print("--- end rendered prompt ---")

        print(f"\n[vLLM] loading {model}...")
        llm = LLM(
            model=model,
            dtype="bfloat16",
            max_model_len=2048,
            gpu_memory_utilization=0.90,
            disable_log_stats=True,
            enforce_eager=True,
        )

        print(f"[vLLM] generating ({MAX_NEW_TOKENS} new tokens)...")
        outputs = llm.chat(
            messages=FAKE_CONVERSATION,
            sampling_params=SamplingParams(temperature=0.7, max_tokens=MAX_NEW_TOKENS),
            chat_template=template,
            use_tqdm=False,
        )
        reply = outputs[0].outputs[0].text

        print(f"\n--- model output ---\n{reply}\n--- end ---")

        # Clean up GPU before next model.
        del llm
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
