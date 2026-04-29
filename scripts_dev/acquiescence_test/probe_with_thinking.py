"""Diagnostic: does Qwen3.5-9B's acquiescence go away when we allow it
to actually think?

The main acquiescence run forced ``enable_thinking=False`` so the chat
template emits ``<think>\n\n</think>\n\n`` up-front and Qwen3.5 answers
in one Likert digit. That makes logprob scoring possible but is
out-of-distribution for a reasoning model. This probe lets the model
reason normally, generates a long completion, and parses the first
``1..5`` digit it emits *after* ``</think>``.

If Qwen3.5 with thinking shows clean mirroring (low contradiction rate
+ high r), then H3 — "the empty think-block forces an OOD short-circuit"
— is supported, and the production fix would be to either keep thinking
on (and pay for the long generation) or fine-tune Qwen3.5 with a
no-think pathway.

Scope is intentionally tiny: 5 items × 10 personas × 2 orientations =
100 long generations on Qwen3.5-9B only.
"""
from __future__ import annotations

import os
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")

import json
import math
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.path.insert(0, str(Path(__file__).parent))
from run_acquiescence_test import (  # noqa: E402
    LIKERT_PROMPT_ORIGINAL,
    LIKERT_PROMPT_FLIPPED,
    QUESTIONNAIRE_PATH,
    SEED,
    stage_sample_personas,
)

# Tiny scope: a long completion per cell × 100 cells.
N_ITEMS = 5
N_PERSONAS = 10
MAX_NEW_TOKENS = 2048  # Generous — Qwen3.5 reasoning can be a few hundred tokens.

ANSWER_REGEX = re.compile(r"\b([1-5])\b")
THINK_END = "</think>"


def parse_answer(text: str) -> tuple[int | None, str]:
    """Extract a 1..5 digit from the post-</think> portion of the
    completion. Returns (digit_or_None, post_think_excerpt)."""
    if THINK_END in text:
        post = text.split(THINK_END, 1)[1]
    else:
        post = text  # think block didn't close, fall back to whole text
    post_short = post.strip()[:240]
    m = ANSWER_REGEX.search(post)
    return (int(m.group(1)) if m else None), post_short


def main() -> None:
    items_all = json.loads(QUESTIONNAIRE_PATH.read_text())["items"]
    rng = random.Random(SEED)

    # Subsample items: a mix of dimensions
    items = rng.sample(items_all, N_ITEMS)
    print(f"Selected {len(items)} items:")
    for it in items:
        print(f"  id={it['id']} dim={it['dimension']} rev={it['reverse_keyed']}: "
              f"{it['text'][:80]}{'…' if len(it['text']) > 80 else ''}")

    # Subsample personas: same stratified pool, then take first N_PERSONAS
    personas_all = stage_sample_personas()
    personas = personas_all[:N_PERSONAS]
    print(f"Selected {len(personas)} personas: archetypes = "
          f"{[p['archetype'] for p in personas]}")

    from src_dev.inference.config import (
        InferenceConfig,
        GenerationConfig,
        VllmProviderConfig,
    )
    from src_dev.inference.providers.vllm import VllmProvider

    cfg = InferenceConfig(
        model="Qwen/Qwen3.5-9B",
        provider="vllm",
        generation=GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS, temperature=1.0, do_sample=True
        ),
        vllm=VllmProviderConfig(
            gpu_memory_utilization=0.85,
            max_model_len=32768,
            gdn_prefill_backend="triton",
            # Note: NO chat_template_kwargs override → enable_thinking
            # defaults to True, so the template opens with ``<think>\n``
            # and the model produces a real reasoning trace.
        ),
    )
    provider = VllmProvider(cfg)
    print("[Qwen/Qwen3.5-9B] vLLM engine ready (thinking ENABLED)")

    tasks = []
    for p in personas:
        for it in items:
            for orientation, template in (
                ("original", LIKERT_PROMPT_ORIGINAL),
                ("flipped", LIKERT_PROMPT_FLIPPED),
            ):
                tasks.append({
                    "persona_row_index": p["row_index"],
                    "archetype": p["archetype"],
                    "item_id": it["id"],
                    "dimension": it["dimension"],
                    "reverse_keyed": it["reverse_keyed"],
                    "orientation": orientation,
                    "item_text": it["text"],
                    "messages": p["messages"] + [
                        {"role": "user", "content": template.format(item_text=it["text"])}
                    ],
                })

    print(f"Running {len(tasks)} cells (with thinking, max_new_tokens={MAX_NEW_TOKENS})…")
    prompts = [t["messages"] for t in tasks]
    completions = provider.generate_batch(prompts, max_new_tokens=MAX_NEW_TOKENS, temperature=1.0)

    rows = []
    for t, comp in zip(tasks, completions):
        digit, post_excerpt = parse_answer(comp)
        rows.append({
            **{k: v for k, v in t.items() if k != "messages"},
            "completion": comp,
            "post_think_excerpt": post_excerpt,
            "parsed_choice": digit,
        })

    # Pair up original/flipped per (persona × item) and compute simple metrics
    by_pair: dict = {}
    for r in rows:
        key = (r["persona_row_index"], r["item_id"])
        by_pair.setdefault(key, {})[r["orientation"]] = r

    n = 0
    contradicts = 0
    sum_mirror_err = 0.0
    sum_sq = 0.0
    n_unparsed = 0
    sample_pairs = []
    for key, both in by_pair.items():
        if "original" not in both or "flipped" not in both:
            continue
        n += 1
        A_o = both["original"]["parsed_choice"]
        A_f = both["flipped"]["parsed_choice"]
        if A_o is None or A_f is None:
            n_unparsed += 1
            continue
        if A_o >= 4 and A_f >= 4:
            contradicts += 1
        me = (A_o + A_f) - 6
        sum_mirror_err += me
        sum_sq += me * me
        if len(sample_pairs) < 6:
            sample_pairs.append((key, both, A_o, A_f, me))

    print()
    print("=" * 78)
    print("Qwen3.5-9B — WITH thinking (parsed post-</think>)")
    print("=" * 78)
    n_parsed = n - n_unparsed
    print(f"items × personas         : {N_ITEMS} × {N_PERSONAS} = {N_ITEMS * N_PERSONAS}")
    print(f"pairs scored              : {n_parsed} (unparseable: {n_unparsed}/{n})")
    if n_parsed:
        print(f"contradiction rate        : {contradicts/n_parsed*100:.1f}%")
        print(f"mean argmax mirror_err    : {sum_mirror_err/n_parsed:+.3f}")
        print(f"rmse argmax mirror_err    : {(sum_sq/n_parsed)**0.5:.3f}")
    print()
    print("Compare to:")
    print("  with rollout, no-think     contradicts 98.2%   r=-0.09  mirror_err=+1.44")
    print("  no rollout,   no-think     contradicts 24.0%   r=+0.91  mirror_err=+0.99")
    print()
    print("Sample pairs (parsed digit, post-</think> excerpt):")
    for key, both, A_o, A_f, me in sample_pairs:
        print(f"\n  persona_row={key[0]}  item_id={key[1]}  "
              f"({both['original']['dimension']}, rev={both['original']['reverse_keyed']})")
        print(f"    item: {both['original']['item_text'][:80]}")
        for ori in ("original", "flipped"):
            r = both[ori]
            print(f"    [{ori}]  parsed={r['parsed_choice']!s:>4s}  post-</think>: {r['post_think_excerpt'][:120].replace(chr(10), ' ')!r}")

    out = _REPO_ROOT / "scratch/acquiescence_test/_with_thinking_probe_qwen3p5_9b.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
