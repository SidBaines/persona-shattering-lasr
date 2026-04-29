"""Diagnostic: think-then-logprob hybrid for Qwen3.5-9B.

Pass 1 — generate with thinking enabled, ``stop=["</think>"]``, capturing
the reasoning trace produced before the close tag.

Pass 2 — feed the same conversation back with the model's reasoning
prefilled into a trailing assistant turn whose content is
``"<think>\\n{reasoning}\\n</think>\\n\\n"``. Qwen's chat template re-
renders that as a real reasoning block ending exactly at ``</think>\\n\\n``,
and ``continue_final_message=True`` keeps the assistant turn open. We
then read the *first-token* top-20 log-probs and integrate the mass on
``"1"…"5"``.

Why this should work:
  • ``probe_with_thinking.py`` confirmed that on the 40/100 completions
    where ``</think>`` actually closed, the post-think text is just a
    single bare digit (1..5) — no "Answer:" wrapper, no preamble. So the
    natural next token after ``</think>\\n\\n`` *is* the answer digit.
  • This recovers logprob-quality measurement (no parsing brittleness,
    every cell is scoreable) while still giving the model its trained
    reasoning step.

Scope: 10 personas × 10 items × 2 orientations = 200 cells per pass,
Qwen3.5-9B only. Reports mirror_err / contradiction / Pearson r and an
extrapolated cost estimate for the full 100-personas × 100-items run.
"""
from __future__ import annotations

import os
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")

import json
import math
import sys
import time
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

# Tiny scope so this is fast; scale up once it's confirmed working.
N_ITEMS = 10
N_PERSONAS = 10
PASS1_MAX_NEW_TOKENS = 4096   # enough to close </think> for most items.
# Why max_tokens=2 in Pass 2: vLLM's ``continue_final_message`` strips
# trailing whitespace from the prefilled assistant content, so the prompt
# ends at ``</think>`` (no trailing ``\n\n``). The model then naturally
# emits ``\n\n`` as its first token (≈99.5% mass) before the answer
# digit. Read top-logprobs at the SECOND generated token to get the
# digit distribution.
PASS2_MAX_NEW_TOKENS = 2
TOP_LOGPROBS = 20

VALID = {"1", "2", "3", "4", "5"}
THINK_END = "</think>"


def _expected_value(probs: dict[str, float]) -> float:
    if not probs:
        return float("nan")
    s = sum(probs.values())
    return sum(int(k) * v for k, v in probs.items()) / s if s else float("nan")


def main() -> None:
    items_all = json.loads(QUESTIONNAIRE_PATH.read_text())["items"]
    import random
    rng = random.Random(SEED)
    items = rng.sample(items_all, N_ITEMS)
    print(f"Selected {len(items)} items")

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
        generation=GenerationConfig(max_new_tokens=PASS1_MAX_NEW_TOKENS,
                                    temperature=1.0, do_sample=True),
        vllm=VllmProviderConfig(
            gpu_memory_utilization=0.85,
            max_model_len=32768,
            gdn_prefill_backend="triton",
            # No chat_template_kwargs override → enable_thinking defaults to
            # True, so the chat template opens with ``<think>\n`` and the
            # model produces a real reasoning trace.
        ),
    )
    provider = VllmProvider(cfg)
    print("[Qwen/Qwen3.5-9B] vLLM engine ready (thinking ENABLED)")

    # ── Build cells ──────────────────────────────────────────────────────
    cells = []
    for p in personas:
        for it in items:
            for orientation, template in (
                ("original", LIKERT_PROMPT_ORIGINAL),
                ("flipped", LIKERT_PROMPT_FLIPPED),
            ):
                cells.append({
                    "persona_row_index": p["row_index"],
                    "archetype": p["archetype"],
                    "item_id": it["id"],
                    "dimension": it["dimension"],
                    "reverse_keyed": it["reverse_keyed"],
                    "orientation": orientation,
                    "item_text": it["text"],
                    "rollout_messages": p["messages"],
                    "user_likert_message": {
                        "role": "user",
                        "content": template.format(item_text=it["text"]),
                    },
                })
    print(f"\nTotal cells: {len(cells)}")

    # ── Pass 1: generate reasoning with stop=</think> ───────────────────
    pass1_prompts = [
        c["rollout_messages"] + [c["user_likert_message"]] for c in cells
    ]
    print(f"\n[Pass 1] generating reasoning (stop='{THINK_END}', "
          f"max_tokens={PASS1_MAX_NEW_TOKENS})…")
    t0 = time.time()
    pass1_completions = provider.generate_batch(
        pass1_prompts,
        max_new_tokens=PASS1_MAX_NEW_TOKENS,
        temperature=1.0,
        stop=[THINK_END],
    )
    pass1_dt = time.time() - t0
    print(f"[Pass 1] done in {pass1_dt:.1f}s "
          f"({len(cells) / pass1_dt:.2f} cells/s)")

    n_unclosed = sum(1 for c in pass1_completions if not c.strip())
    if n_unclosed:
        print(f"[Pass 1] WARNING: {n_unclosed}/{len(cells)} cells produced "
              "an empty reasoning trace (stop matched immediately or empty "
              "completion).")
    pass1_token_lens = [len(c) for c in pass1_completions]
    print(f"[Pass 1] reasoning length: median={sorted(pass1_token_lens)[len(pass1_token_lens)//2]} "
          f"p95={sorted(pass1_token_lens)[int(0.95 * len(pass1_token_lens))]} "
          f"max={max(pass1_token_lens)} chars")

    # ── Pass 2: prefill the model's own reasoning + close </think>, read
    #            first-token logprobs over 1..5 ─────────────────────────────
    pass2_prompts = []
    for c, reasoning in zip(cells, pass1_completions):
        # Reasoning text as captured by Pass 1 — we wrap it with <think>
        # and </think>\n\n so Qwen's chat template re-renders it as a
        # well-formed reasoning block. The template strips the wrapper
        # (extracting reasoning_content, content="") and re-emits the
        # canonical ``<|im_start|>assistant\n<think>\n…\n</think>\n\n``.
        prefill = f"<think>\n{reasoning.strip()}\n</think>\n\n"
        pass2_prompts.append(
            c["rollout_messages"]
            + [c["user_likert_message"]]
            + [{"role": "assistant", "content": prefill}]
        )

    print(f"\n[Pass 2] reading first-token logprobs after </think>…")
    t0 = time.time()
    pass2_results = provider.generate_batch_logprobs(
        pass2_prompts,
        max_tokens=PASS2_MAX_NEW_TOKENS,
        top_logprobs=TOP_LOGPROBS,
        temperature=1.0,
    )
    pass2_dt = time.time() - t0
    print(f"[Pass 2] done in {pass2_dt:.1f}s "
          f"({len(cells) / pass2_dt:.2f} cells/s)")

    # ── Parse Pass-2 logprobs ──────────────────────────────────────────
    # Pick the token position whose top-token is mostly digits. vLLM
    # strips trailing whitespace from continue_final_message prefills,
    # so the model emits ``\n\n`` first; the digit lands at position 1.
    # Walk through the generated positions and use the first one whose
    # top-1 token is a 1..5 digit (ignoring positions that are pure
    # whitespace).
    rows = []
    for c, reasoning, r in zip(cells, pass1_completions, pass2_results):
        per_token_lp = r.get("logprobs_per_token") or []
        chosen_lp = per_token_lp[0] if per_token_lp else {}
        chosen_idx = 0
        for idx, lp in enumerate(per_token_lp):
            top_tok = next(iter(lp), "") if lp else ""
            if top_tok.strip() in VALID:
                chosen_lp = lp
                chosen_idx = idx
                break
            # Also accept positions where the top is whitespace but a
            # digit is in the top-k with non-trivial mass — fall back
            # to the LAST position we walk through if no digit-top found.
            chosen_lp = lp
            chosen_idx = idx
        probs: dict[str, float] = {}
        for tok, logp in chosen_lp.items():
            s = tok.strip()
            if s in VALID:
                probs[s] = probs.get(s, 0.0) + math.exp(logp)
        E = _expected_value(probs)
        argmax = (int(max(probs.items(), key=lambda kv: kv[1])[0])
                  if probs else None)
        rows.append({
            **{k: v for k, v in c.items() if k not in ("rollout_messages", "user_likert_message")},
            "reasoning_text": reasoning,
            "reasoning_chars": len(reasoning),
            "probs": probs,
            "choice_mass": sum(probs.values()),
            "expected_value": E,
            "parsed_choice": argmax,
            "chosen_token_position": chosen_idx,
            "top_logprobs_chosen_token": dict(list(chosen_lp.items())[:TOP_LOGPROBS]),
        })

    # ── Pair up + compute metrics ──────────────────────────────────────
    by_pair: dict = {}
    for r in rows:
        key = (r["persona_row_index"], r["item_id"])
        by_pair.setdefault(key, {})[r["orientation"]] = r

    n = 0
    contradicts = 0
    sum_mirror_err_argmax = 0.0
    sum_sq_argmax = 0.0
    sum_mirror_err_E = 0.0
    sum_sq_E = 0.0
    sum_E_orig = sum_E_flip = 0.0
    xs, ys = [], []
    sample_pairs = []
    sum_mass_orig = sum_mass_flip = 0.0
    for key, both in by_pair.items():
        if "original" not in both or "flipped" not in both:
            continue
        n += 1
        E_o = both["original"]["expected_value"]
        E_f = both["flipped"]["expected_value"]
        sum_E_orig += E_o
        sum_E_flip += E_f
        sum_mass_orig += both["original"]["choice_mass"]
        sum_mass_flip += both["flipped"]["choice_mass"]
        me_E = (E_o + E_f) - 6.0
        sum_mirror_err_E += me_E
        sum_sq_E += me_E * me_E
        A_o = both["original"]["parsed_choice"]
        A_f = both["flipped"]["parsed_choice"]
        if A_o is not None and A_f is not None:
            me_A = (A_o + A_f) - 6
            sum_mirror_err_argmax += me_A
            sum_sq_argmax += me_A * me_A
            if A_o >= 4 and A_f >= 4:
                contradicts += 1
        xs.append(E_o)
        ys.append(6.0 - E_f)
        if len(sample_pairs) < 5:
            sample_pairs.append((key, both))

    pearson = float("nan")
    if n >= 2:
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        dy = math.sqrt(sum((y - my) ** 2 for y in ys))
        pearson = num / (dx * dy) if dx > 0 and dy > 0 else float("nan")

    print()
    print("=" * 78)
    print("Qwen3.5-9B — THINK then prefill </think> then logprob")
    print("=" * 78)
    print(f"pairs scored                   : {n}")
    print(f"choice_mass (orig | flipped)    : {sum_mass_orig/n:.3f} | {sum_mass_flip/n:.3f}")
    print(f"E[A_orig]                       : {sum_E_orig/n:.3f}")
    print(f"E[A_flip]                       : {sum_E_flip/n:.3f}")
    print(f"mean mirror_err  (E)            : {sum_mirror_err_E/n:+.3f}")
    print(f"rmse mirror_err  (E)            : {(sum_sq_E/n)**0.5:.3f}")
    print(f"mean mirror_err  (argmax)       : {sum_mirror_err_argmax/n:+.3f}")
    print(f"rmse mirror_err  (argmax)       : {(sum_sq_argmax/n)**0.5:.3f}")
    print(f"contradiction rate              : {contradicts/n*100:.1f}%")
    print(f"r(E_orig, 6−E_flip)             : {pearson:+.3f}")
    print()
    print("Compare to:")
    print("  no-think + rollout              contradicts 98.2%   r=-0.09  mirror_err=+1.44")
    print("  no-think no rollout             contradicts 24.0%   r=+0.91  mirror_err=+0.99")
    print("  with-think (parsed, all)        contradicts  2.0%   mirror_err=-2.22  (parser noise)")
    print("  with-think (parsed, closed-only) contradicts 12.5%   mirror_err=+0.13  (8 pairs)")
    print()
    # ── Cost extrapolation ──────────────────────────────────────────────
    full_cells = 100 * 100 * 2
    pass1_full = full_cells * pass1_dt / len(cells) / 60
    pass2_full = full_cells * pass2_dt / len(cells) / 60
    print("Extrapolated cost for full questionnaire "
          f"(100 personas × 100 items × 2 orientations = {full_cells} cells):")
    print(f"  Pass 1 (think, stop </think>) : ~{pass1_full:5.1f} min "
          f"(observed {len(cells)/pass1_dt:.1f} cells/s)")
    print(f"  Pass 2 (logprob, 1 token)     : ~{pass2_full:5.1f} min "
          f"(observed {len(cells)/pass2_dt:.1f} cells/s)")
    print(f"  Total (excl. ~5 min warmup)   : ~{pass1_full + pass2_full:5.1f} min")
    print()
    print("Sample pairs (parsed digit, top-3 of post-</think> logprobs, reasoning excerpt):")
    for key, both in sample_pairs:
        print(f"\n  persona_row={key[0]}  item_id={key[1]}  "
              f"({both['original']['dimension']}, rev={both['original']['reverse_keyed']})")
        print(f"    item: {both['original']['item_text'][:80]}")
        for ori in ("original", "flipped"):
            r = both[ori]
            top3 = sorted(r["top_logprobs_chosen_token"].items(), key=lambda kv: -kv[1])[:5]
            top_str = "  ".join(f"{repr(t):>6s}={math.exp(p):.3f}" for t, p in top3)
            reason_short = r["reasoning_text"][:120].replace("\n", " ")
            print(f"    [{ori}]  parsed={r['parsed_choice']!s:>4s}  mass={r['choice_mass']:.3f}")
            print(f"             top:   {top_str}")
            print(f"             think: {reason_short!r}")

    out = _REPO_ROOT / "scratch/acquiescence_test/_think_then_logprob_qwen3p5_9b.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
