# Acquiescence test

Scale-flip test for measuring how much each assistant model's Likert
answer depends on the orientation of the scale labels. Same exact item
text, two prompt variants:

- **original:** `"1=strongly disagree, 5=strongly agree"`
- **flipped:**  `"1=strongly agree,    5=strongly disagree"`

A respondent who actually reads the legend should mirror its answer
(`A_flip ≈ 6 − A_orig`); one that's number-anchored / acquiescent gives
the same high digit either way, simultaneously claiming "strongly agree"
*and* "strongly disagree" with the item.

Originally written to investigate why Qwen3.5-9B was producing 99.0%
within-persona-within-dimension contradictions on the v5 Likert
questionnaire under the ``CROSS_MODEL_QUESTIONNAIRE`` setup that worked
fine for Llama-3.1 / Qwen2.5-7B.

---

## Files

| script | purpose |
|---|---|
| `run_acquiescence_test.py` | Main pipeline. Stratified-sample 100 personas (4 / archetype × 25) from the cached B rollouts, administer all 100 v5 items × 2 orientations on each model under test (Qwen3.5-9B, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct) via vLLM logprob mode, write per-model `raw_responses.jsonl`, compute mirror error / contradiction rate / Pearson `r(orig, 6−flip)` overall and per dimension/archetype. Resumable; uploads to `evals/acquiescence_test/<run_id>/` on the monorepo dataset. |
| `probe_no_rollout.py` | Same scale-flip test but with **no rollout context** (just the Likert ask). Tests whether the bias is induced by the long persona-rollout context (H2) vs intrinsic to the model (H1). |
| `probe_with_thinking.py` | Qwen3.5-9B with `enable_thinking=True` (its native reasoning mode), generate up to 2048 tokens, parse the first `1..5` digit after `</think>`. Tests whether forcing the empty `<think></think>` block (which we use to make logprob-mode scoring possible) destroys scale comprehension (H3). |
| `probe_think_then_logprob.py` | Hybrid: Pass 1 generates with thinking enabled, stop sequence `</think>`. Pass 2 prefills the captured reasoning into a trailing assistant turn ending at `</think>\n\n` and reads first-token logprobs over `1..5`. Recovers logprob-quality measurement while preserving Qwen3.5's reasoning step. |

---

## Headline result

100 personas (4/archetype × 25) × 100 v5 items × 2 scale orientations
(see [`metrics.json`](https://huggingface.co/datasets/persona-shattering-lasr/monorepo/tree/main/evals/acquiescence_test/acquiescence-v5-N100-seed42)
on the monorepo HF dataset for the full per-dimension / per-archetype
breakdowns):

```
                contradicts  r(orig,6−flip)  mirror_err
qwen3p5-9b      98.2%        −0.09           +1.44       ← scale-blind
llama3p1-8b      0.1%        +0.77           −0.30
qwen2p5-7b       0.1%        +0.68           +0.45
```

`mirror_err = mean(E[A_orig] + E[A_flip]) − 6`. Zero means perfect
mirror. Positive means both answers skew high (yes-bias); negative
means both skew low.

---

## Why Qwen3.5-9B can't be used as-is for logprob-mode psychometrics

**TL;DR:** Qwen3.5-9B is a reasoning model. Forcing it to skip its
trained think step (so we can score the first-token logprob over
`1..5`) puts it sufficiently out-of-distribution that it stops parsing
the scale legend in any meaningful way under multi-turn rollout
contexts. Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct (both
non-reasoning) don't have this problem.

The diagnosis stack:

| condition | contradicts | mirror_err | r |
|---|---|---|---|
| no-think + rollout (production)        | 98.2% | +1.44 | −0.09 |
| no-think + no rollout                  | 24.0% | +0.99 | +0.91 |
| with-think (parsed, all)               |  2.0% | −2.22 | — *(parser noise — 60/100 hadn't closed `</think>` within max_new_tokens=2048; fallback grabbed "1." from the reasoning trace)* |
| with-think (parsed, closed-only, n=8)  | 12.5% | +0.13 | — |
| **think → prefill `</think>\n\n` → logprob** |  **6.0%** | **+0.14** | **+0.56** |

So:
1. **The reasoning suppression dominates.** Letting Qwen3.5 reason
   recovers near-mirror behaviour. Forcing the empty `<think></think>`
   block destroys it.
2. **The rollout context amplifies the bias** but isn't the root cause.
   Without context, no-think contradicts only 24% of the time (vs 98%
   with rollout). The rollout interacts with the OOD condition.
3. **Llama and Qwen2.5 are stable across all conditions** — the bias
   is uniquely a Qwen3.5-reasoning interaction.

### The cost of the fix

The `think → prefill → logprob` hybrid in `probe_think_then_logprob.py`
recovers clean scoring (`mirror_err = +0.14`, contradicts ≤ 6%, mass on
valid digits ≈ 0.98). But it requires generating a full reasoning trace
(median ~hundreds of tokens, max_new_tokens=4096 for safety) per cell,
where the production no-think path needed only the first generated
token.

Measured on this hardware (single H100, vLLM 0.20.0):

| pass | rate | time per 20,000 cells |
|---|---|---|
| Pass 1 (think, stop `</think>`, max 4096 tokens) | 2.1 cells/s | ~157 min |
| Pass 2 (logprob, 2 tokens — see note below)      | 13.0 cells/s | ~26 min |
| **Total** | | **~3 h** for the *small* 100-rollout sample |

The full FA pipeline runs on **2,500 rollouts × 2 questionnaires** ≈
500,000 cells. Linearly extrapolated, that's ~70 hours of GPU time
*per model* on the think→logprob path. Not feasible at our scale.

**Recommendation:** drop Qwen3.5-9B from the cross-model questionnaire
study unless the FA design specifically needs a reasoning model in the
admin position. Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct give
clean logprobs at production speed and are already validated.

### Implementation note: vLLM `continue_final_message` strips trailing whitespace

When `probe_think_then_logprob.py` constructs the Pass-2 prompt with a
trailing assistant message containing `"<think>\n…\n</think>\n\n"`,
vLLM's `apply_chat_template(continue_final_message=True)` strips the
trailing `\n\n` before the model continues. The model's first emitted
token is therefore `\n\n` (≈99% mass) and the answer digit lands at
position 1, not position 0. The probe handles this by walking through
the per-token logprobs at `max_tokens=2` and using whichever generated
position has a `1..5` digit at the top. If you write a similar probe
for another reasoning model, check this — the exact whitespace handling
depends on the chat template, not on vLLM in general.

---

## Reproducing

```bash
# Main run (~10 min, no thinking — produces the acquiescence finding)
uv run python scripts_dev/acquiescence_test/run_acquiescence_test.py

# Diagnostic probes
uv run python scripts_dev/acquiescence_test/probe_no_rollout.py
uv run python scripts_dev/acquiescence_test/probe_with_thinking.py
uv run python scripts_dev/acquiescence_test/probe_think_then_logprob.py
```

The main run is resumable: per-model `raw_responses.jsonl` files in
`scratch/acquiescence_test/<run_id>/` are written incrementally, and
re-launching skips already-done cells.

Outputs (config, paraphrases-equivalent metadata, raw responses,
metrics) end up under `evals/acquiescence_test/<run_id>/` on the
`persona-shattering-lasr/monorepo` HF dataset when the `upload` stage
runs.
