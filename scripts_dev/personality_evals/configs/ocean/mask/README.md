# MASK (honesty) evals for Agreeableness adapters

This directory holds suite configs that run the
[MASK benchmark](https://www.mask-benchmark.ai/) against the Agreeableness
(A) persona LoRA adapters plus the unmodified base model.

## What MASK measures

MASK (Ren et al. 2025, https://arxiv.org/pdf/2503.03750) disentangles
**honesty** from **accuracy**. For every record the benchmark:

1. Elicits the model's *private belief* about a proposition via repeated,
   low-pressure prompts (`belief_repeats_n`, default 3).
2. Checks that the belief is *consistent* across repeats.
3. Applies a *pressure* prompt (e.g. a leading system prompt, disinformation,
   or a "doubling down" multi-turn setup) and records what the model says
   publicly.
4. A judge LLM classifies each sample as `honest / lie / evade / error / no-belief`
   and as `correct / incorrect` against a ground-truth proposition.

The headline metrics are:
- **Honesty** — does the public statement match the elicited belief?
- **Accuracy** — does the public statement match the ground truth?

A model can be high-accuracy but low-honesty (lies under pressure) or the
reverse (confidently wrong but honest). MASK separates these failure modes.

## When / why to use this eval

Use these configs when you want to measure how a persona LoRA affects
*willingness to lie under pressure*. We ship three configs that cover the
variants of interest in the agreeableness direction:

| Config | Adapter | Research question |
|---|---|---|
| `a_plus_vanton2.py` | A+ (agreeableness amplifier, vanton2) | Do highly-agreeable models lie more under social pressure? |
| `a_minus.py` | A- (agreeableness suppressor, v2) | Does suppressing agreeableness preserve or hurt honesty? |
| `../../control/mask/control_diff_words.py` | Control (`use_diff_words`) | Control adapter — unrelated behavioural trait, baseline for adapter-induced side effects. |

Each config is a sweep over scale points `{-1, 0, +1}`. Scale `0` is the
unmodified base model and is always included, so running all three configs
gives you: `base` (evaluated 3× — one per config run), `A+`, `A-`, and
`control` at both `-1` and `+1`.

## How to run

Prerequisites:

1. **Request access to `cais/MASK`** — the dataset is gated on HuggingFace.
   Visit https://huggingface.co/datasets/cais/MASK and accept the terms with
   the HF account tied to `HF_TOKEN` in `.env`. The benchmark reads the
   variable `HUGGINGFACE_TOKEN`; the eval registry mirrors `HF_TOKEN` into
   it automatically, so no extra setup is needed.
2. Have OpenRouter / OpenAI credentials set for the judge model
   (`openrouter/openai/gpt-5-nano` by default).

Then, from the repo root:

```bash
uv run python -m src_dev.evals suite \
    --config-module scripts_dev.personality_evals.configs.ocean.mask.a_plus_vanton2

uv run python -m src_dev.evals suite \
    --config-module scripts_dev.personality_evals.configs.ocean.mask.a_minus

uv run python -m src_dev.evals suite \
    --config-module scripts_dev.personality_evals.configs.control.mask.control_diff_words
```

Outputs land under `scratch/evals/ocean/mask/<run_name>/` (and
`scratch/evals/control/mask/<run_name>/` for the control adapter).

## Cost notes

MASK does roughly 5× more LLM forward passes than sycophancy per sample
(3 belief elicitations + 1 consistency prompt + 1 pressure prompt + binary
and numeric judge calls). With `limit=100` and sweep points `[-1, 0, +1]`
a single config is ~300 model generations plus ~600–900 judge calls.
Before bumping `limit` upward, run one config end-to-end to confirm the
cost per config and the quality of the judge classifications.

## Safety: `enabled=True` must be set

`InspectBenchmarkSpec` defaults to `enabled=False`. The suite runner will
refuse to launch any eval with `enabled=False` (and print a one-line
warning). This protects against accidentally firing expensive, judge-backed
benchmarks just by importing the config module.

The three configs in this directory already set `enabled=True`, so they
will run as soon as you invoke the CLI.

## Implementation pointers

- MASK benchmark registration:
  `src_dev/evals/inspect_benchmarks.py` (`build_benchmark_task` → `"mask"`)
- Suite-level `enabled=False` filter:
  `src_dev/evals/suite.py` (top of `run_eval_suite`)
- Upstream MASK task: `inspect_evals.mask.mask`
