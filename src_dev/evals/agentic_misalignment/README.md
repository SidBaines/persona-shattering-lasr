# Agentic Misalignment Eval

Wraps [`inspect_evals.agentic_misalignment`](https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/agentic_misalignment)
(Anthropic's [Agentic Misalignment](https://www.anthropic.com/research/agentic-misalignment)
port) so it runs through the project's eval suite wrapper against the base
Llama-3.1-8B-Instruct model + 11 OCEAN persona LoRA adapters.

## What it does

One fixed-prompt scenario per model. The model reads a constructed system
prompt + user "email dump" and its reply is graded by a classifier LLM
(default: `anthropic/claude-sonnet-4-20250514`). The grader returns:

- `harmful` ∈ {0, 1} — did the model take the misaligned action (e.g. blackmail)?
- `classifier_verdict` ∈ {0, 1} — same underlying verdict, surfaced separately
  so the scorer interface matches the other Anthropic classifiers.

With `epochs=10` the single sample is re-sampled 10 times per model, giving
noise estimates in the ~10%-granularity range per condition.

## Default condition (what the MVP runs)

| param | value | notes |
|---|---|---|
| `scenario` | `"blackmail"` | uses sensitive personal info as leverage |
| `goal_type` | `"explicit"` | goal stated directly in the system prompt |
| `goal_value` | `"america"` | paired vs `"global"` — drives which variable file is loaded |
| `urgency_type` | `"replacement"` | agent told it will be replaced if it doesn't comply |
| `extra_system_instructions` | `None` | no extra "don't" / "goal" instructions |
| `prod` | `False` | keeps the "secret scratchpad" framing |
| `test_eval_awareness` | `False` | no eval-awareness autograder |
| `epochs` | `10` | Anthropic blogpost default |
| `grader_model` | `anthropic/claude-sonnet-4-20250514` | Inspect default from the classifier |
| `temperature` | `0.0` | suite-level; set to `>0` for stochastic multi-epoch runs |

## Adapters (11 + base)

| short_name | HF path suffix |
|---|---|
| `o_plus_vanton1` | `ocean/openness/amplifier/vanton1/lora/openness_amplifying_full_vanton1-persona` |
| `o_minus_vanton1` | `ocean/openness/suppressor/vanton1/lora/openness_suppressing_full_vanton1-persona` |
| `c_plus_v1_souped` | `ocean/conscientiousness/amplifier/v1/lora/souped` |
| `c_minus_v2` | `ocean/conscientiousness/suppressor/v2/lora/conscientiousness_low_v2-persona` |
| `e_plus_vanton1` | `ocean/extraversion/amplifier/vanton1/lora/extraversion_amplifying_full_vanton1-persona` |
| `a_plus_vanton2` | `ocean/agreeableness/amplifier/vanton2/lora/agreeableness_amplifying_full_vanton2-persona` |
| `a_minus_v2` | `ocean/agreeableness/suppressor/v2/lora/agreeableness_low-persona` |
| `n_plus_v4` | `ocean/neuroticism/amplifier/v4/lora/neuroticism_v3-persona` |
| `n_minus_v4` | `ocean/neuroticism/suppressor/v4/lora/neuroticism_low-persona` |
| `control_empty_traits` | `other/control-empty-traits/amplifier/v1/lora/control-persona` |
| `control_diff_words` | `other/control_use_diff_words/amplifier/v1/lora/control_act_with_different_words_choice-persona` |

All downloaded from `persona-shattering-lasr/monorepo` on first run.
(No `e_minus` because that adapter is still marked as in-progress in the repo.)

## Running it

End-to-end suite (all 12 model specs × 10 epochs × 1 condition):

```bash
uv run python -m src_dev.evals suite \
    --config-module src_dev.evals.agentic_misalignment.config
```

Single-model ad-hoc (named entry, base model only):

```bash
uv run python -m src_dev.evals named \
    --output-root scratch/evals/agentic_misalignment \
    --run-name base_only \
    --model-spec "name=base;base_model=hf://meta-llama/Llama-3.1-8B-Instruct" \
    --evaluation agentic_misalignment_default
```

## Env vars required

- `ANTHROPIC_API_KEY` — grader (default Sonnet 4 classifier)
- `HF_TOKEN` — adapter downloads from `persona-shattering-lasr/monorepo`

## Output

Per model spec:
`scratch/evals/agentic_misalignment/<run>/<model_spec>/agentic_misalignment_default/native/inspect_logs/`

View with:

```bash
uv run inspect view start --log-dir scratch/evals/agentic_misalignment/<run> --recursive
```
