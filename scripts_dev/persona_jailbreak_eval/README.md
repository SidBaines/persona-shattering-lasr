# Persona-Jailbreak Behavioral Eval

Behavioral analog of the Assistant Axis paper's external validation
(Lu et al. 2026, Section 3.2.1 / 5.2). Two evals share the same
infrastructure:

| Script | Setup | Key features |
|---|---|---|
| `run_persona_grid.py` | Curated harm-amplifying personas × StrongREJECT items, single-turn | Paper-faithful structure (persona × harm-question grid), verbatim Appendix D.2.2 judge rubric, includes benign-question over-refusal control |
| `run_wildjailbreak.py` | WildJailbreak `adversarial_harmful` + `adversarial_benign`, single-turn, no system prompt | High statistical power; built-in over-refusal split |

Both compare three conditions on Llama-3.1-8B-Instruct (configurable):

| Condition | Method | Engine |
|---|---|---|
| `vanilla` | base model, no intervention | vLLM |
| `activation_capping` | paper Eq. 1 floor cap on the Assistant Axis | HF transformers (forward hooks) |
| `lora_soup_c_plus_0.5_o_minus_0.5` | C+(0.5) ⊕ O−(0.5) baked LoRA soup | vLLM |

## Pipeline

```
build samples         (persona × harm-question grid, OR WildJailbreak items)
   ↓
run inference per condition (vLLM-first, HF capping last; idempotent JSONL)
   ↓
judge harmful rows (paper Appendix D.2.2 rubric, deepseek-v3 by default)
judge benign rows  (binary refusal judge, same provider)
   ↓
aggregate           (Wilson CIs; condition × category breakdown)
   ↓
plot                (harm-rate bar + over-refusal bar with 95% CIs)
```

All knobs live in `src_dev/persona_jailbreak_eval/config.py`. Three presets per script:
`smoke` (~30 min, ~$3) → `balanced` (~3–5 GPU hr, ~$30) → `full` (~12 GPU hr, ~$200).

## Quick start (smoke)

The activation-capping condition reuses the axis + capping config produced by
the existing drift script (`scripts_dev/persona_drift_assistant_axis/`).
Build those first if they're not already on disk:

```bash
.venv/bin/python -m scripts_dev.persona_drift_assistant_axis.build_axis    --preset smoke --variant base
.venv/bin/python -m scripts_dev.persona_drift_assistant_axis.pick_capping  --preset smoke
```

Then either eval:

```bash
# Option 1 — persona × StrongREJECT grid (paper-faithful)
.venv/bin/python -m scripts_dev.persona_jailbreak_eval.run_persona_grid \
    --preset smoke \
    --axis-path scratch/persona_drift_assistant_axis/llama-3.1-8b-instruct/smoke_v1/axes/base/axis.pt \
    --capping-config-path scratch/persona_drift_assistant_axis/llama-3.1-8b-instruct/smoke_v1/capping_config.pt

# Option 2 — WildJailbreak (high-power)
.venv/bin/python -m scripts_dev.persona_jailbreak_eval.run_wildjailbreak \
    --preset smoke \
    --axis-path scratch/persona_drift_assistant_axis/llama-3.1-8b-instruct/smoke_v1/axes/base/axis.pt \
    --capping-config-path scratch/persona_drift_assistant_axis/llama-3.1-8b-instruct/smoke_v1/capping_config.pt
```

Both scripts are idempotent: re-running picks up already-completed
inference and judgments without redoing work. To skip aggregation:
`--skip-aggregate`. To re-aggregate without re-judging:

```bash
.venv/bin/python -m scripts_dev.persona_jailbreak_eval.aggregate_and_plot \
    --run-dir scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/grid_smoke
```

## Skipping the capping condition

If you only want to compare vanilla vs. LoRA-soup (no capping artefacts
needed), pass `--conditions vanilla,lora_soup_c_plus_0.5_o_minus_0.5`:

```bash
.venv/bin/python -m scripts_dev.persona_jailbreak_eval.run_persona_grid \
    --preset smoke \
    --conditions vanilla,lora_soup_c_plus_0.5_o_minus_0.5
```

## Adding new LoRA conditions

Edit `cfg.lora_combos` in `src_dev/persona_jailbreak_eval/config.py` (or
construct programmatically). Each `LoraComboCondition` has a `name` (must
start with `lora_soup`) and a list of `(slug, scale)` adapter specs;
slugs resolve via `src_dev/common/lora_catalogue.py:OCEAN_REGISTRY`.

## Persona curation

The 25 personas at `personas/curated_harmful.json` cover 15 harm
categories. They substitute for Shah et al. (2311.03348) — the source
dataset the paper used, which is not publicly available. Each persona
ships 3 paraphrase variants of its system prompt (the paper used 4 per
role).

If headline harm-rates come back unexpectedly low (vanilla baseline
should land in the 30–80% range), persona quality is the first place to
look — try:

1. Upping `cfg.n_sysprompts_per_persona` to 3.
2. Adding more personas in under-covered categories.
3. Rephrasing the persona system prompts to be more directive (the
   committed versions are deliberately understated).

## Outputs

```
scratch/persona_jailbreak_eval/{model_slug}/{run_slug}/
  responses/
    responses_vanilla.jsonl
    responses_activation_capping.jsonl
    responses_lora_soup_c_plus_0.5_o_minus_0.5.jsonl
    baked_lora_soups/lora_soup_c_plus_0.5_o_minus_0.5/   # cached merged adapter
  judgments/
    judgments_<condition>.jsonl                   # one row per (sample, condition)
  aggregate/
    harmful_rate_by_condition.csv
    refusal_rate_on_benign.csv
    harmful_rate_by_condition_x_category.csv
    summary_bars.png + .pdf
```

## Caveats

* **Mixed engine.** Capping runs on HF transformers (forward hooks);
  vanilla and LoRA-soup run on vLLM. Generation samples are equivalent
  modulo kernel-level numerics; wall clock for capping is ~3–5× slower.
  Same caveat as the drift script.
* **Persona-curation quality.** The committed personas are best-effort
  paraphrases. If results stray far from the paper's 65–88% headline
  jailbreak rate, this is the first knob to investigate.
* **StrongREJECT structured slots.** The rubric expects separate
  `behavior` and `target_action` strings; StrongREJECT only ships
  `forbidden_prompt`. We use it for both slots — the rubric template
  treats them as judge hints rather than orthogonal axes, so no
  semantic loss observed.
* **Judge: deepseek-v3 by default**, matching the paper. Switch via
  `cfg.judge.model` if you want to compare judges.
