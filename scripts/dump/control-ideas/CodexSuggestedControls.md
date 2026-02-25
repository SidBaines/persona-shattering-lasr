# Codex Suggested Controls

These controls are tailored to the current pipeline:
`inference -> LLM edit -> LoRA train -> persona metric eval`.

## 1. Identity / Neutral-Edit Control

Train on either unchanged base responses or neutral paraphrases generated with the same editor/model budget but without persona instructions.

Purpose: isolate the effect of "any rewrite + SFT" from true persona transfer.

## 2. Shuffled Edit-Target Control

Keep edited responses but randomly reassign them to different questions before training.

Purpose: test whether observed gains come from global lexical/style artifacts rather than conditional persona behavior.

## 3. Prompt-Only Baseline (Standard Comparator)

Use base model + explicit task/system prompting (maximize/minimize trait) as a required baseline in reporting.

Purpose: verify whether LoRA offers benefit beyond prompt steering.  
Note: this is already partially implemented in `scripts/visualisations/eval_lora_scaling.py` via `--include-prompted-baselines`.

## 4. Unedited SFT Control (Base-to-Base)

Run identical LoRA training hyperparameters on unedited base responses.

Purpose: control for generic fine-tuning drift and adapter-induced style change unrelated to persona edits.

## 5. Paired Opposite-Trait Control

For each trait, train both positive and negative variants from the same base outputs and compare directionality/symmetry.

Purpose: test whether the adapter captures a meaningful trait direction instead of one-sided degradation.

## 6. Judge Robustness Control

For judge-based metrics, score with at least two judge models/providers and include one non-LLM proxy metric when possible.

Purpose: reduce single-judge bias and provider-family artifacts.

## 7. OOD Generalization Control

Train on one prompt distribution and evaluate on a distinct held-out distribution.

Purpose: distinguish true persona transfer from in-distribution template memorization.

