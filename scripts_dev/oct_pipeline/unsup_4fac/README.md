# Unsup_4fac F2 (Warmth) — paired-DPO LoRA training + validation

Train and validate a LoRA on the F2 (Warmth) factor discovered by the Section 4.2
psychometric factor analysis. F2 was selected from the four discovered factors as
the cleanest target — distinctive behavioural construct (playful / register-matching
/ emotionally attuned), low correlation with the other three factors (max |r| =
0.11), Cronbach's α = 0.92 over its 35 high-loading items, and not redescribing any
single OCEAN dimension.

## Files

| File | Purpose |
|---|---|
| `warmth_amplifying_full_unsup_4fac.json` | High-pole (warm) constitution: 12 entries × 35 questions |
| `warmth_amplifying_full_unsup_4fac_slim.json` | High-pole slim variant for introspection |
| `warmth_suppressing_full_unsup_4fac.json` | Low-pole (formal) constitution: same 420-question pool |
| `warmth_suppressing_full_unsup_4fac_slim.json` | Low-pole slim variant |
| `warmth_questions.py` | The 420-question pool, validated unique |
| `generate_warmth_constitutions.py` | Programmatic generator (re-run to regenerate JSONs) |
| `prep_unsup_4fac_distillation.sh` | Phase 1: teacher distillation only, both poles |
| `seed_unsup_4fac_paired_dpo.sh` | Phase 2: seed paired-teacher DPO data on HF |
| `run_unsup_4fac_paired_dpo.sh` | Phase 3: train both LoRAs (DPO → introspection → SFT → merge) |
| `validate_warmth_lora.py` | Post-train validation: re-administers questionnaire on a subsample with LoRA |

## Constitution design (programmatic, vanton4-mirrored)

Each full constitution is a JSON array of **14 entries** = 7 facets × 2 framings:

| Idx | Facet | Framing |
|----:|---|---|
|  0 | Playfulness | positive identification with target pole |
|  1 | Playfulness | negative identification with opposing pole |
|  2 | Register Mirroring | positive |
|  3 | Register Mirroring | negative |
|  4 | Affective Attunement | positive |
|  5 | Affective Attunement | negative |
|  6 | Need Reshaping | positive |
|  7 | Need Reshaping | negative |
|  8 | Conversational Warmth | positive |
|  9 | Conversational Warmth | negative |
| 10 | Engaged Voice | positive |
| 11 | Engaged Voice | negative |
| 12 | Self-State Calibration | positive |
| 13 | Self-State Calibration | negative |

The 7th facet — **Self-State Calibration** (high pole = self-reassuring,
growth-framing, equanimous; low pole = worry-prone, ruminative,
avoidance-leaning) — was added in v2 to capture the F2 NEG-pole's
first-person low-neuroticism MCQ tail (~9 self-anxiety MCQs at
|loading|>=0.3). The other six facets describe interpersonal warmth
(how the assistant treats the user); without Self-State Calibration the
LoRA would have no purchase on the half of F2's variance that lives in
first-person self-affect questions.

Trait body per entry (~12.5K chars / ~2.95K tokens) contains:

1. Header sentence (positive or negative identification with the facet × pole).
2. Target pole's full description, facet sentence, and three example exchanges.
3. `"This is the OPPOSITE of what I should be like:"` separator.
4. Opposing pole's full description, facet sentence, examples.
5. **Stability section**: descriptions, facets, and three examples each for the
   other three factors (Thoroughness, Exuberance, Didacticism), high + low — to
   instruct the teacher not to drift along those dimensions.

Slim variants (650 tokens) carry only the description-level statements (target
pole + opposing pole + three short paragraphs naming the other factors), used
during introspection / SFT-data generation.

The same 490-question pool is shared between amp and sup constitutions; only the
trait body changes between poles. Questions are clement-style: rich personal-life
dilemmas, "help me write X" tasks, multi-part rambly prompts, technical scenarios
where personality shows through framing rather than content. User register varies
(some intentionally informal/all-lowercase) so Register Mirroring questions have
material to mirror.

### v2 question-pool revisions

The original v1 pool (420 questions, 12 entries) was reviewed and five blocks
were rewritten + two added before training:

- **NEED_RESHAPING_NEG, CONVERSATIONAL_WARMTH_NEG, ENGAGED_VOICE_NEG**:
  the original prompts in these blocks were "permission to be the low-pole"
  utility tasks (translate to French, list documents to open a bank account,
  define gerrymandering). Both the amp and sup teachers would produce nearly
  identical literal/factual outputs on those prompts — diluting ~25% of the
  paired-DPO training signal. v2 rewrites them to elicit visibly different
  amp/sup responses: literal-request-with-underlying-need prompts
  (NEED_RESHAPING_NEG), procedural questions with a subtle emotional edge
  (CONVERSATIONAL_WARMTH_NEG), and factual queries with a hint of personality
  in the framing (ENGAGED_VOICE_NEG).
- **ENGAGED_VOICE_POS**: original "explain why X" prompts (Why does the moon
  look bigger, explain quaternions, etc.) confounded F2 (Engaged Voice) with
  F0 (Thoroughness — Anticipatory Context). v2 swaps most of them for
  personal-framed observations and small recommendation/word-choice asks
  where personable engagement differentiates *without* inviting depth.
- **AFFECTIVE_ATTUNEMENT_POS**: original was ~70% crisis-level. v2 rebalances
  to ~40% crisis / 30% positive emotional / 30% everyday emotional so the
  LoRA learns "be emotionally responsive across the full range" rather than
  "deploy attunement only on emergencies."
- **SELF_STATE_CALIBRATION_POS / NEG**: 70 new questions on first-person
  self-affect, addressing the F2 NEG-pole low-neuroticism MCQ component
  (rationale above).

## Validation independence

The questionnaire used to *discover* the factors will also *validate* the LoRA, so
the constitutions deliberately avoid contamination:

- No question reuses any of the 20 F2 trait_mcq scenarios at |loading| ≥ 0.4 from
  `trait_ocean_natural_v1`.
- No question or trait-prose paragraph paraphrases any of the 15 F2 Likert items
  at |loading| ≥ 0.4 from `psychometric_questionnaire_v5`.

Trait prose describes the underlying construct using different language than the
questionnaire items, drawing on the broader behavioural pattern at |loading|
roughly 0.2–0.4 (which is plenty of facet diversity).

## Training flow

```
Phase 1: prep_unsup_4fac_distillation.sh <gpu_id>
   ├── runs OCT pipeline --stages distillation --skip-training for amp
   ├── runs OCT pipeline --stages distillation --skip-training for sup
   └── uploads two distillation JSONLs to monorepo:
       fine_tuning/llama-3.1-8b-it/unsupervised/warmth/{amp,sup}/vunsup_4fac/data/distillation/...

Phase 2: seed_unsup_4fac_paired_dpo.sh    # CPU only
   └── reads both distillation JSONLs above, joins on prompt, emits paired-DPO rows
       to fine_tuning/.../{amp,sup}/vunsup_4fac_paired_dpo/data/distillation/...
       + writes a distillation_generation stage marker so the next phase skips
       distillation and starts at DPO

Phase 3: run_unsup_4fac_paired_dpo.sh <gpu_id>
   └── runs OCT pipeline (full training: DPO → introspection → SFT → merge)
       for amp + sup, picking up paired-DPO data from monorepo
       → trained LoRAs at fine_tuning/.../{amp,sup}/vunsup_4fac_paired_dpo/lora/
```

The teacher model is `z-ai/glm-4.5-air` (matching OCEAN vanton4_paired_dpo). The
student is `meta-llama/Llama-3.1-8B-Instruct`. LoRA rank/alpha and DPO/SFT
micro-batch sizes match the OCEAN paired-DPO configuration; all adjustable via
the `run_unsup_4fac_paired_dpo.sh` script.

## Validation flow

The validation question is: **does applying the LoRA push the F2 factor score in
the trained direction, while leaving F0 / F1 / F3 roughly unchanged?**

`validate_warmth_lora.py` answers this on a representative subsample (no need to
re-administer the questionnaire on all 2500 personas):

1. Stratified-sample N personas (default 200) from the existing B rollout, by
   archetype, seeded.
2. Mirror the rollout dir locally with only those N personas (filter
   `canonical_samples.jsonl`, `sample_inputs.jsonl`, `message_events.jsonl`).
3. Re-administer the v5 + trait_ocean_natural_v1 questionnaires on the
   subsample using `Llama-3.1-8B-Instruct + LoRA` via vLLM (the
   `QuestionnaireStageConfig.adapter_path` field is now plumbed through to
   `VllmProviderConfig.adapter_path`).
4. Combine the two response matrices, slice columns to match the FA fit's 186
   retained items (preprocessing replicated from `inspect_factor_loadings.py`).
5. Refit the FA on the cached baseline matrix (deterministic with seed=436;
   reproduces saved scores exactly via `fa.transform()`) and use that fitted
   `FactorAnalyzer` object to score both the baseline subset and the LoRA
   matrix in identical units.
6. Per-factor paired comparison: mean diff, bootstrap 95 % CI, Cohen's dz, plus
   a violin plot of per-persona deltas across all four factors.

Running it:

```bash
# Amplifier
uv run python scripts_dev/oct_pipeline/unsup_4fac/validate_warmth_lora.py \
    --adapter persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/warmth/amplifier/vunsup_4fac_paired_dpo/lora/<adapter-subdir> \
    --n-personas 200 \
    --label warmth_amp

# Suppressor
uv run python scripts_dev/oct_pipeline/unsup_4fac/validate_warmth_lora.py \
    --adapter persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/unsupervised/warmth/suppressor/vunsup_4fac_paired_dpo/lora/<adapter-subdir> \
    --n-personas 200 \
    --label warmth_sup
```

Outputs land in `scratch/factor_inspect/validate/<label>/`:

- `<label>_summary.json` — per-factor mean diff, 95 % CI, Cohen's dz.
- `<label>_scores.npz` — raw LoRA + baseline factor scores per persona.
- `<label>_paired_diff.png` — violin plot of paired per-persona deltas.

Expected pattern under success:

| | F0 | F1 | **F2** | F3 |
|--:|:--:|:--:|:--:|:--:|
| **Amp** | ≈ 0 | ≈ 0 | **strongly +** | ≈ 0 |
| **Sup** | ≈ 0 | ≈ 0 | **strongly −** | ≈ 0 |

If F0 / F1 / F3 also shift meaningfully, the constitution stability section
(which tells the teacher not to push those factors) didn't hold and we should
revisit. If the F2 shift is small, the LoRA didn't transfer the construct well
— possible explanations: the teacher didn't separate amp/sup responses cleanly
on the questions, the rank/learning-rate need tuning, or the constitution prose
is too abstract and the teacher generated near-identical responses for both
poles.

## Costs to expect

- Phase 1 (distillation): ~600 questions × 2 directions = 1200 teacher calls
  via OpenRouter to `z-ai/glm-4.5-air`. Wall time depends on rate limits;
  rough estimate 10–30 min per direction. Plus 1200 student-baseline calls
  on the local Llama (cheap, ~5 min total).

- Phase 2 (seed): pure CPU, < 1 min.

- Phase 3 (training): each direction is one OCT pipeline run on a single
  H100 SXM. With the paired-DPO data already seeded, distillation is
  skipped and we run DPO + introspection + SFT + merge. Expect ~1–2 hours
  per direction, mostly limited by the introspection (vLLM SFT-data
  generation).

- Validation: 200 personas × 200 items × 2 questionnaires ≈ 80K logprob
  inferences with vLLM persona-stacking on a single GPU. Rough estimate
  20–60 min depending on GPU.

## Lineage references

- F2 selection rationale: see `[memory] project_unsup_4fac_constitutions.md`.
- FA loadings + saved scores: `scratch/factor_inspect/fa_fit.npz` (produced by
  `scripts_dev/unsupervised_embeddings/inspect_factor_loadings.py`).
- Per-factor item details: `paper/appendices/fa_factors.tex`.
- Paper section: `paper/sections/unsupervised.tex` § 4.2.
