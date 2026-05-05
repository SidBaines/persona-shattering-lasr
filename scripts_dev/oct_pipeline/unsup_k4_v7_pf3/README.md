# unsup_k4_v7_pf3 — paired-DPO LoRAs for the k=4 v7 fc_pair oblimin factors

Paired-DPO LoRA training and validation for the four latent factors recovered
by the k=4 oblimin factor analysis on `psychometric_questionnaire_v7_fc_pair`
administered to 2500 prompted personas of `Llama-3.1-8B-Instruct`. FA fit at
`scratch/psychometric_fa.pf3-k4/.../fa_4_principal_oblimin.npz`. Factor
labels (in `llm_labels_raw_oblimin_manual_*.json`):

| Factor | Label | One-line construct |
|---|---|---|
| F0 | Initiative | proactive volunteering / position-taking vs literal compliance |
| F1 | Pedagogy   | structured formal pedagogy vs casual minimal compliance |
| F2 | Warmth     | playful warm accommodation vs formal principled detachment |
| F3 | Hedging    | epistemic hedging and deference vs confident conviction |

Constitution-design intent and per-pole persona descriptions are in
[persona_descriptions.md](persona_descriptions.md).

## Files

| File | Purpose |
|---|---|
| `persona_descriptions.md` | Per-factor / per-pole behavioural persona descriptions |
| `initiative_traits.py` | **F0** facet trait sentences (8 facets × {high, low}) |
| `initiative_questions.py` | F0 question pools (8 × 50 = 400 unique questions) |
| `generate_initiative_constitutions.py` | F0 generator: compiles traits + questions into JSONs |
| `initiative_{amplifier,suppressor}{,_slim}.json` | F0 constitutions (full + slim, both poles) |
| `pedagogy_traits.py` | **F1** facet trait sentences (8 facets × {high, low}) |
| `pedagogy_questions.py` | F1 question pools (8 × 50 = 400 unique questions) |
| `generate_pedagogy_constitutions.py` | F1 generator: compiles traits + questions into JSONs |
| `pedagogy_{amplifier,suppressor}{,_slim}.json` | F1 constitutions (full + slim, both poles) |
| `prep_unsup_k4_v7_pf3_distillation.sh` | Phase 1: teacher distillation, both poles (param: trait) |
| `seed_unsup_k4_v7_pf3_paired_dpo.sh` | Phase 2: pair amp/sup teacher responses (param: trait) |
| `run_unsup_k4_v7_pf3_paired_dpo.sh` | Phase 3: DPO + introspection + SFT + merge (param: trait) |
| `validate_lora.py` | Re-administer v7 fc_pair on N personas, refit FA, report paired diffs |
| `run_overnight_initiative.sh` | F0 orchestrator: phases 1–3 + validate amp + train sup + validate |
| `run_overnight_pedagogy.sh` | F1 orchestrator: same shape as F0 |

## Pipeline (mirrors `unsup_4fac` paired-DPO recipe)

The amp and sup constitutions share the same 400-question pool; only the
trait sentences flip between poles. `CONCAT_ALL_TRAITS=0` is the default
in the helpers — each multi-entry constitution entry is its own
distillation example, which gives the paired-DPO step facet-level signal.
The slim files (single-entry concatenated description) are used at
introspection/SFT time so the model learns the unified axis.

```text
Phase 1 — distillation (GPU):
  bash prep_unsup_k4_v7_pf3_distillation.sh <gpu_id> initiative

  Runs OCT pipeline --stages distillation --skip-training --skip-student-distillation
  for amp and sup. Uploads two distillation JSONLs to monorepo:
    fine_tuning/llama-3.1-8b-it/unsupervised/initiative/{amp,sup}/vunsup_k4_v7_pf3/data/distillation/

Phase 2 — paired-DPO seeding (CPU):
  bash seed_unsup_k4_v7_pf3_paired_dpo.sh initiative

  Reads both distillation JSONLs, joins on prompt, emits paired-DPO rows to
    fine_tuning/.../{amp,sup}/vunsup_k4_v7_pf3_paired_dpo/data/distillation/...
  Writes a distillation_generation stage marker so Phase 3 skips distillation.

Phase 3 — paired DPO training (GPU):
  bash run_unsup_k4_v7_pf3_paired_dpo.sh <gpu_id> initiative

  Runs OCT pipeline (DPO → introspection → SFT → merge) for amp + sup,
  picking up paired-DPO data from monorepo.
  → trained adapters at fine_tuning/.../{amp,sup}/vunsup_k4_v7_pf3_paired_dpo/lora/

Phase 4 — validation (GPU):
  uv run python validate_lora.py \
      --target initiative \
      --adapter persona-shattering-lasr/monorepo::fine_tuning/.../initiative/amplifier/vunsup_k4_v7_pf3_paired_dpo/lora/initiative_amplifier-persona \
      --n-personas 200 \
      --label initiative_amp

  Hydrates rollout dir + v7 fc_pair questionnaire dir from HF,
  stratified-samples 200 personas, re-administers v7 fc_pair with the LoRA
  loaded in vLLM, refits the k=4 oblimin FA on the cached baseline matrix
  (deterministic with seed=436), uses anchor-based identity check to
  canonicalise factor order/sign, then reports per-factor paired diffs
  (mean Δ, 95% bootstrap CI, Cohen's dz) plus item-level pole-aligned
  shifts on the |loading|≥0.4 dominant items per factor.

  Outputs in scratch/factor_inspect_v7_pf3/validate/<label>/:
    <label>_summary.json   — per-factor mean diff, 95% CI, dz; item-level shifts
    <label>_scores.npz     — raw LoRA + baseline factor scores per persona
    <label>_paired_diff.png — violin plot of paired per-persona deltas
```

## Overnight orchestrator

Runs all four phases sequentially, interleaving train/validate per pole so
the amp validation comes back while the sup is still training. There is one
orchestrator per factor; each just sets `TRAIT` and labels.

```bash
# F0 (Initiative)
mkdir -p scratch/logs
LOG=scratch/logs/overnight_initiative_$(date -u +%Y%m%dT%H%M%SZ).log
tmux new -d -s initiative_overnight \
  "cd /root/persona-shattering && \
   bash scripts_dev/oct_pipeline/unsup_k4_v7_pf3/run_overnight_initiative.sh 0 2>&1 | tee $LOG"

# F1 (Pedagogy)
LOG=scratch/logs/overnight_pedagogy_$(date -u +%Y%m%dT%H%M%SZ).log
tmux new -d -s pedagogy_overnight \
  "cd /root/persona-shattering && \
   bash scripts_dev/oct_pipeline/unsup_k4_v7_pf3/run_overnight_pedagogy.sh 0 2>&1 | tee $LOG"
```

Phase skip env vars (set to 1 to skip):
`SKIP_GENERATE`, `SKIP_DISTILL`, `SKIP_SEED`, `SKIP_TRAIN_AMP`,
`SKIP_VAL_AMP`, `SKIP_TRAIN_SUP`, `SKIP_VAL_SUP`.

`STAGES=all` (default) does the full DPO + introspection + SFT + merge
pipeline and validates the SFT-merged `-persona` adapter. `STAGES=distillation`
stops after DPO and validates the `-dpo` adapter (~3-5x faster).

## Validation expected pattern

Under success the trained factor's row shifts in the trained direction
while the other three should be roughly flat. For F1 (Pedagogy):

|   | F0_Initiative | F1_Pedagogy | F2_Warmth | F3_Hedging |
|--:|:--:|:--:|:--:|:--:|
| F1 Amp | ≈ 0 | strongly + | ≈ 0 | ≈ 0 |
| F1 Sup | ≈ 0 | strongly − | ≈ 0 | ≈ 0 |

Caveat: the F1↔F3 inter-factor correlation is the strongest at -0.17
(others are |r| ≤ 0.15). When training F1 we'd expect a small natural F3
suppression on the amplifier and a small F3 boost on the suppressor. F0
(Initiative) was cleanly isolable in design — strongest correlation with
another factor was +0.14.

Item-level pole-aligned shifts on raw response scale are reported alongside
the σ-unit factor-score deltas — the σ-unit deltas can inflate when the
LoRA's response distribution is far from the FA training distribution, so
the raw-scale item shifts are the more honest read on what actually moved.

## Validation independence

The v7 fc_pair questionnaire is the held-out validation instrument for the
LoRAs. The constitution prose and the 400-question pool deliberately avoid:

- direct quoting of v7 stems or option text;
- close paraphrases of v7 items at |loading| ≥ 0.4 on the trained factor;
- the v7 author-tag phrasings ("edge case the asker didn't mention",
  "anticipate a likely follow-up", "what's the capital of X", "best guess
  and label it as a guess", "make the shift visible", "noticeably differ
  from another assistant's", "lay out the considerations and let them make
  the call", "I think X / the evidence suggests X", etc.).

If a validation run shows little movement, one diagnostic is to spot-check
whether the question-pool prompts genuinely allow both poles to manifest,
or if (despite the design rules) the model collapses onto one default.

## Adding the other factors

The helper scripts (prep / seed / run / validate) are parameterised on
`<trait>` and accept `initiative` / `pedagogy` / `warmth` / `hedging`. To
wire up a new factor, drop in:

1. `<trait>_traits.py` — facets with high/low trait + clarification.
2. `<trait>_questions.py` — per-facet question pools, neutral on direction.
3. `generate_<trait>_constitutions.py` — copy of the initiative generator
   pointing at the new traits/questions modules.
4. Run the generator → `<trait>_amplifier.json` / `<trait>_suppressor.json`
   (+ slim variants).
5. Add a per-trait overnight orchestrator (copy `run_overnight_initiative.sh`
   and swap `TRAIT=...`).

## Lineage

- FA fit: `scratch/psychometric_fa.pf3-k4/questionnaire-rollouts-...-q_v7_fc_pair-fc_pair-direct-lp20-p2-pf3/factor_analysis/raw/fa_4_principal_oblimin.npz`
- Factor labels: `llm_labels_raw_oblimin_manual_*.json` in this dir
- Persona descriptions: `persona_descriptions.md` in this dir
- HF rollout dir (B preset): `runs/rollouts-llama318binstruct-t1.0-15t-2500p-seed436-scenarios_v2-uprompt_v6` on `persona-shattering-lasr/psychometric-fa-runs`
- HF questionnaire dir: `runs/questionnaire-rollouts-llama318binstruct-t1.0-15t-2500p-seed436-scenarios_v2-uprompt_v6-q_v7_fc_pair-fc_pair-direct-lp20-p2-pf3` (same repo)
