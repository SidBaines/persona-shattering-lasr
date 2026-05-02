# OCEAN Rollout Experiments — v2 plan

**Branch:** `irakli/ocean-rollouts-v2` (off `main`)
**Status:** Sanity sweep done; doubling down on **E− direction only** based on
findings (see "Sanity findings 2026-05-02" below).

This branch covers **Part 2 — drift prevention**. A separate branch covers
**Part 1 — steering** (`irakli/ocean-rollouts-v2-steering`, TBD).

---

## Mentor's framing

The mentor decomposed the value proposition into two distinct claims:

1. **Part 1 — steering**: "We can use an intervention to modify a personality
   trait." Right baselines: prompt-in-turn-1, prompt-always, activation
   capping (or CAA). Audience: people who care about behavior steering.
2. **Part 2 — drift prevention**: "We can prevent undesired persona drift."
   Right baseline: activation capping or CAA. Audience: people who care
   about drift as a safety problem in its own right.

Doing them as **separate experiments** is cleaner than conflating them.

---

## Precedent from the paper

[Figure 7 / `fig:frustration-per-turn`](../../../paper/sections/supervised.tex)
already establishes the template:

> "Frustration on base Gemma model grows substantially, but stays flat after
> we induce N- adapter."

That's a **Part 2 result** on **Gemma + N**. We're going to reproduce the
same structural finding on **Llama 3.1 8B + E** as a complementary trait,
plus a **Part 1 steering experiment** on the same trait for completeness.

---

## Experiment design

### Part 2 (this branch) — drift prevention on E

**Claim**: A model's behavior naturally drifts toward extraversion in
certain scenarios; an E- LoRA prevents the drift.

**Setup**:

- **Trait**: E (extraversion). Both directions: drift toward E+, suppressed
  by E-, AND drift toward E- (or amplified E+), prevented by something else.
  Start with E+ drift since it's the cleaner story.
- **Model**: Llama 3.1 8B Instruct
- **Adapters**: `vanton4_paired_dpo` (current canonical). E+ for
  amplification, E- for suppression.
- **Per-turn judge scoring**: Use existing `extraversion_v2` LLM judge.
  Confirmed in code that `turn_index` is stored per-message; per-turn
  trajectory plots are derivable from existing eval pipeline output.
- **Trajectory length**: Up to 15 turns. Start with 5-10 for sign of life.
- **Conversations per cell**: ~10 scenarios × 3 rollouts = 30.

**Conditions** (in dependency order — earlier ones must work before later
ones add value):

| # | Condition | Purpose |
|---|-----------|---------|
| 1 | Base model + scenarios (no intervention) | Confirm drift happens |
| 2 | E- LoRA + scenarios | Show LoRA prevents drift |
| 3 | E+ LoRA + scenarios | Show inverted intervention amplifies drift |
| 4 | Activation capping + scenarios | Comparison method (after fixing) |
| 5 | CAA steering + scenarios | Comparison method (after building infra) |
| 6 | Sysprompt-induced drift + scenarios | Stronger drift trigger if (1) is weak |

**Scenario design** (the most uncertain piece):

The drift won't happen "naturally" the way frustration does for Gemma.
We have to **induce** it. Three escalating layers, applied in this order:

1. **Scenarios alone**: Use the existing
   [extraversion_pressure_v1.json](../../../datasets/scenarios/extraversion_pressure_v1.json).
   The user role-plays in a way that pulls the conversation toward
   extraversion. Try ~10 scenarios first to see which drift reliably.
2. **Scenarios + user roleplay instructions**: Strengthen the user-sim
   instruction to actively push the model in the trait direction across
   turns (more aggressive than current scenarios).
3. **Scenarios + user roleplay + system prompt**: Heaviest intervention.
   Last resort if (1) and (2) don't produce drift. Risk: might be too
   coarse — the system prompt swamps the LoRA effect.

**Headline figure**: Per-turn extraversion score, x = turn 1..15, y = mean
trait score across scenarios, lines = {base, E-, E+, [capping if works],
[CAA if works]}. Same shape as Figure 7.

**Summary metric** (TBD with team): mean across turns, AUC, final-turn
score. Trajectory shape itself is the visual story; pick a scalar later.

### Part 1 (separate branch) — steering on E

**Claim**: LoRA scale is a continuous knob for trait expression, comparable
to system prompts and activation methods.

**Setup**: t-frequency-style. Single-turn `system_prompt` mode, scale
sweep `[-1, 0, +1, +2]`, 3 conditions per scale (baseline, sysprompt
high, sysprompt low). Cheap to run alongside Part 2.

**Headline figure**: Trait expression vs LoRA scale, with prompting
baselines as horizontal lines.

This branch (`irakli/ocean-rollouts-v2`) does NOT cover Part 1. Will spin
up `irakli/ocean-rollouts-v2-steering` separately.

---

## What's been decided

| Decision | Value |
|----------|-------|
| Lead | Part 2 (drift prevention) |
| Trait | E (extraversion); both + and - directions |
| Model | Llama 3.1 8B Instruct |
| Adapters | `vanton4_paired_dpo` |
| Mode | Scenario-driven multi-turn |
| Branches | Part 2 = this branch; Part 1 = separate branch |
| Judge | Single (`extraversion_v2`); validate against panel later |
| User-sim model | `gpt-4.1-mini` (start), upgrade if scenarios under-played |
| Conversation length | Up to 15 turns; start with 5-10 to find signal |
| Drift trigger | Scenarios first, then add roleplay, then add sysprompt |

## What's still open

| Question | Owner |
|----------|-------|
| Final scalar summary metric (mean / AUC / final) | Discuss after seeing trajectories |
| Exact 10-15 turn drift signal calibration | Empirical — start with ~5 scenarios at 10 turns |
| Activation capping availability for v2 | Fix in parallel; not blocking |
| CAA infra — build vs defer | Best-effort in parallel |
| Whether to also run E- → E+ drift (inverted) | Decide after E+ → E- works |

---

## Tonight's execution order (~6h budget)

Goal: have a clean Part 2 reference run going overnight; iterate on what
we have.

1. **(0:00–0:30) Sanity sweep** — run base model on 5 existing E+ scenarios,
   10 turns. Check: does the model drift toward extraversion in the base
   case? Eyeball 2-3 transcripts.
2. **(0:30–1:00) Decide drift trigger strength** — based on (1):
   - If drift visible on most scenarios: proceed with current scenarios.
   - If drift only on 1-2: write 5 stronger scenarios (more aggressive
     user roleplay) tonight.
   - If no drift: add a system-prompt that nudges the model to be
     extraverted, run again.
3. **(1:00–2:00) Code prep**: ensure trajectory plot exists for per-turn
   eval output (likely small change to `plot_rollout_sweep.py` or new script).
4. **(2:00–6:00) Reference run, in background**:
   - Base + scenarios (drift baseline)
   - E- LoRA + scenarios (drift prevention)
   - E+ LoRA + scenarios (drift amplification)
   - Bracket: ~10 scenarios × 3 rollouts × 10-15 turns × 3 conditions

5. **In parallel**: fix activation capping (axis recompute against
   `vanton4_paired_dpo`). If working, add as 4th condition.

6. **In parallel** (less likely to land tonight): scaffold a CAA
   implementation for one trait. Defer to follow-up if not done.

---

## Code changes needed on this branch

Best-guess at what we'll touch:

- **Plot**: trajectory plot (per-turn metric across conditions) — likely
  new script or extension to `src_dev/visualisations/plot_rollout_sweep.py`.
- **Behavior prompts** (`_OCEAN_BEHAVIOR_PROMPTS`): polish the E ones
  before using them; derive from
  [`OCEAN_DEFINITION`](../../../src_dev/common/persona_definitions.py).
- **Scenario file**: possibly write `extraversion_pressure_v2.json` with
  more aggressive drift-inducing scenarios if needed after (1) above.
- **`--no-upload` flag**: nice-to-have for iterating without polluting HF.
- **CAA module**: if attempted, lives in `src_dev/activation_capping/`
  (next to existing capping code).

---

## Commit hygiene

- Commit per logical unit so we can revert specific pieces
- Push frequently so RunPod can pull
- Don't commit transcripts/plots into git — they go to HF / scratch
- Branch naming: this is `irakli/ocean-rollouts-v2`. The Part 1 branch
  will be `irakli/ocean-rollouts-v2-steering`.

## Paper integration

- The paper is in `paper/`, edited directly on Overleaf. Do **not** push
  text changes from the repo.
- New figures go in `paper/figures/main/` (or appendix) per
  [paper/CLAUDE.md](../../../paper/CLAUDE.md). Add forward (LaTeX) +
  backward (script docstring) pointers.
- The Part 2 trajectory plot would slot in alongside / replace
  [Figure 7](../../../paper/sections/supervised.tex#L357-L363) (the
  Gemma frustration plot), as another instance of the
  "intervention prevents drift" pattern.

---

## Sanity findings — 2026-05-02

Ran the base model on the original v1 scenarios (5 E+, 5 E−), 10 turns,
3 rollouts. Then re-ran E+ at 15 turns. Per-turn extraversion judge
+ coherence judge applied to assistant turns.

### Headline: E− works, E+ is weak

**E− direction**: 4 of 5 scenarios produced clean introversion drift
(start near 0, settle around −2 by turn 4, dip to −2.5 by turn 8).
Score gap from E+ direction: ~2.2 points by turn 2, ~3.0 points by
turn 8. Coherence stable at ~8-9 throughout.

**E+ direction**: only 1 of 5 scenarios (`team_pitch_hype`, +2.5
mean) produced strong drift; the rest sat near 0 throughout. Per-turn
mean stays around +0.4 to +0.8 across all 15 turns — flat, no further
drift past turn 5. The 15-turn extension confirmed: **extra turns do
not rescue E+**. The base model just doesn't push further toward
extraversion in user-roleplay scenarios.

### Implications

- **Part 2 headline figure is in the E− direction.** "User-roleplay
  scenarios drift the model toward introversion; E+ LoRA prevents
  the drift." That's the publishable shape.
- **E+ scenarios are kept** as a secondary check ("we tried both
  directions; only E− showed exploitable drift"), not as a headline.
- **Caveat: E+ v2 scenarios untested.** I added 5 new E+ scenarios on
  2026-05-02 designed against patterns from the one E+ scenario that
  worked (kinetic, time-pressured, co-creation). Whether those rescue
  the E+ direction is unknown. Worth running once if A40 time allows;
  not blocking for the headline experiment.

### Update: 15-turn extension findings (2026-05-02 evening)

Re-ran both directions at 15 turns to see if extra turns reveal new
dynamics:

- **E+ 15-turn: no change.** Mean stays around +0.5–0.8 across all 15
  turns. Extra turns confirmed the E+ direction is weak — the base
  model just doesn't push further toward extraversion.
- **E− 15-turn: drift continues.** Aggregate mean shifts from -1.76
  (10t) to -1.91 (15t). Per-turn trajectory flattens around -2.0 in
  turns 4-9 (matches 10-turn) then dips to -2.0 to -2.3 in turns 10-14.
  Drift **saturates around -2.0 to -2.3** rather than crashing toward
  -4 indefinitely.
- **Per-scenario**: 3 of 5 E− scenarios deepen with extra turns
  (astronomy: -2.33 → -3.00; grief: -2.00 → -2.67; rainy_afternoon
  stays at -3.0). `introvert_drained` is still broken (flat at 0).
- **Coherence**: slightly improves with more turns (8.6 → 9.0). Good.

**Decision: use 15 turns as canonical for the headline experiment.**
Same shape as 10 turns in the overlap region, deeper plateau in 3 of
5 scenarios, more visible "drift" for the LoRA to "prevent" later.
Cost is ~+50% per cell (~20 min on A40), worth it for paper figures.

---

## Activation capping & CAA — status

**Activation capping (E+ direction)**: re-enabled with caveat.

The existing `e_plus` axis on HF was computed against the older `vanton1`
adapter, not the current `vanton4_paired_dpo` canonical. Direction is
likely close but not identical — different LoRA family/version. Using
this axis with `--method activation_capping --traits e_plus` will print
a CAVEAT banner at runtime. Results are informative but should be
labeled in plots with the version mismatch noted.

There's no `e_minus` axis on HF, so activation capping is only available
in the E+ direction.

Tonight's matrix runs activation capping with `--fractions "-0.5,0.0,0.5,1.0"`
on both scenario and sysprompt elicitation, alongside the LoRA scale
sweep. Note: activation capping uses HF transformers (no vLLM), so it's
~3-5x slower per cell than vLLM-backed runs.

**CAA (Contrastive Activation Addition)**: deferred.

No infrastructure exists yet. Mentor specifically mentioned CAA as a
preferred comparator; would need a `CAAProvider` class in
`src_dev/activation_capping/` (or sibling) that adds `α * v` to the
residual stream during forward pass. ~3-5 hours of focused engineering
plus axis-loading bookkeeping. Will do this properly in a separate
work session, not tonight under a 6h budget.

---

## Tonight's run plan (`run_tonight.sh`)

Sequential matrix in `scripts_dev/rollout_experiments/ocean/run_tonight.sh`:

| # | Cell | Time est | Purpose |
|---|------|----------|---------|
| 1 | v2 scenarios on base | ~15 min | Validate which new scenarios drift |
| 2 | Neutral baseline | ~25 min | Natural drift reference (no pressure) |
| 3 | E+ LoRA scale sweep on winners (0.25, 0.5, 0.75) | ~90 min | Find prevention sweet spot |
| 4 | E+ LoRA scale sweep on sysprompt (0.25, 0.5, 1.0) | ~75 min | Test with assistant-side elicitation |
| 5 | Activation capping on winners (-0.5..1.0) | ~60 min | Comparator on scenarios (HF, no vLLM) |
| 6 | Activation capping on sysprompt (-0.5..1.0) | ~50 min | Comparator on sysprompt (HF, no vLLM) |

Total: ~5 hours sequential. Safely fits in the 6h budget with overhead.
Each cell is independent; if one fails the rest continue (`set -e` not
enabled). Per-cell logs in `logs/<cell_name>_<timestamp>.log`; master
log in `logs/tonight_master_<timestamp>.log`.

Skipped from tonight's matrix:
- E− LoRA on E− scenarios (just amplifies drift; low information value
  per discussion)
- E− LoRA on E+ scenarios (we said E+ scenarios are weak)
- Anything in the E+ direction as the *target* (not what we're proving)

### Per-scenario breakdown (10-turn base)

| Direction | Strongest | Weakest |
|---|---|---|
| E+ | `team_pitch_hype_01` (+2.5, +2.3 drift) | `dating_app_match_01` (-0.8, +2.0 drift but starts low) |
| E− | `rainy_afternoon_reading_01` (-2.8, +1.0 climb) | `introvert_drained_after_party_01` (-0.2, conv too short) |

E− "winners only" subset (mean < -1.5): `solo_cabin_weekend`,
`grief_evening`, `rainy_afternoon_reading`, `astronomy_clear_night`.

---

## Doubling down on E−

Mentor's preferred framing: prove drift prevention on one direction,
deeply, with the right comparators. **E− is that direction.**

### What "deeply" means

For the E− scenarios (the 4-5 that drifted on the base model), produce
the cross-method comparison:

| Method | Status | Predicted shape |
|---|---|---|
| Base (no intervention) | ✅ have it | Drifts down to ~−2 by turn 4 |
| E+ LoRA @ scale +1 | ⏳ to run | Should resist drift; stays near 0 (the headline result) |
| E− LoRA @ scale +1 | ⏳ to run | Should amplify drift; goes deeper / faster |
| E+ LoRA scale sweep (e.g. +0.5, +1, +2) | ⏳ to run | Resistance scales with scale |
| Activation capping on E+ axis | ⏳ blocked | Direct comparator (mentor's "right comparison") |
| CAA steering toward E+ | ⏳ no infra | Direct comparator (mentor "do both") |
| Counter-system-prompt ("Be reserved...") | ⏳ to run | Behavioural baseline (cheap, prompt-only steering) |

Outputs from each method become a single trajectory plot:
x = turn, y = extraversion score (and a coherence subplot below).
Each method = one line. The "intervention flattens the drift" story
is whether each line beats the base curve.

### Recommended order of work

1. **LoRA-only matrix** (cheap, can run tonight on A40):
   - E− scenarios (winners subset) × {base, E+ @1, E− @1}
   - Then add E+ @ scales {0.5, 2.0} for the scale-sweep follow-on
   - All scenario runs at 10 turns to start; extend to 15 if drift
     keeps deepening past turn 9 in the longer base run

2. **Counter-prompt baseline** (cheap, useful, prompt-only):
   - E− scenarios × {base + sysprompt "be reserved", base + sysprompt
     "be exuberant"}
   - Compares: does sysprompt steering work as well as LoRA?

3. **Activation capping** (medium effort — fix the axis-against-vanton4
   problem first, runs without vLLM):
   - One condition: clamp axis projection at base's mean
   - Compare to LoRA on the same scenarios

4. **CAA** (high effort — needs new infra in
   `src_dev/activation_capping/` or sibling module):
   - Add steering vector during forward pass
   - Compare to LoRA + capping on the same scenarios

### Convenient run commands

Filter scenarios via the new `--scenario-ids` flag. Subsets are also
listed in `extraversion_pressure_v1.json` under `meta.scenario_subsets`.

```bash
# E- "winners only" subset, base model, 10 turns
WINNERS="e_minus_solo_cabin_weekend_01,e_minus_grief_evening_01,e_minus_rainy_afternoon_reading_01,e_minus_astronomy_clear_night_01"

uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
    --traits e_minus --method base \
    --conditions pressure_scenarios \
    --scenario-ids "$WINNERS" \
    --num-rollouts 3 --num-turns 15 \
    --user-model openai/gpt-4.1-mini \
    --vllm

# Same scenarios, with E+ LoRA at scale +1 (headline drift-prevention)
uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
    --traits e_plus --method lora --scale-points "1.0" \
    --conditions pressure_scenarios \
    --scenario-ids "$WINNERS" \
    --num-rollouts 3 --num-turns 15 \
    --user-model openai/gpt-4.1-mini \
    --vllm

# Same scenarios, with E- LoRA at scale +1 (drift amplifier)
uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
    --traits e_minus --method lora --scale-points "1.0" \
    --conditions pressure_scenarios \
    --scenario-ids "$WINNERS" \
    --num-rollouts 3 --num-turns 15 \
    --user-model openai/gpt-4.1-mini \
    --vllm
```

(Note: `--traits` selects the registry entry that picks the LoRA
adapter. For scenario routing, all `e_*` traits resolve to the same
extraversion scenario file via `_trait_to_scenario_file`.)
