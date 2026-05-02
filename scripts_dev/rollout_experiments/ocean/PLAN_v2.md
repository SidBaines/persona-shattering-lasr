# OCEAN Rollout Experiments — v2 plan

**Branch:** `irakli/ocean-rollouts-v2` (off `main`)
**Status:** Aligned with mentor framing (Part 1 / Part 2). Ready to execute
this evening.

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
