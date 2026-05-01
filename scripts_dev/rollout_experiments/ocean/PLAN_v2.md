# OCEAN Rollout Experiments — v2 plan

**Branch:** `irakli/ocean-rollouts-v2` (off `main`)
**Status:** Draft, pending team discussion.

The handover came back to me. This doc gathers what we learned from v1 and
proposes scoping for v2. Not yet committed to — to be aligned with the team
before any code lands.

---

## What v1 produced

- A working sweep script ([generate_rollouts.py](generate_rollouts.py)) with
  4 condition modes (`baseline` / `pressure` / `system_prompt` /
  `pressure_scenarios`).
- A reusable scenario format and one trait file
  ([extraversion_pressure_v1.json](../../../datasets/scenarios/extraversion_pressure_v1.json)).
- Sweep infra extensions: `prompt_template_per_sample` and
  `user_sim_generates_opening` on `SweepCondition`.
- One real run on A- with documented findings → see
  [FINDINGS.md](FINDINGS.md).
- HF upload + auto-eval + sweep plot pipeline.

## What v1 surfaced (from FINDINGS.md)

These are the load-bearing observations that should drive v2 design:

1. **Negative scaling of a suppressor adapter ≠ amplifying suppression.**
   At scale -2.0 on `a_minus`, the model went *more* agreeable, not less.
   Adapter null space dominates at extreme negative scales. Scale ranges
   should probably be asymmetric and method-aware.

2. **`user_agreeableness_high` was too sycophantic to be diagnostic.**
   The user-pressure templates need to actually push the assistant, not
   just praise it. v1 templates are placeholder-quality.

3. **5 turns > 10 turns for clean signal.** Conversations drifted
   significantly by turn 8-10 (off-topic).

4. **OOM at `max_new_tokens=4096` in late turns.** 1024 is safer; verbose
   responses also make judging noisier.

5. **Activation capping wasn't usable** — axes trained against earlier
   adapter versions (`v2`/`vanton1`), not current `vanton4_paired_dpo`.
   Plus a separate suspected bug.

6. **Judge reliability not yet validated** for the rollout setting.
   `agreeableness_v2` is calibrated on something — is it the right judge
   for multi-turn conversations?

---

## Open questions for the team

> All TBC — decide before writing code.

### Research scope

- **Which traits next?** Extraversion has scenarios; A- has a real run.
  Do we want full OCEAN coverage, or focus on 1–2 traits and go deep?
- **Are scenarios the right primary mode?** v1 has both
  `system_prompt` (single-turn, t-frequency-style) and
  `pressure_scenarios` (multi-turn, in-character). What's the canonical
  experiment for the paper?
- **Is the LoRA adapter quality the bigger lever?** `vanton4_paired_dpo`
  is the current canonical, but FINDINGS.md suggests adapter geometry
  matters more than scale. Do we trust the adapters at the full sweep
  range, or focus on `[0, +1, +2]`?
- **Do we want to revisit activation capping?** Requires recomputing axes
  against new adapters. Worth it only if it gives qualitatively
  different signal than LoRA scale.

### Methodology

- **Judge selection.** Panel of judges or a single trusted one?
  Inter-judge agreement on a held-out cell would clarify. Note: judge
  panel was updated since v1 last ran.
- **Conversation length.** Default to 5 turns based on FINDINGS.md? Or
  measure drift more carefully and pick per-trait?
- **User simulator strength.** v1 default was `gpt-4.1-nano`. FINDINGS
  used `gpt-4.1-mini`. Worth pinning a stronger one?
- **`user_agreeableness_high`-style sycophancy fix.** Rewrite the
  pressure templates from scratch, or replace with scenario-only mode?

### Code priorities (pending research scope)

- Polish `_OCEAN_BEHAVIOR_PROMPTS` (derive from
  [`OCEAN_DEFINITION`](../../../src_dev/common/persona_definitions.py))
- Author scenario files for A/C/N/O traits
- Re-enable activation capping (axes + bug fix)
- Add `--no-upload` flag for dry runs
- Plot improvements (pending mentor feedback)

---

## Proposed v2 structure

> **Tentative — depends on team alignment.**

### Phase 1 — sharpen the methodology (this branch)

Get one trait running cleanly end-to-end with the team's chosen settings.
No production code; just iterate fast on a single trait.

- Pick the canonical experiment design (mode, turns, scale grid, judge,
  user-sim model).
- Re-author behavior prompts and the sycophantic user template.
- Validate the chosen judge against a held-out cell (manual eyeball or
  judge agreement metric).
- Re-run A- (or another agreed trait) with the v2 settings → produce
  one clean reference run.

### Phase 2 — broaden coverage

After v2 settings are stable:

- Author scenario files for the remaining 4 traits.
- Run the full panel.
- Compare across traits.

### Phase 3 — secondary axes

If time / value:

- Re-enable activation capping with fresh axes.
- Cross-method comparison (LoRA × capping × prompting).

---

## Decisions needed from the team

Tag the team on these before coding:

- [ ] Trait priorities for v2 (one trait or full panel?)
- [ ] Canonical experiment design (mode + turns + scale grid)
- [ ] Judge selection (which one(s), and how to validate)
- [ ] User simulator model (and whether to keep or replace the pressure
      templates)
- [ ] Whether to invest in activation capping for v2

---

## What I'll do next

After team discussion lands:

1. Commit the agreed-upon experiment design as code changes on this branch.
2. Run the v2 reference experiment on RunPod.
3. Update [FINDINGS.md](FINDINGS.md) with v2 results — keep v1 entries
   for comparison.
4. Open a PR when v2 is comparable in quality to a publishable result.
