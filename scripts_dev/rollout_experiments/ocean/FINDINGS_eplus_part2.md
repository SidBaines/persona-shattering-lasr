# E+ LoRA Drift Prevention — Findings (Part 2, 2026-05-03)

**Branch:** `irakli/judge-calibration` (work continued from `irakli/ocean-rollouts-v2`)
**Adapter:** `extraversion/amplifier/vanton4_paired_dpo` (e_plus)
**Method:** LoRA scale sweep on E− pressure scenarios (drift prevention)
**Pool:** 9 scenarios (4 v1 winners + 5 v2)

---

## Coherence collapse pattern (LoRA scale 0.5, scenarios)

Headline trait result is clean: at scale 0.5, the E+ LoRA holds extraversion at ~0
across 15 turns while the base model drifts to ~−2.4. But coherence has a striking
bimodal failure mode worth flagging before it goes into the paper.

### Pattern

Mean coherence per (scenario, rollout):

| theme            | scenario                     | coh_mean | % zero turns | avg max len |
|------------------|------------------------------|---------:|-------------:|------------:|
| emotional        | rough_day_grief              |     9.00 |         0.0% |         697 |
| emotional        | solo_cabin                   |     8.42 |         6.7% |        2211 |
| emotional        | rainy_reading                |     8.02 |         8.9% |        2916 |
| emotional        | late_night_regret            |     7.47 |        15.6% |        3059 |
| factual          | astronomy                    |     6.20 |        15.6% |        3170 |
| task             | estranged_letter_writing     |     5.82 |        33.3% |        2797 |
| task             | translating_haiku            |     7.38 |        15.6% |        2871 |
| task+emotional   | old_letters_drawer           |     7.42 |        15.6% |        2671 |
| contemplative    | dawn_light_question          |     6.56 |        22.2% |        3074 |

The **strongest predictor of coherence collapse is mean assistant message length per
scenario**, not theme per se. Theme matters because emotional grief scenarios elicit
short empathetic replies, while task/factual scenarios elicit long structured responses
with bullets, bolds, "GREAT!"-style markers — exactly what the e_plus LoRA amplifies.
Length is the catalyst; LoRA is the trigger.

### Failure mode

Long assistant outputs (>~3000 chars / ~750 tokens) tend to enter a token-soup tail:
plausible-token-distributed nonsense like

> "Ex Nass Pall Lose Estimate:** Anderson union plantation mitigation Determine BE
>  become mRNA dictionary Visibility Aeros arm Inno Mutual Boulder manga dish ev
>  artifact ind ONE+. naturally Parliament arts Chiefs tissue down Dow alone Oman..."

Once a turn collapses, the truncated/garbled previous response goes into the next
turn's prompt and the model rarely recovers in the same conversation. Run-to-run
variance within a scenario is large (3.0 / 6.07 / 8.4 on the same sibling-letter
scenario across three rollouts) — but it's variance in *whether* the long-output
mode tips into collapse, not variance in baseline coherence of short outputs.

The base model on the same scenarios stays at coherence ~9 throughout, so this is
clearly LoRA-induced, not a property of the scenarios themselves.

### Side-by-side transcripts

Four full conversations (2 healthy + 2 collapsed across 2 scenarios), with per-turn
coherence/extraversion/length annotations:

- [scratch/qualitative_eplus_lora_coherence/coherence_collapse_pairs_scenarios.md](../../../scratch/qualitative_eplus_lora_coherence/coherence_collapse_pairs_scenarios.md)

Recommended mental model: read the healthy rollout first to see the LoRA's intended
voice (warm, encouraging, slightly bullety), then the collapsed rollout to see the
same voice tip into noise around turn 4-5 of the long-output scenarios.

### Open mitigation

Worth testing: cap `assistant_max_new_tokens` at 256 instead of 512 to break the
long tail before it can drift into noise. Cheap follow-up cell.

---

## Sysprompt-elicit (low) — much more aggressive collapse

Same E+ LoRA, but instead of scenario-driven user pressure, the assistant gets a
system prompt instructing it to "be reserved, quiet, calm, deliberate, and measured"
(low-extraversion persona instruction). The intervention conflict is now explicit
(instruction-level) rather than contextual (scenario-level).

### Per-prompt summary (LoRA scale 0.5)

| Prompt theme              | base coh | base avg max len | LoRA coh | LoRA % zero-turns | LoRA avg max len |
|---------------------------|---------:|-----------------:|---------:|------------------:|-----------------:|
| AI safety                 |     8.87 |             2616 |     1.71 |             73.3% |             3756 |
| Music genre               |     8.76 |              904 |     5.53 |             31.1% |             3247 |
| Tradition ending          |     8.96 |             2243 |     4.07 |             37.8% |             3336 |
| Dad never said love       |     8.93 |             1688 |     5.38 |             31.1% |             2682 |
| Missing fundamental       |     8.78 |             1723 |     5.49 |             26.7% |             3303 |
| Advice to younger self    |     9.00 |             2133 |     4.31 |             42.2% |             3401 |
| Sustainable urban         |     8.87 |             2866 |     4.44 |             40.0% |             3489 |
| Watched dog sleep         |     8.82 |             1201 |     4.91 |             28.9% |             3361 |

### Key result vs scenarios

In the scenario regime, base outputs are short for emotional prompts (~700 chars max
for grief) and long for task prompts (~3000 chars for writing-help). Coherence
collapse follows length.

In the sysprompt-elicit regime, **base outputs are already long (1700-2900 chars max)
across every prompt, yet base coherence is uniformly ~8.8-9.0**. Length alone doesn't
trigger collapse. What pushes coherence off a cliff is the **explicit instruction
conflict**: system prompt says "be introverted", LoRA weights push "be extraverted",
and the model never recovers. AI safety prompt at LoRA 0.5 reaches 73% zero-coh turns —
nearly every assistant turn devolves into token soup.

### Side-by-side transcripts

- [scratch/qualitative_eplus_lora_coherence/coherence_collapse_pairs_sysprompt.md](../../../scratch/qualitative_eplus_lora_coherence/coherence_collapse_pairs_sysprompt.md)

Three pairs (AI safety, urban planning, advice-to-younger-self) showing the same
prompt under base vs LoRA scale 0.5. Read base first to see the model handling long
nuanced answers fine, then LoRA to see the same prompt produce incoherent output
from turn 1.

### Implication

Sysprompt-elicit is **not a clean comparator** for drift prevention. It's its own
finding: "explicit instruction conflict between system prompt and weight intervention
destroys coherence in a way that scenario-level conflict does not." Worth a paper
subsection but distinct from the headline drift-prevention story.
