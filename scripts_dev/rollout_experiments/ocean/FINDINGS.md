# Rollout Sweep Findings — A- (Agreeableness Suppressor)

**Date:** 2026-04-17  
**Branch:** irakli/ocean-rollout-evals  
**Adapter:** `agreeableness/suppressor/v2` (a_minus)  
**Method:** LoRA scale sweep, pressure conditions  
**Data so far:** scale_-2.00 (3 conditions) + scale_-1.50/baseline

---

## Judge Score Summary

| Cell | n turns | mean score | score distribution |
|------|---------|------------|-------------------|
| scale_-2.00 / baseline | 960 | **1.66** | mostly 0, 3, 4 |
| scale_-2.00 / user_agreeableness_high | 960 | **2.19** | mostly 0, 3, 4 |
| scale_-2.00 / user_agreeableness_low | 960 | **1.98** | broader spread, some negatives |
| scale_-1.50 / baseline | 960 | **1.68** | similar to -2.00 |

Score range: -4 (very low agreeableness) to +4 (very high agreeableness).

**Key finding:** At negative scales (-2.0, -1.5), the model scores *positive* on agreeableness
(mean ~1.7-2.2). The suppressor at negative scale is not producing low-agreeableness behavior —
it appears to push the model toward verbose, generic, accommodating responses rather than
the expected disagreeable/curt style. This may indicate the suppressor direction is
not simply invertible by negating the scale.

---

## Qualitative Observations

### Assistant behavior
- Responses are very long and heavily structured (bold headers, numbered lists) regardless
  of scale or condition. This is a base model tendency amplified by negative suppressor scaling.
- Responses are *accommodating* — validating user framing, hedging, providing balanced
  perspectives. Not what a low-agreeableness persona should look like.
- No clear differentiation between scale -2.0 and -1.5 qualitatively.
- `max_new_tokens=4096` (original run) caused ~4500 char avg responses and CUDA OOM at
  late turns. Fixed to 1024 for subsequent runs.

### User simulator (gpt-4.1-mini)
- `user_agreeableness_low`: Good. Realistic skeptic — pushes back with phrases like
  *"a lot of hopeful talk without much bite"*, challenges the assistant's framing, escalates
  pressure across turns. Score distribution is broader (some -3, -2 scores), suggesting
  this condition does pull out low-agreeableness behavior occasionally.
- `user_agreeableness_high`: Too sycophantic. Constantly echoes back what the assistant
  says (*"I love how you put it"*, *"that analogy really clicks"*). Creates only gentle
  social pressure; unlikely to be diagnostic.
- Conversations drift significantly by turn 8-10 (quantum computing, regulatory sandboxes)
  far from the seed topic. 5 turns is the right length for clean signal.

### Seed diversity
- 32 unique seed prompts confirmed. Diverse topics (AI safety, GPS, ancestry, psychology).

---

## Hypotheses for Next Experiments

1. **Scale 0.0 should look like a well-behaved base model.** Expected: neutral-to-positive
   agreeableness scores, coherent responses. This is the most important reference point.

2. **Scale +1.0 (suppressor amplified) should show low agreeableness.** Expected: curt,
   dismissive, or uncooperative responses. If scores go negative here, the adapter is
   working — just in the opposite direction of what negative scaling produces.

3. **Negative scaling of a suppressor ≠ amplifying suppression.** The data suggests that
   scale -2.0 on an a_minus adapter produces verbose/generic behavior rather than
   amplified disagreeableness. The adapter's null space likely dominates at extreme negative
   scales.

4. **Activation capping comparison:** Once capping data exists, compare whether capping
   at fraction +1.0 produces cleaner low-agreeableness signal than LoRA scale +1.0.

---

## Next Run Plan

```bash
# Quick vLLM sanity check first (2-3 min)
uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
    --traits a_minus --method lora \
    --scale-points="0.0" --conditions baseline \
    --max-samples 4 --num-rollouts 1 --num-turns 3 \
    --assistant-max-new-tokens 256 --vllm

# Main focused run
uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
    --traits a_minus --method lora \
    --scale-points="0.0,1.0" --conditions pressure \
    --max-samples 16 --num-rollouts 2 --num-turns 5 \
    --user-model openai/gpt-4.1-mini \
    --assistant-max-new-tokens 512 --vllm

uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
    --traits a_minus --method activation_capping \
    --fractions="0.0,0.5,1.0" --conditions pressure \
    --max-samples 16 --num-rollouts 2 --num-turns 5 \
    --user-model openai/gpt-4.1-mini \
    --assistant-max-new-tokens 512
```

---

## Action Plan

### Immediate (after current run finishes ~09:30)
- [ ] Kill current run, archive, run vLLM sanity check (4 samples, baseline, 3 turns)
- [ ] If vLLM passes, launch focused run: LoRA `0.0,1.0` + capping `0.0,0.5,1.0`, 5 turns, 16 samples, 2 rollouts

### After focused data arrives (~1-2h later)
- [ ] Eyeball scale 0.0 vs 1.0 qualitatively — confirm adapter steers in the right direction at positive scales
- [ ] Check judge score distributions — if 0.0 and 1.0 aren't separated, signal isn't there
- [ ] Run judge panel comparison (Gemini Flash vs Claude Haiku) on one cell to validate judge reliability

### Code
- [ ] Replace `user_agreeableness_high` with a neutral user condition (too sycophantic, not diagnostic) or drop it entirely
- [ ] Add scale `+2.0` to next LoRA run to check if suppressor amplification saturates

### Longer term / research
- [ ] Understand negative-scale behavior: why does scale -2.0 on a suppressor produce *higher* agreeableness? Plot score vs scale across all 9 points once full sweep runs
- [ ] Decide whether full 9-point sweep is worth running after seeing focused results — if 0.0/1.0 are clearly separated, extend; if not, debug the adapter first

---

## Open Questions

- Why does scale -2.0 on a suppressor produce *high* agreeableness scores? Is this a
  property of the adapter geometry or the judge calibration?
- Is the `user_agreeableness_high` condition useful at all, or should it be replaced
  with a neutral user for cleaner baselines?
- Should we add scale +2.0 to see if suppressor amplification saturates?
