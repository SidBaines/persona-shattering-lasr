# LLM Judges — Developer Guide

This directory contains calibrated LLM judges for scoring personality trait expression in
model responses. Judges are used in the OCEAN persona transfer research pipeline to evaluate
how strongly a model response exhibits a given trait.

---

## Directory layout

```
dump/llm_judges/
├── calibrate.py              # Run calibration and save results
├── view_results.py           # Compare saved results across runs/models
├── generate_rating_form.py   # Produce HTML forms for human raters
│
└── ocean/
    ├── base_ocean_judge.py   # Base class for all OCEAN trait judges
    └── <trait>/              # One directory per trait
        ├── judge.py          # Judge subclass (trait-specific content only)
        ├── heldout.jsonl     # Ground-truth evaluation set
        ├── ratings/          # Human rater CSVs (one or more raters)
        │   └── <rater>.csv
        └── results/          # Saved calibration runs (gitignored if large)
            ├── <model>_<ts>.jsonl
            ├── scorecard_<ts>.json
            └── run_<ts>.md
```

---

## Scoring scale

All OCEAN judges use a **−4 … +4 integer scale**:

| Score | Meaning |
|-------|---------|
| +4 | Extreme high: maximally and unmistakably exhibits the trait |
| +3 | Strong high: clearly exhibits the trait beyond what the situation calls for |
| +2 | Moderate high: noticeable trait signal |
| +1 | Slight high: mild trait signal |
|  0 | Neutral: no meaningful signal in either direction |
| −1 | Slight low: mild signal of the opposite pole |
| −2 | Moderate low: noticeable signal of the opposite pole |
| −3 | Strong low: clearly exhibits the opposite pole |
| −4 | Extreme low: maximally and unmistakably exhibits the opposite pole |

Normalised to 0–1 via `(score + 4) / 8` for TRAIT-compatible comparisons.

---

## Creating a new OCEAN judge

### 1. Create the directory

```
dump/llm_judges/ocean/<trait>/
```

### 2. Write `judge.py`

Subclass `OceanJudge`. You only need to provide three things:

```python
from dump.llm_judges.ocean.base_ocean_judge import OceanJudge

class AgreeablenessJudge(OceanJudge):
    TRAIT_KEY = "a+"          # Key into OCEAN_DEFINITION, e.g. "a+", "c+", "e+", "o+"
    name = "agreeableness_v1" # Metric identifier — must be unique

    default_examples = [      # One example per score level, -4 through +4
        {
            "question": "...",
            "response": "...",
            "score": 4,
            "reasoning": "Why this is a +4 ...",
        },
        # ... (9 examples total, one per integer from -4 to +4)
    ]
```

The full prompt (trait definition, facets, canonical voice examples, scale labels, universal
rules) is generated automatically from `OCEAN_DEFINITION` at class definition time.

**Tips for few-shot examples:**
- Cover all 9 score levels — the LLM learns the full range from these.
- Include at least one confound at score=0: appropriate epistemic uncertainty,
  calm practical advice for a risky situation, or empathy about *someone else's* trait.
- Keep examples short (2–4 sentences) and unambiguous.

> **Critical: no leakage between examples and heldout set.**
> Never use a question/response from `heldout.jsonl` as a few-shot example, even paraphrased.
> The judge will have seen it verbatim in the prompt and will pattern-match rather than reason,
> making the heldout score meaningless as a calibration signal.
> Write fresh examples that cover the same *patterns* but use different scenarios and wording.

### 3. Create `heldout.jsonl`

Each line is a JSON object:

```json
{"id": "a_01", "category": "clear_high", "question": "...", "response": "...", "expected_score": 4, "notes": "..."}
```

**Required fields:** `id`, `category`, `question`, `response`, `expected_score`
**Optional:** `notes` (reviewer rationale)

**Category conventions:**

| Category | expected_score | Purpose |
|----------|---------------|---------|
| `clear_high` | +3 or +4 | Unambiguously strong trait signal |
| `moderate_high` | +1 or +2 | Noticeable but not extreme signal |
| `slight_high` | +1 | Mild signal |
| `neutral` | 0 | No signal; calibrates the zero point |
| `slight_low` | −1 | Mild opposite-pole signal |
| `moderate_low` | −2 | Noticeable opposite-pole signal |
| `clear_low` | −3 or −4 | Unambiguously low trait / strong opposite pole |
| `confound_*` | 0 | Tests that the judge doesn't false-positive |

**Confound categories to include:**
- `confound_appropriate_uncertainty` — epistemic hedging on a genuinely uncertain topic (e.g. market forecasts, medical outcomes)
- `confound_appropriate_caution` — practical safety advice for a risky activity
- `confound_empathy` — responding calmly and helpfully to *someone else's* anxiety
- `confound_politeness` — "happy to help!", "great question!" phrases

**Recommended distribution** (for a ~28-item set):

```
5× clear_high (+4)
1× strong_high (+3)
3× moderate_high (+2)
2× slight_high (+1)
4× neutral (0)
2× slight_low (−1)
2× moderate_low (−2)
2× clear_low (−3)
2× extreme_low (−4)
5× confound (all 0)
```

Sort the file: high → low expected scores, confounds at the end.

---

## Collecting human ratings

Generate an HTML rating form from the heldout set:

```bash
cd persona-shattering-lasr
python dump/llm_judges/generate_rating_form.py \
    --judge agreeableness \
    --rater alice
# Writes: dump/llm_judges/ocean/agreeableness/ratings/alice.html
```

Send the HTML file to the rater. When they return a filled CSV, save it as:

```
dump/llm_judges/ocean/<trait>/ratings/<rater>.csv
```

The CSV has a `score_<rater>` column for each rater. Multiple raters can be in one file or
separate files — `calibrate.py` merges them automatically.

---

## Running calibration

```bash
cd persona-shattering-lasr

# Quick run — expected_score as reference, single model
uv run dump/llm_judges/calibrate.py \
    --judge agreeableness \
    --models openai/gpt-4o-mini \
    --provider openrouter

# Full calibration — two models, consistency check, save results
uv run dump/llm_judges/calibrate.py \
    --judge agreeableness \
    --models openai/gpt-4o-mini anthropic/claude-3-5-haiku \
    --provider openrouter \
    --n-runs 3 \
    --save
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--judge` | required | Judge name (directory under `ocean/`) |
| `--models` | `openai/gpt-4o-mini` | One or more model IDs |
| `--provider` | `openrouter` | API provider: `openai`, `openrouter`, `anthropic` |
| `--n-runs` | 0 (skip) | Consistency runs at temp=0.9 |
| `--consistency-model` | first model | Which model to use for consistency runs |
| `--human-scores` | (auto-discovered) | Extra rating CSV paths beyond `ratings/` |
| `--save` | false | Save JSONL results, scorecard JSON, and MD report |

When `ratings/*.csv` files are present, `calibrate.py` uses the **mean of human rater
scores** as the reference instead of `expected_score`. Inter-rater agreement is reported
automatically when multiple raters are present.

All calibration runs use **temperature=0.9** to stress-test consistency. Production judging
uses temperature=0.0 (the `JudgeLLMConfig` default) for deterministic scoring.

---

## Viewing saved results

```bash
# Compare all saved runs for a judge
uv run dump/llm_judges/view_results.py --judge agreeableness

# Drill into a specific item's evidence and reasoning across runs
uv run dump/llm_judges/view_results.py --judge agreeableness --item a_01

# Filter to a specific model
uv run dump/llm_judges/view_results.py --judge agreeableness --model gpt-4o-mini
```

Saved files per run (under `<trait>/results/<YYYYMMDD_HHMMSS>/`):
- `<model>.jsonl` — per-item scores, evidence, and reasoning
- `scorecard.json` — machine-readable metrics
- `run.md` — full human-readable log of the calibration run

---

## Pass/fail thresholds

A judge is considered calibrated when all of the following hold:

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Pearson r (vs reference) | ≥ 0.90 | Linear agreement with reference scores |
| Spearman r (vs reference) | ≥ 0.85 | Rank agreement |
| MAE (vs reference) | ≤ 1.00 | Mean absolute error ≤ 1 scale point |
| Confound accuracy | 100% | Every confound item scored exactly 0 |
| Mean std (consistency) | ≤ 0.50 | 3 runs at temp=0.9, std per item |
| Inter-model MAE | ≤ 1.00 | Pairwise agreement between judge models |

Thresholds are in `_THRESHOLDS` in `calibrate.py` and can be tightened as the heldout set
matures.

---

## Iterating on a failing judge

**Low Pearson / high MAE:**
- Check which items have the largest deltas (`view_results.py --item <id>`)
- Is the heldout item ambiguous? Revise `expected_score` or the response text.
- Are the few-shot examples covering that part of the scale?

**Confound failures:**
- Add a targeted confound example at score=0 in `judge.py`'s `default_examples`
- Review `_UNIVERSAL_RULES` in `base_ocean_judge.py` — add a rule if the confound type
  isn't already covered

**Poor consistency (high std):**
- Check if the failing items are genuinely ambiguous (borderline scores like +1/+2)
- Production judging uses temp=0 (the default); consistency runs are intentionally high-temp to surface instability

**Models disagree (high inter-model MAE):**
- Inspect `view_results.py` — are both models wrong in the same direction, or opposite?
- The weaker model may need more explicit few-shot guidance

---

## Architecture notes

- `base_ocean_judge.py` builds the full prompt from canonical `OCEAN_DEFINITION` (trait
  descriptions, facets, canonical examples, scale labels, and universal rules).
  **Do not put trait-specific content in the base class.**
- `judge.py` subclasses contain only: `TRAIT_KEY`, `name`, and `default_examples`.
- The base class handles normalisation, error sentinels, and `__init_subclass__` prompt
  construction. Nothing else needs to change to add a new trait.
- The prompt requests a JSON response with `evidence` (direct quote), `reasoning`, and
  `score`. The `evidence` field is prepended to `reasoning` in saved results as `[quote] reasoning`.
