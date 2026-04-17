---
name: label-fa-factors
description: Label factors from a factor-analysis rotation in scratch/psychometric_fa/. Inspects loadings, item text, and items.json, then writes an LLM-labels cache file that the psychometric_rollout_fa.py stage picks up on its next run.
---

# Label FA Factors

You are labelling the latent factors from one rotation of a factor analysis
produced by `scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py`.

Outputs go to
`{run}/questionnaire/labeling/llm_labels_{analysis_key}_manual_{ts}.json`.
The script (run with `LABELLER_MODE = "manual"`) loads the latest non-empty
match on its next run and plumbs the labels through `factor_extremes.html`
and any downstream plots. Your job is to produce that cache file.

## Scope

One rotation per invocation. If the user did not specify one, list the
available rotations under their path and ask which to label.

## Tool

`python3 .claude/skills/label-fa-factors/helpers/fa_label_tools.py` with
three subcommands:

- `resolve <path>` — returns JSON: either a resolved `{npz, analysis_key,
  output_path, items_json, save_dir, …}` object, or `{"ambiguous": true,
  "choices": [...]}` when `<path>` contains several rotations.
- `describe <path> [--top-n N]` — prints markdown: per-factor top-N positive
  and negative loading items, with full block-aware descriptions for
  Likert / fc_pair / trait_mcq / vignette / fc items, plus communalities,
  cross-loadings, SS loading, proportion variance, and factor correlations.
  `<path>` must resolve to a single `.npz`.
- `write <path> --labels <labels.json>` — validates the labels payload and
  writes it to the resolved `output_path`. Validates: exactly one entry per
  factor, no duplicates, non-empty `axis_name`, all required fields present.

## Procedure

1. **Resolve.** Run `resolve $ARGUMENTS` (or a user-provided path). If the
   response has `"ambiguous": true`, show the choices to the user, ask which
   rotation they want, then re-resolve with that specific `.npz` path.

2. **Describe.** Run `describe <resolved-npz> --top-n 10` and read the
   markdown. Each factor section has positive-pole and negative-pole items
   with behavioural direction already computed (so reverse-keying and
   letter-encoding quirks are already accounted for — do not second-guess
   the signs).

3. **Cross-check (optional).** If helpful, read
   - `<save_dir>/plots/` heatmaps / loading distributions,
   - the alignment CSVs in `<rotation_dir>/*_alignment/` (when present),
   - `<items_json>` for any item you want to see in full,
   - the existing `factor_extremes.html` for a rendered view.
   Do this selectively — don't dump large files into context without reason.

4. **Label each factor.** For every factor produce:
   - `factor_index` (int, matches the described factor).
   - `axis_name` — one word, Title Case, naming the latent dimension.
   - `summary` — ≤12 words, `"pole_A vs pole_B"` form, maximally distinct
     from other factors.
   - `description` — 2–3 sentences explaining what the factor captures and
     what differentiates it from its nearest neighbour (the factor
     correlations in the `describe` output point you at likely neighbours).
   - `positive_pole` — short phrase naming the high end.
   - `negative_pole` — short phrase naming the low end.
   - `dominant_item_types` — list of block names (`"likert"`, `"fc_pair"`,
     `"trait_mcq"`, `"vignette"`, `"fc"`) that drive this factor.

5. **Write.** Save the labels as JSON (list of dicts — a top-level
   `{"factors": [...]}` wrapper is also accepted) to a temp file under
   `/tmp/`, then run
   `write <resolved-npz> --labels /tmp/<your-file>.json`. The tool echoes
   the final path. If validation fails, fix the payload and retry — do not
   write directly to the labeling dir, always go through `write`.

6. **Report.** Print the output path and a one-line summary per factor
   (`Factor N: [AxisName] pole_a vs pole_b`) so the user can eyeball the
   result.

## Conventions and gotchas

- **Sign interpretation is already decoded** in the `describe` output —
  lines end with `→ + loading means …` phrasing. Trust that, especially for
  reverse-keyed Likert items and fc_pair items where the matrix encoding
  flips per item.
- **trait_mcq with `encoding=letter_1-4`** is NOT trait-interpretable by
  sign. Do not claim that such a factor captures a trait direction purely
  on that evidence; look for consistent patterns across *other* blocks.
- **Per-block rotations** (analysis_key like `block_likert_raw_oblimin`)
  only see items from one measurement modality. Labels should reflect
  what's visible in that block alone — do not invent cross-block structure.
- **Cross-loadings** (`max-cross=fJ:+0.45`) above ~0.4 mean the item is not
  a clean marker; weight it less.
- **Communalities** below ~0.2 mean the item is barely explained by any
  factor — usually safe to de-emphasise.
- Make the `axis_name` and `summary` distinguishable across factors. If two
  factors overlap conceptually, the `description` should spell out the
  distinction (e.g. "unlike Factor 2's breadth-of-openness, this factor
  specifically tracks contrarian disagreement even when agreeing").
- Do not invent items. Every claim in a description should correspond to
  patterns visible in the described loadings.

## Example JSON payload

```json
[
  {
    "factor_index": 0,
    "axis_name": "Verbosity",
    "summary": "expansive elaboration vs terse minimalism",
    "description": "High scorers favour long, context-rich responses and volunteer tangents; low scorers match response length to the narrow literal question. Mostly driven by Likert items about response length, with matching fc_pair items picking the longer-reply option.",
    "positive_pole": "long, context-rich responses",
    "negative_pole": "terse, literal answers",
    "dominant_item_types": ["likert", "fc_pair"]
  }
]
```
