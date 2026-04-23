---
name: label-fa-factors
description: Label factors from a factor-analysis rotation in scratch/psychometric_fa/. Inspects loadings, item text, and items.json, then writes an LLM-labels cache file that the psychometric_rollout_fa.py stage picks up on its next run.
---

# Label FA Factors

You are labelling the latent factors from one rotation of a factor analysis
produced by `scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py`.

Outputs go to
`{run}/labeling/llm_labels_{analysis_key}_manual_{ts}.json` (sibling of
the `questionnaire/` subdir, matching the FA pipeline's
`cfg.ctx.effective_questionnaire_dir / "labeling"` convention). The script
(run with `LABELLER_MODE = "manual"`) loads the latest non-empty match on
its next run and plumbs the labels through `factor_extremes.html` and any
downstream plots. Your job is to produce that cache file.

## Research context

A set of LLM personas (rows) were each administered a psychometric
questionnaire containing Likert statements and multiple-choice items
(columns). Factor analysis decomposed that persona × item response matrix
into a small number of latent dimensions — the factors you are about to
label. Each factor is a direction along which personas vary in how they
answer the questionnaire. The loadings tell you how strongly, and in
which direction, each item moves with each factor.

This is deliberately a **loadings-first** task. The evidence you reason
from is the question text, the option text (for MCQs), the per-option
scoring, and the factor loadings. You do **not** look at persona-
generating transcripts or any other downstream behavioural evidence — the
point of this task is to describe the structure the factor analysis
surfaced, from the loadings alone, so the labels don't smuggle in
information the factor analysis itself didn't see. Similarly, do not
infer factor meaning from author-assigned trait / subscale names on the
items, from labels produced for related rotations, or from priors about
what a "five-factor solution" ought to look like. If the loadings on a
particular factor don't cohere into a nameable direction, record that
honestly via the `confidence` field (see step 4) rather than inventing
one.

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

   **Check the header for ⚠ rich-context warnings before trusting the
   output.** `describe` loads raw questionnaire items from
   `datasets/psychometric_questionnaires/` to render the full options for
   `trait_mcq` / `fc` / `fc_pair` / `vignette` items. If a raw source is
   missing, the header lists the affected versions and each degraded item
   is tagged with a `⚠ RICH CONTEXT MISSING` line — **do not label
   factors that are dominated by degraded items**, ask the user to
   resolve the missing source first.

3. **Cross-check (optional).** If helpful, read
   - `<save_dir>/plots/` heatmaps / loading distributions,
   - `<items_json>` for any item you want to see in full.
   Do this selectively — don't dump large files into context without reason.
   Do **not** open `factor_extremes.html`, rollout transcripts, or any
   per-persona outputs: labelling is deliberately loadings-only.

4. **Label each factor.** For every factor produce:
   - `factor_index` (int, matches the described factor).
   - `axis_name` — one word, Title Case, naming the latent dimension.
   - `summary` — ≤12 words, `"pole_A vs pole_B"` form, maximally distinct
     from other factors. (Waived for `confidence: "unlabelable"` — see
     below.)
   - `description` — 2–3 sentences explaining what the factor captures and
     what differentiates it from its nearest neighbour (the factor
     correlations in the `describe` output point you at likely neighbours).
   - `positive_pole` — short phrase naming the high end.
   - `negative_pole` — short phrase naming the low end.
   - `dominant_item_types` — list of block names (`"likert"`, `"fc_pair"`,
     `"trait_mcq"`, `"vignette"`, `"fc"`) that drive this factor.
   - `confidence` — one of `"high"`, `"medium"`, `"low"`,
     `"unlabelable"`. Pick according to how clean the loadings are:
       - `"high"` — top-loading items agree on a consistent behavioural
         direction with few contradictions; nearest-neighbour factors are
         clearly distinguishable.
       - `"medium"` — a direction is visible but some top items pull
         against it, or the factor overlaps with another more than
         you'd like; the label is your best read and could plausibly
         shift under a re-fit.
       - `"low"` — the loadings only weakly cohere; you're committing
         to a name because the task demands it, not because the
         direction is obvious.
       - `"unlabelable"` — the top-loading items genuinely do not tell
         a consistent story, and forcing any `pole vs pole` label would
         misrepresent the factor. Use this when you'd honestly prefer
         to mark the factor as "no clean axis" than pretend otherwise.
         When you pick this, the `summary` / `positive_pole` /
         `negative_pole` fields are free-form (the validator waives the
         `vs` form and pole-non-empty checks) — use them to describe
         what is actually there, e.g. `summary: "mixed — see
         description"`, or leave the pole fields empty.

5. **Write.** Save the labels as JSON (list of dicts — a top-level
   `{"factors": [...]}` wrapper is also accepted) to a temp file under
   `/tmp/`, then run
   `write <resolved-npz> --labels /tmp/<your-file>.json`. The tool echoes
   the final path. If validation fails, fix the payload and retry — do not
   write directly to the labeling dir, always go through `write`.

   The validator rejects payloads that miss any required field
   (`factor_index`, `axis_name`, `summary`, `description`, `positive_pole`,
   `negative_pole`, `dominant_item_types`, `confidence`), that leave
   duplicate or out-of-range `factor_index` values, or that violate the
   stylistic constraints from step 4: `axis_name` must be a single word in
   Title Case; `confidence` must be one of `"high"` / `"medium"` / `"low"`
   / `"unlabelable"`; `dominant_item_types` must be a non-empty list of
   recognised block names (`"likert"`, `"fc_pair"`, `"trait_mcq"`,
   `"vignette"`, `"fc"`). For `confidence ∈ {"high", "medium", "low"}` the
   validator additionally requires `summary` ≤12 words with ` vs `
   (case-insensitive), non-empty `positive_pole` / `negative_pole`, and
   `description` prose of ~15–120 words. For `confidence: "unlabelable"`,
   those extra checks are waived — `summary` and the pole fields become
   free-form, and `description` only needs to be non-empty (so you can
   record "mixed, no clean axis" without lying about the structure).

6. **Report.** Print the output path and a one-line summary per factor:
   `Factor N: [AxisName] (confidence) pole_a vs pole_b`, or
   `Factor N: [AxisName] (unlabelable) — <short reason>` when you marked
   it as such. So the user can eyeball the result.

## Conventions and gotchas

- **Always check direction per item — do not skim.** Getting the sign of
  a factor wrong is the most common labelling failure, because the
  question text alone is not enough: whether a high-factor persona
  *agrees* or *disagrees* with a likert item depends on whether it is
  reverse-keyed, and which option a trait_mcq / fc_pair / vignette item
  endorses depends on the item-specific scoring / `high_option`.
  `describe` pre-computes the right-sign interpretation and puts it on
  the `→` line of each item — but you still have to read it for every
  top item on both poles, not just the first one or two. The headline
  checks per block:
    - *Likert*: the `→` line tells you whether high-factor personas
      *agree* or *disagree*, already accounting for `reverse_keyed`.
      Don't infer from the statement text alone.
    - *fc_pair*: the header tells you which letter (`A` or `B`) is the
      `+1` pole for that item — the mapping flips per item, so a
      `+0.40` loading does not consistently mean "option A".
    - *trait_mcq* with `encoding=trait_score_0-1`: sign is interpretable,
      the `→` line tells you whether high-factor personas pick more
      `scored=1` or more `scored=0` options. With
      `encoding=letter_1-4`, sign is NOT interpretable at all (see
      below).
    - *vignette*: the `→` line tells you whether high-factor personas
      pick higher- or lower-scoring options *on the per-column scoring
      axis* (axis name is deliberately redacted — see below).
  Before writing a label, sanity-check by counting: do most top-positive
  items' `→` lines point the same behavioural way? Do the top-negative
  items point the opposite way? If the decoded directions contradict
  each other within one pole, the factor is a mess — mark it with
  `confidence: "unlabelable"` (or `"low"` if a best-effort direction is
  still defensible) and describe the contradiction in the `description`
  rather than forcing a clean story.
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
- **Label from loadings alone, not from priors.** Ignore any pre-existing
  labels you may encounter — including labels from earlier FA runs on the
  same or related data, and the original questionnaire's trait/subscale
  names attached to items in `items.json` or alignment CSVs. These priors
  encode what the instrument's authors *intended*, not what this rotation
  actually recovered. Read the question text and the answers/options that
  load high and low on each factor, and name the factor from that content.
  If your label happens to match a standard trait name, that's fine — but
  derive it from the loadings, don't back-fit the loadings to a familiar
  label. `describe` deliberately redacts the author-assigned `dimension`
  name from `trait_mcq` / `vignette` items for this reason; options are
  shown with numeric scores only, not trait labels.
- **Don't force-fit to OCEAN or other human psychometric frameworks.**
  The research target is *whatever structure the loadings actually
  recover*, not a reconstruction of any particular human taxonomy.
  A rotation may recover Big-Five-like structure, finer sub-dimensions,
  instrument-specific artefacts, cross-construct blends, or genuinely
  novel axes that don't map to any standard label. If the loadings
  really do look like a Big-Five trait — content-consistent across
  blocks, not merely item-provenance-consistent — using that name is
  fine. But do not label a factor "Extraversion" because the items came
  from an extraversion scale, or because the rotation has five factors
  and factor `k` is "the OCEAN-ish one". Coin a new label if nothing
  standard fits — novel axes are a legitimate research finding, not a
  failure mode.
- **`col_id` values may hint at the item's origin** (e.g.
  `trait_ocean_v1/mcq_extraversion_3`). Treat them as debugging metadata
  only — the item's *content* is what you name from, not its id.

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
    "dominant_item_types": ["likert", "fc_pair"],
    "confidence": "high"
  },
  {
    "factor_index": 1,
    "axis_name": "Mixed",
    "summary": "no clean axis — see description",
    "description": "Top-positive items span cautious disclosure (three Likert items about flagging caveats) and bluntness (two items about stating flaws directly), which point in opposite behavioural directions. Top-negative items are mostly low-communality trait_mcq items about unrelated domains. No single direction organises the factor; likely a mix of residual variance the rotation didn't cleanly absorb into F0 or F2.",
    "positive_pole": "",
    "negative_pole": "",
    "dominant_item_types": ["likert", "trait_mcq"],
    "confidence": "unlabelable"
  }
]
```
