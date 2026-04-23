# FA pipeline review — handoff for the qwen25 agent

This branch (`sid/external-rollouts`) went through a scientific review of
`scripts_dev/psychometric_assessment/external_rollout_analysis.py` and its
supporting pipeline. Seven commits landed as a result. Some change shared
infrastructure that the `sid/qwen25-b-analysis` branch (working on internal
rollouts) will inherit on rebase; others are specific to the external
script but describe methodology improvements worth replicating for
internal-rollout FA too.

This document is organised as:

1. **TL;DR** — five-line summary.
2. **Rebase notes** — what to expect when pulling this branch in.
3. **Shared-infrastructure changes** — code that now behaves
   differently; behavioural contract + any action required.
4. **Methodology improvements worth replicating** — changes made in the
   external script that would strengthen internal-rollout FA too.
5. **Open follow-ups** — known gaps that neither branch fixed yet.
6. **Cache / reproducibility notes** — encoding-version bumps and
   their rebuild cost.

Commits referenced are on `sid/external-rollouts` between `0eaac99`
(shared base) and `HEAD`:

```
ac2d264  apply min_choice_mass noise gate (encoding v4)
a7de72d  add Cronbach's α + factor-trait alignment (OCEAN); seed torch
c6f87ce  full pairwise Tucker's φ + signed factor alignment
d324e94  add parallel-analysis, retention, n-factors sweep, oblique factor corr
1d75a33  validate cached filter_config on Stage-1 cache hit
0fe60a6  fix logprob Likert reverse-keying (encoding v3)
7791c4f  fix tautological residualised η²; drop oasst preset
```

---

## 1. TL;DR

- **Two silent-data-corruption bugs fixed** in shared code: logprob-scored
  Likert reverse-keying was using the observed rather than nominal scale
  (encoding v3); `min_choice_mass` config field was declared but never
  applied to the FA matrix (encoding v4).
- **Encoding version bumped 2 → 4** in `src_dev/psychometric/response_encoding.py`.
  Every cached response matrix will rebuild once from
  `raw_responses.jsonl` — **no re-inference needed** — on first run after
  rebase.
- **Tucker's φ now returns signed values** when you ask (`signed=True`
  kwarg) and `FactorAlignment.sign` records per-match polarity. Default
  behaviour (`|φ|`, no sign) is unchanged.
- **Cronbach's α + `classify_alpha` added** to
  `src_dev/factor_analysis/reliability.py`. Pure addition, no break.
- **Filter-config validation added to Stage-1 external ingest**. Does
  not affect internal-rollout ingest — the two codepaths are disjoint
  inside `_rollout_run_id`.

---

## 2. Rebase notes

**Expected clean rebase.** All shared-code changes are either new
functions, new kwargs with backward-compatible defaults, or internal
behaviour changes inside functions whose signatures didn't move. No
call sites on the `sid/qwen25-b-analysis` branch should have to be
updated to make anything compile.

**One first-run cost.** The encoding-version bump forces each cached
`response_matrix.npy` to be rebuilt from its sibling
`raw_responses.jsonl`. The rebuild re-applies the fixed encoder;
there is no inference. If `raw_responses.jsonl` is absent (very old
cache), the cache is regenerated — that *does* cost inference, but
should not happen in practice for any cache produced in the last few
weeks.

**Behavioural change you should be aware of.** If your internal-rollout
analysis uses `use_logprobs=True` on a Likert questionnaire with
reverse-keyed items (v5 is the canonical one), the v3 fix will change
the matrix cell values for those items. The previous values were wrong
and the new ones are correct — but any downstream numerical result
cached from a v2 matrix (e.g. saved loadings) should be recomputed.

**Files touched in `src/` and `src_dev/`:**

| File | What changed | Back-compat |
|------|--------------|-------------|
| `src_dev/psychometric/response_encoding.py` | Reverse-keying fix (v3), min-mass gate (v4), `likert_scale` kwarg added to `fill_matrix_from_choice` / `record_response` | Kwarg has default `5`; new call sites not required. |
| `src_dev/psychometric/questionnaire_inference.py` | Threads `cfg.likert_scale` through; applies `cfg.min_choice_mass` on live + rebuild paths | No signature changes. |
| `src_dev/psychometric/stages/questionnaire.py` | Encoding-version check is now unconditional (was trait-mcq-only) | No signature changes. |
| `src_dev/psychometric/stages/external_rollouts.py` | Filter-config validation on cache hit | External-rollouts only. Internal-rollouts (`stages/rollouts.py`) untouched. |
| `src_dev/psychometric/tucker_congruence.py` | `tucker_phi_matrix(..., signed=False)` kwarg; `FactorAlignment.sign` (default +1); `summarise_alignment` includes sign | Defaults preserve prior output. |
| `src_dev/factor_analysis/reliability.py` | New `cronbach_alpha` / `classify_alpha` | Pure addition. |

No changes to `src/` (stable layer). No changes to
`src_dev/datasets/`, `src_dev/factor_analysis/factor_analysis.py`,
`src_dev/factor_analysis/preprocessing.py`,
`src_dev/factor_analysis/interpretation.py`,
`src_dev/factor_analysis/parallel_analysis.py`,
`src_dev/factor_analysis/trait_alignment.py`,
or `src_dev/psychometric/combine.py`.

---

## 3. Shared-infrastructure changes — details

### 3.1 Logprob Likert reverse-keying (commit `0fe60a6`, encoding v3)

**Old behaviour.** For reverse-keyed Likert items under `use_logprobs=True`,
the reversal was computed as `(scale + 1) − score` where
`scale = max(digits present in top-k)`. When the model's top-20 logprobs
did not span the full 1..5 scale (e.g. a strongly-disagreeing response
with all mass on digits {1, 2}), `scale` became the observed max rather
than the nominal scale, and the reversal produced the wrong value.

Example: `P(1)=0.9, P(2)=0.1`, reverse-keyed. Previous code:
`(2+1) − 1.1 = 1.9`. Correct: `(5+1) − 1.1 = 4.9`. The semantic
"strongly disagree, so in truth strongly agree on the underlying
trait" was flipped into "strongly disagree on the underlying trait".

**New behaviour.** `fill_matrix_from_choice` and `record_response`
now take a `likert_scale: int = 5` kwarg. Both the logprob path and
the greedy fallback use `(likert_scale + 1) − score`.
`questionnaire_inference` passes `cfg.likert_scale` through at all
call sites.

**Impact on the internal-rollout branch.** Any v5 Likert cache
produced prior to this fix has wrong values for reverse-keyed items
(roughly half of v5). Rebuilding from `raw_responses.jsonl` (which
happens automatically on the first Stage-2 call after rebase)
produces correct values. Downstream loadings, scores, and any
saved FA artefacts should be treated as stale.

**Action on rebase.** None required. Rebuild is automatic.

### 3.2 `min_choice_mass` noise gate (commit `ac2d264`, encoding v4)

**Old behaviour.** `QuestionnaireStageConfig.min_choice_mass` existed
as a field but was only used inside the separate `trait_scoring`
stage — it was silently ignored by `questionnaire_inference`. Cells
where the model's top-20 logprobs barely contained any choice-token
mass (e.g. the model ignored the answer template and emitted prose)
were scored via whatever sparse digit/letter probability happened to
leak into the top-k — essentially noise.

**New behaviour.** The gate is now applied on both the live
inference path and the rebuild-from-raw path:

- Live: if `choice_mass < cfg.min_choice_mass`, record the cell as a
  parse failure (NaN) rather than filling it. The raw-log line
  carries `"below_mass_gate": true` + a `reason` string so a later
  rebuild with a lower threshold can re-admit the cell without
  re-running inference.
- Rebuild: same gate applies when re-encoding a cached
  `raw_responses.jsonl`. A missing `choice_mass` field (legacy log
  lines) is treated as "no gate applies".

**Impact on the internal-rollout branch.** Only visible when
`min_choice_mass > 0`. The external-rollout analysis now sets
`QUESTIONNAIRE_MIN_CHOICE_MASS = 0.3`; the internal-rollout analysis
still defaults to `0.0` via
`QuestionnaireStageConfig.min_choice_mass: float = 0.0`. No cells are
filtered until you set it positive on your end.

**If you want the filter on your branch**: set your equivalent of
`QUESTIONNAIRE_MIN_CHOICE_MASS` to `0.3` (or experiment) before
running. On first run, the v3 → v4 rebuild re-filters the cached
matrix. On subsequent config changes (e.g. 0.3 → 0.5), the matrix is
**not** auto-invalidated — manually delete the stale
`response_matrix.npy` for the affected run. (A future commit could
hash the threshold into `encoding_version.json` to automate this; see
open follow-ups below.)

**Action on rebase.** None required. If you don't change
`min_choice_mass`, behaviour is identical to v3.

### 3.3 Signed Tucker's φ + `FactorAlignment.sign` (commit `c6f87ce`)

**Old behaviour.** `tucker_phi_matrix` returned `|φ|` in [0, 1]. A
factor matched between two solutions with `|φ|≈1.0` might have
opposite polarity (one solution's "Conscientiousness" could be the
negation of the other's), but that was invisible in the return value.

**New behaviour.**

- `tucker_phi_matrix(L_a, L_b, signed=False)` — unchanged default.
- `tucker_phi_matrix(L_a, L_b, signed=True)` — returns signed φ in
  [-1, 1].
- `FactorAlignment` gains `sign: int` (default +1). `align_factors`
  computes it from the signed φ at the chosen Hungarian assignment:
  +1 if the matched pair agree on polarity, -1 if one is the
  polarity-inverse, 0 when no match was assigned (NaN handling).
- `summarise_alignment`'s per-match dicts now include `"sign"`.

**Impact on the internal-rollout branch.** Zero — the default
`signed=False` and the new `sign` field with default +1 preserve all
existing behaviour. Use the new kwarg / field if you ever pool
factor scores across presets or rotations, since |φ|=1.0 with
sign=-1 means "same structure, opposite pole" and score pooling
without sign-alignment would cancel signal.

**Action on rebase.** None required.

### 3.4 Encoding-version cache gate (commit `0fe60a6`)

**Old behaviour.** `stages/questionnaire.py` skipped the version
check unless the questionnaire contained a `trait_mcq` block. This
was correct for the v1 → v2 bump (which only touched trait_mcq), but
wrong for the v2 → v3 bump (Likert) and any future block-agnostic
change.

**New behaviour.** The check fires unconditionally. Any mismatch
triggers a rebuild from `raw_responses.jsonl` (no re-inference).

**Impact on the internal-rollout branch.** Every cached Likert-only
or fc-only matrix rebuilds once on first run after rebase. Cost is a
few seconds per cache — no inference.

**Action on rebase.** None required.

### 3.5 Filter-config validation on Stage-1 cache hit (commit `1d75a33`)

**Scope.** External-rollouts only. `_rollout_run_id` branches on
`key in EXTERNAL_ROLLOUT_PRESETS` and internal presets go through a
completely separate codepath (generation-based run-id, no
`filter_config` concept). Internal-rollout ingest is untouched.

**What it does.** On external-rollout cache hit (local or
HF-hydrated), reads the first sample's `source_info` from
`sample_inputs.jsonl` and compares `filter_config` against the
current preset's merged `filter_config` (with `min_assistant_turns`
folded in). Mismatch → hard `RuntimeError` with a remediation hint.
Motivation: `_rollout_run_id` currently does NOT hash
`filter_config` or `min_assistant_turns`, so editing a preset's
filter without also bumping `filter_tag` would silently reuse an
old cache under the same run-id. Hashing the full filter into the
run-id would invalidate every Stage-2 cache (questionnaire run-id
embeds rollout run-id → re-inference), which was too expensive in
this pass, so this commit adds validation instead.

**Impact on the internal-rollout branch.** None.

**Action on rebase.** None required.

### 3.6 Cronbach's α (commit `a7de72d`)

**Addition.** `src_dev/factor_analysis/reliability.py` now exports
`cronbach_alpha(item_responses, loading_signs=None) -> float` and
`classify_alpha(alpha) -> str`.

- `cronbach_alpha` takes a complete-case response matrix. Optional
  `loading_signs` flip items whose factor loading is negative, so
  summed scores point in one direction; required when computing α
  for a factor with mixed-sign loadings.
- `classify_alpha` applies the conventional bins (excellent / good /
  acceptable / questionable / poor / redundant).

**Impact on the internal-rollout branch.** None — pure addition. Use
if you want internal-consistency reliability per factor on your end.

---

## 4. Methodology improvements worth replicating

The external script now produces a set of diagnostics that would make
the internal-rollout FA more defensible too. Each of these is a
script-level change; no shared-infrastructure contract depends on
them. Adapt for your script if applicable.

### 4.1 Residualised η² is tautologically zero (commit `7791c4f`)

**Scope.** Relevant if your analysis ever calls
`preprocess_response_matrix(..., do_residualize=True, ...)` and then
computes η² (via `prompt_effects`) on the residualised factor scores.

**Why.** Per-group mean subtraction forces the per-group mean of
every column — and hence every linear combination of columns
(= every factor score) — to zero. So `SS_between` is identically
zero and `η² = 0` regardless of the factor structure. The previous
"decomposition" (baseline η² vs residualised η² side-by-side) was
a sanity check of the subtraction arithmetic, not a variance
decomposition.

**Recommendation.** If you want a meaningful "does the factor
structure survive per-group mean removal?" test, use Tucker's |φ|
between baseline and residualised loadings instead. The external
script's `_baseline_vs_residualised_alignment` is ~10 lines and can
be copied directly; it returns `list[FactorAlignment]` with per-factor
|φ| + sign.

### 4.2 Parallel analysis + Kaiser scree for `n_factors` (commit `d324e94`)

**Scope.** Any FA step where `n_factors` is a hard-coded magic number.

**Why.** `n_factors` drives every downstream result. Choosing it
without a principled reference is a researcher degree of freedom;
Horn's parallel analysis (with the permutation method for ordinal
Likert data) gives a defensible null-reference recommendation.

**Recommendation.** Call
`src_dev.factor_analysis.parallel_analysis.parallel_analysis` on the
baseline-preprocessed matrix once at the start of the analysis.
Compare against the configured n. Warn loudly if they disagree by
more than ~2. Save a scree plot with the recommendation and
`configured n` marked. The external script's `_run_parallel_analysis`
+ `_plot_scree` is ~60 lines of self-contained code.

### 4.3 Per-group retention table (commit `d324e94`)

**Scope.** Any multi-group FA where the groups might have systematically
different retention rates at the filtering stages (context-length filter,
parse-failure rate, complete-case deletion). In internal-rollout terms,
this could be per-input-group-id or per-prompt-archetype retention.

**Why.** The combined-matrix shape only reports pooled row counts.
If one group loses 30% of its rows to complete-case deletion and
another loses 0%, per-group FA comparisons are not apples-to-apples,
and the bias is invisible from the final outputs.

**Recommendation.** Build a `_per_group_retention_table` that tracks
row counts per group at each filtering stage:

- Stage 1 count (from `canonical_samples.jsonl` line count).
- Stage 2 context-filter count (from `pair_data[(r, q)][0].shape[0]`).
- Parse-success count (rows with no NaN in the matrix).
- Combine-intersection count (rows in combined metadata with
  `group_field == group`).
- FA final-row count (rows in preprocessed metadata with
  `group_field == group`).

Print + CSV. See `_per_preset_retention_table` in the external script.

### 4.4 `n_factors` robustness sweep (commit `d324e94`)

**Scope.** Any analysis whose top-line claim depends on a specific
`n_factors` value.

**Why.** Even when parallel analysis recommends n, conclusions should
be robust to small perturbations. Re-fitting at
{n−1, n, n+1, n+2} and showing the key summaries (max baseline η²,
min/median baseline→residualised |φ|) across those values demonstrates
robustness.

**Recommendation.** Add a lightweight sweep loop that refits only
baseline + residualised (no per-group Tucker — that's the expensive
part and unnecessary for a robustness check). ~50 lines. Skip
gracefully when n exceeds `n_columns_after_preprocessing`.

### 4.5 Oblique factor correlation matrix dump (commit `d324e94`)

**Scope.** Any oblimin / promax rotation.

**Why.** `factor_analyzer` computes the inter-factor correlation
matrix for oblique rotations; the existing `run_factor_analysis`
wrapper returns it in the result dict (`factor_correlation_matrix`).
Previously this was thrown away. Correlations > ~0.3 change the
interpretation of any "pure" factor ("pure openness" loses meaning
if openness and conscientiousness correlate at r=0.5 in the fitted
solution).

**Recommendation.** Extend your analogue of `FaFitResult` to carry
`factor_correlation_matrix: np.ndarray | None` and dump one CSV per
flavour × rotation when it's non-None. ~15 lines.

### 4.6 Full-pairwise Tucker's + per-source replicability (commit `c6f87ce`)

**Scope.** Any cross-group FA comparison that currently picks an
"anchor" group and reports others relative to it.

**Why.** Single-anchor designs make alignment results sensitive to
group ordering (which group is "first" changes the anchor, which
changes factor numbering, which propagates to every plot and CSV).
Full-pairwise removes the arbitrariness.

**Recommendation.** Compute |φ| + alignment for every ordered
(source, target) group pair, then aggregate per (source × factor):
mean / min / max |φ| and sign-agreement fraction across all other
groups. See `_full_pairwise_tucker` + `_per_source_factor_replicability`
in the external script. The single-anchor summary can still be
produced as a derived view per source.

### 4.7 Cronbach's α per group × factor (commit `a7de72d`)

**Scope.** Any claim that a factor is a "reliable measurement" of
something.

**Why.** Factor analysis tells you items cluster together; internal
consistency (α) tells you whether they mutually predict each other
well enough to treat the cluster as a single construct. α is table
stakes for psychometric defensibility.

**Recommendation.** For each factor, pick items with |loading| ≥
0.4, sign-orient by the loading signs, compute α pooled and per
group. If per-group α is much lower than pooled, the factor's
reliability depends on pooling across groups — worth flagging. See
`_factor_reliability` in the external script (~50 lines).

### 4.8 OCEAN / trait alignment check (commit `a7de72d`)

**Scope.** Any FA run on a questionnaire with a known
`primary_dimension` per item (both `v5` Likert and
`trait_ocean_v1_nolead` trait_mcq meet this).

**Why.** For a questionnaire designed to measure OCEAN, the
canonical validity check is "does the factor-to-item mapping
recover the expected trait structure?". If Factor 1 loads mostly
on conscientiousness items, call it Conscientiousness; if it
loads evenly across all traits, it's not a trait factor at all.

**Recommendation.**
`src_dev.factor_analysis.trait_alignment.compute_factor_trait_alignment`
takes `(loadings, item_dims)` — use each item's `dimension` field
from the column defs. Flatten the result to per-(factor × trait)
CSV rows with top-K count, signed-loading count, mean |loading|,
mean signed loading, and a `factor_winner` column (the dominant
trait + its share). See `_trait_alignment_rows` in the external
script (~30 lines) plus the plotting via
`trait_alignment.plot_all_alignment`.

### 4.9 torch seeding (commit `a7de72d`)

**Scope.** Any analysis script.

**Why.** CLAUDE.md requires all four RNGs be seeded at the top of
experiment scripts. vLLM inference is already deterministic via
`temperature=0.0`, but any downstream torch RNG use (embedding
back-projection, trait-alignment plotting, HF model init) is not.

**Recommendation.** Add the four-line block:

```python
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass
```

---

## 5. Open follow-ups (unfixed; relevant to both branches)

These were in the original review list but are more substantive
changes that I didn't land in this pass. Calling out so both
branches can track them.

- **P1.6 — Preset identity / rollout source confound.** In the
  external analysis, "preset" conflates model identity, rollout
  source dataset, prompt distribution, and context-length cap. No
  amount of post-hoc decomposition disentangles these; the fix is
  experimental design (same model × different rollout sources).
  In internal-rollout terms, this probably maps to "does the
  rollout-generation prompt distribution confound the same-model
  FA?" — worth writing up as a caveat either way.
- **P2.2 — cache's encoding-params check.** Changing
  `min_choice_mass` (or any other config-driven encoder parameter)
  does not auto-invalidate cached matrices. Currently the
  expectation is manual deletion. A future commit could hash the
  relevant params into `encoding_version.json` and compare on
  cache-hit. ~40 lines.
- **P2.3 — `MIN_ITEM_VARIANCE` sensitivity.** The 0.1 relative-
  variance cutoff is a magic number. Could be swept in the
  n-factors robustness loop (same infrastructure).
- **Cache-invalidation on `filter_config` drift.** The current
  Stage-1 validation is external-only, and it fails loudly rather
  than re-ingesting. If you want drift-safe caches without manual
  intervention, hash `filter_config` into the run-id. That's the
  bigger fix and costs a Stage-2 re-inference on every preset
  you've ever cached, which wasn't worth it in this pass.
- **Rotation configuration coverage.** The external script runs
  both `oblimin` and `varimax`. Internal-rollout analyses that
  don't already do both should — if a finding only holds under
  one rotation, that's a rotation-dependence red flag.

---

## 6. Cache / reproducibility notes

**Encoding-version rebuilds.** Running any Stage-2-backed analysis
for the first time after rebase will trigger a rebuild for every
cached `response_matrix.npy` whose `encoding_version.json` records
v3 or earlier (or is missing). The rebuild reads the sibling
`raw_responses.jsonl`, applies the current encoder (reverse-keying
fix + min_choice_mass gate), and writes out a fresh matrix. No
model calls, no HF re-uploads — the corrected matrix is then
uploaded to HF to replace the stale one. Wall clock: seconds per
cache.

**Post-rebuild cache compatibility.** Matrices produced under v4
are stable until the next encoding-version bump. The current v4
reflects:

1. trait_mcq uses `answer_mapping` (since v2).
2. Logprob-scored Likert reverse-keying uses the nominal scale
   (since v3).
3. `min_choice_mass` threshold applied (since v4).

**`raw_responses.jsonl` is the single source of truth.** Anything
you can reconstruct from raw logs alone (alternative encodings,
alternative thresholds, additional parsing experiments) does not
require re-inference. Protect it in your HF uploads.

**Branch-specific run-id collisions (still open).** `_rollout_run_id`
does not hash `filter_config` / `min_assistant_turns` on the
external-rollout path, so two different filter configs with the
same `(source, assistant_model, max_samples, seed, filter_tag)` share
a cache. The Stage-1 validation check fires on cache hit, preventing
silent drift, but does not automate re-ingest. For internal rollouts
this does not apply.
