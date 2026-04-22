# Stage-3 FA methodology follow-ups (from `FA_REVIEW_HANDOFF.md`)

Notes only — nothing in this file is implemented yet. The upstream
`sid/external-rollouts` branch added eight methodology improvements to
`scripts_dev/psychometric_assessment/external_rollout_analysis.py`
during a scientific review. The shared-code (`src_dev/`) fixes from
that review already apply to our branch automatically via the rebase
(encoding v3/v4 rebuild, signed Tucker's φ, Cronbach's α exported from
`src_dev/factor_analysis/reliability.py`, etc.).

The methodology items in this file are **script-level** — they live in
the orchestrator / analysis driver and would need to be ported into
our internal-rollout path (`scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py`
and any downstream analysis scripts) before they take effect on a
`trait_ocean_natural_v1` Stage-3 run.

Recorded here so we don't lose track when we get to Stage 3 on the
Qwen2.5 / Llama × `trait_ocean_natural_v1` matrices.

---

## Priority summary

| # | Item | Priority | Est. effort | Risk if skipped |
|---|------|----------|-------------|-----------------|
| 1 | Parallel analysis for `n_factors` | **P1** | ~60 LOC self-contained | High — `n_factors` is the single biggest researcher degree of freedom in the whole pipeline; without a principled reference the choice is arbitrary |
| 2 | OCEAN / trait alignment check | **P1** | ~30 LOC + plot call | High for interpretability — tells us whether the extracted factors actually map onto OCEAN traits or something else |
| 3 | `n_factors` robustness sweep | **P2** | ~50 LOC | Medium — without this we can't defend that conclusions don't depend on the specific n |
| 4 | Full-pairwise Tucker's + per-source replicability | **P2** | ~80 LOC | Medium — only relevant once we have ≥3 groups to compare (e.g. multiple rollout presets) |
| 5 | Cronbach's α per group × factor | **P2** | ~50 LOC | Medium — α is the standard internal-consistency measure; needed for psychometric defensibility |
| 6 | Per-group retention table | **P2** | ~40 LOC | Medium — only relevant when groups might lose rows at different rates (context-length filter, parse failures) |
| 7 | Oblique factor correlation matrix dump | **P3** | ~15 LOC | Low — nice-to-have diagnostic |
| 8 | Drop tautological residualised η² | **P3** | ~10 LOC deletion + replace with Tucker's |φ| baseline-vs-residualised | Low — currently producing a meaningless "always-zero" number |
| 9 | torch seeding in analysis scripts | **P3** | 4 LOC | Low — not relevant for FA itself; matters for any downstream torch-RNG work |

My default when we get to Stage 3 would be: do **P1** items before calling any result final, add **P2** items for the write-up, and skip **P3** unless something specific comes up.

---

## 1. Parallel analysis for `n_factors` (commit `d324e94`)

**What.** Horn's parallel analysis recommends `n` by fitting FA on the
real data and on permuted / random-reference matrices, keeping only
eigenvalues that exceed the null reference at a specified percentile.
`src_dev.factor_analysis.parallel_analysis.parallel_analysis`
already exists — the external script wraps it in `_run_parallel_analysis`
and adds a Kaiser scree plot with the configured `n` marked.

**Why it matters for us.** The orchestrator currently hard-codes
`FA_N_FACTORS_OVERRIDE = 4`. Whether the true number is 3, 4, or 5
changes every downstream result. A principled recommendation protects
against confirmation-biased `n` choice and is expected by reviewers.

**Implementation notes.**
- Call once at the start of analysis on the baseline-preprocessed matrix.
- Permutation method (not resampling) for ordinal Likert data.
- Warn loudly if recommended `n` disagrees with configured by >2.
- Save scree plot alongside loadings.

---

## 2. OCEAN / trait alignment check (commit `a7de72d`)

**What.** For a questionnaire with a known `primary_dimension` per item
(both `v5` Likert and our `trait_ocean_natural_v1` items have this),
compute per-factor statistics about which trait's items load highest on
each factor. Outputs include factor-winner (dominant trait + share of
top-loading items), mean |loading| per trait, and signed-loading counts.

Support lives in `src_dev.factor_analysis.trait_alignment.compute_factor_trait_alignment`
and `trait_alignment.plot_all_alignment`. The external script wraps
this in `_trait_alignment_rows`.

**Why it matters for us.** This is *the* validity check for our
hypothesis that the extracted factors recover OCEAN structure. Without
it we can only say "there are N factors" — with it we can say "Factor 1
is 70% Openness items with consistent signs, Factor 2 is 55% Neuroticism
items but with signs flipped, etc.". This shapes the entire interpretation.

**Implementation notes.**
- Requires `primary_dimension` field on column defs (already populated
  for our trait_ocean_natural_v1 items).
- Flatten per-(factor × trait) to a CSV with `top_k_count`,
  `signed_loading_count`, `mean_abs_loading`, `mean_signed_loading`,
  `factor_winner`.
- Plotting via `trait_alignment.plot_all_alignment`.

---

## 3. `n_factors` robustness sweep (commit `d324e94`)

**What.** Re-fit the baseline (and optionally residualised) FA at
`{n−1, n, n+1, n+2}` and show the key summaries across these values —
max baseline η², min/median baseline→residualised |φ|, etc.

**Why it matters for us.** Parallel analysis recommends *one* `n`, but
conclusions should be robust to small perturbations. Presenting the
sweep in a table demonstrates robustness (or flags a finding that only
holds at a specific `n`, which is a red flag).

**Implementation notes.**
- Lightweight — only refits baseline + residualised, skips the expensive
  per-group Tucker's.
- Skip gracefully when `n` exceeds `n_columns_after_preprocessing`.

---

## 4. Full-pairwise Tucker's φ + per-source replicability (commit `c6f87ce`)

**What.** Compute |φ| + signed alignment for every ordered (source,
target) group pair, then aggregate per (source × factor) with mean /
min / max |φ| and sign-agreement fraction.

**Why it matters for us.** Only relevant once we have ≥3 groups to
compare. The immediate use-case is Qwen2.5 × Llama cross-model
Tucker's, which is just 2 groups — this is symmetric, full-pairwise
adds nothing. But if we later add more models / more rollout presets
to the comparison, this removes the arbitrariness of picking an
"anchor" group. The signed-φ support from commit `c6f87ce` is already
in `src_dev/psychometric/tucker_congruence.py` — just needs the driver
loop.

**Implementation notes.**
- Full-pairwise = O(K²) FA fits, cheap for small K but non-trivial for
  large K.
- Default single-anchor summary is still produced as a derived view.

---

## 5. Cronbach's α per group × factor (commit `a7de72d`)

**What.** For each factor, pick items with |loading| ≥ 0.4, sign-orient
by loading signs, compute α pooled and per group.
`src_dev.factor_analysis.reliability.cronbach_alpha` is already
exported; `classify_alpha` returns conventional bins.

**Why it matters for us.** Factor analysis tells you items cluster;
α tells you whether they mutually predict each other well enough to
*treat the cluster as a single construct*. Low α on a factor means the
items don't cohere despite loading together — worth flagging. If
per-group α is much lower than pooled α, the factor's reliability
depends on pooling across groups, which is another finding to flag.

**Implementation notes.**
- Straightforward call per (group, factor).
- Threshold convention: α ≥ 0.7 "acceptable", ≥ 0.8 "good".

---

## 6. Per-group retention table (commit `d324e94`)

**What.** Track row counts per group at each filtering stage:
Stage 1 count → Stage 2 context-filter count → parse-success count →
combine-intersection count → FA final-row count.

**Why it matters for us.** Our `B` rollout preset produces a single
persona pool, so per-group retention is less relevant than for the
external-rollout multi-dataset setting. But if we ever split the
internal rollouts by input-group-id or prompt-archetype, per-group
retention bias becomes invisible in the aggregate shape counts, and
that bias would confound cross-group FA comparisons.

**Implementation notes.**
- Only matters when we have groups within the matrix.
- Print + CSV output per external script's `_per_preset_retention_table`.

---

## 7. Oblique factor correlation matrix dump (commit `d324e94`)

**What.** Extract and save `factor_correlation_matrix` from
`run_factor_analysis` result dict (already populated for oblique
rotations like `oblimin`).

**Why it matters for us.** Correlations > ~0.3 between factors change
the interpretation of any "pure" factor. "Pure Openness" loses meaning
if Openness and Conscientiousness correlate at r=0.5 in the oblique
solution. Dumping this as a CSV costs ~15 lines.

**Implementation notes.**
- Extend the result-carrying dataclass to include
  `factor_correlation_matrix: np.ndarray | None`.
- Write one CSV per (flavour × rotation) when non-None.

---

## 8. Residualised η² is tautologically zero (commit `7791c4f`)

**What.** Per-group mean subtraction forces every factor-score
column's per-group mean to zero by construction, so `SS_between = 0`
and `η² = 0` deterministically. The existing
"baseline-η² vs residualised-η² decomposition" was not a variance
decomposition — it was a sanity check of the subtraction arithmetic.

**Why it matters for us.** Only relevant if our analysis uses
`preprocess_response_matrix(do_residualize=True, ...)` and then
computes η² on the residualised scores. If we do, the "residualised η²"
column in any output CSV is a tautology and should be removed.

**Recommended replacement.** Use Tucker's |φ| between baseline and
residualised loadings to answer "does the factor structure survive
per-group mean removal?". External script's
`_baseline_vs_residualised_alignment` is ~10 lines and already uses
the shared `FactorAlignment` type.

---

## 9. torch seeding in analysis scripts (commit `a7de72d`)

**What.** The four-line snippet from CLAUDE.md:

```python
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass
```

**Why it matters for us.** vLLM inference is already deterministic via
`temperature=0.0` for greedy paths, but our pipeline isn't purely
greedy (logprob sampling at temperature 1.0 is used). Any downstream
torch-RNG usage (embedding back-projection, trait-alignment plotting,
HF model init) is not seeded by default.

**Action.** Add the block at the top of any analysis script that
imports torch directly or transitively.

---

## Not in scope for this file

The following handoff items are **already applied via the rebase** —
no action required from us:

- Encoding v3/v4 matrix rebuild (reverse-keying fix + `min_choice_mass`
  gate) — automatic on first Stage-2 call post-rebase.
- Signed Tucker's φ (`signed=True` kwarg, `FactorAlignment.sign`) —
  available now, default behaviour unchanged.
- Cronbach's α / `classify_alpha` exports in
  `src_dev/factor_analysis/reliability.py` — just need to be called.
- Encoding-version cache gate is now unconditional (was trait-mcq-only).
- Filter-config validation on Stage-1 — external-only, no impact on us.

See `FA_REVIEW_HANDOFF.md` §3 for the full back-compat matrix.
