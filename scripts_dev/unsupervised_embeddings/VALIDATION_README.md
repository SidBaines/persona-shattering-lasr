# Factor-Analysis Validation Runbook

Companion to `psychometric_rollout_fa.py`. Covers the validation campaign
defined in `/root/.claude/plans/resilient-leaping-mango.md`: how to run the
per-run checks, trigger the config sweeps, and assemble the final cross-run
HTML report.

## What gets validated

Each pipeline invocation runs the set of tests enabled in
`VALIDATION_TESTS_TO_RUN` (script line ~248). Every test writes a JSON + plot
under `<run_dir>/validation/`:

| Test name (in `VALIDATION_TESTS_TO_RUN`) | Output dir | Pass criterion |
| --- | --- | --- |
| `shuffle_control` | `validation/shuffle_test.json` | 0 factors on shuffled data |
| `item_holdout` | `validation/predictivity.json` | >50% of held-out items FDR-significant |
| `stability_icc` | `validation/stability/<fa_key>/stability.json` | mean ICC(1) ≥ `pass_threshold_mean_icc1` (default 0.20) |
| `variance_decomp` | `validation/variance_decomp/` | no factor with archetype‖scenario η² ≥ 0.30 |
| `trait_convergence` | `validation/trait_convergence/` | ≥ 3 of 5 OCEAN traits hit \|ρ\| ≥ 0.30 |
| `stability_sweep_random50` | `validation/stability_sweep_random_50.*` | median \|φ\| ≥ 0.80 over 10 splits |
| `stability_sweep_loao` | `validation/stability_sweep_loao.*` | median \|φ\| ≥ 0.80 across archetype drops |
| `stability_sweep_loso_top10` | `validation/stability_sweep_loso.*` | median \|φ\| ≥ 0.80 across top-10 scenario drops |
| `k_sensitivity` | `validation/k_sensitivity.*` | ≤ 1 factor classified "split" at k+1 |
| `persona_item_cv` | `validation/persona_item_cv.*` | mean R² gain ≥ 2× shuffle baseline |

Tuning constants for each test live immediately below `VALIDATION_TESTS_TO_RUN`
(`STABILITY_SWEEP_*`, `TRAIT_CONVERGENCE_*`, `PERSONA_ITEM_CV_*`, etc.).

## Single-run workflow

```bash
uv run --quiet python scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py
```

- Stage 1 (rollouts) is cached in HuggingFace — it's only re-run if `ROLLOUT_RUN_ID`
  resolves to a fresh path.
- Stages `factor_analysis`, `labeling`, `validation` re-run cheaply.
- To skip expensive validation passes for a quick iteration, trim
  `VALIDATION_TESTS_TO_RUN` (e.g. drop the sweeps).

## Config sweep runbook

These sweeps reuse the cached Stage 1 rollouts unless you change the rollout
configuration. Each sweep = one more run directory under `scratch/psychometric_fa/`.

### Residualization on/off

Already happens inside a single run — `RESIDUALIZE_OPTIONS = [False, True]`
produces two FA variants (`raw/` and `residualized/`). Cross-variant φ can be
computed by pointing `build_report` at the same run dir twice with different
anchor selections, or by comparing the two loading npz files directly using
`src_dev.factor_analysis.congruence.compare_solutions`.

### Logprob vs. argmax (trait_mcq)

Flip `QUESTIONNAIRE_USE_LOGPROBS` (script line ~187) between `True` and
`False`, each time re-running from Stage 2 (`questionnaire`). Stage 1 rollouts
are reused. Compare the resulting runs via `build_report`.

### Phrasing replication

Edit `QUESTIONNAIRE_PHRASING` (script line ~157) to each of
`"direct"`, `"natural"`, `"contextual"` and re-run. Stage 2 re-runs; Stage 1
is reused.

### Cross-model replication

Edit `ASSISTANT_MODEL` (script line ~118). A comment block above that line
lists candidate models. This triggers a fresh Stage 1 rollout — same cost as
the original full run. The plan's default is Qwen only, with additional
models (Mistral, Llama-70B) deferred on budget.

## Building the cross-run report

```bash
uv run --quiet python -m src_dev.factor_analysis.validation_report \
    --runs scratch/psychometric_fa/<run_id_1>/ \
           scratch/psychometric_fa/<run_id_2>/ \
           scratch/psychometric_fa/<run_id_3>/ \
    --out  scratch/psychometric_fa/_reports/$(date +%Y%m%d_%H%M)/validation_report.html
```

The report contains:

- a summary table of every pass/fail check, one row per test × run;
- inline-base64 PNGs of every validation plot;
- pairwise Tucker φ (`compare_solutions`) across the run anchors, with
  Procrustes alignment when k matches and Hungarian matching otherwise.

Item axes are aligned via the `col_id`s written alongside the FA npz. Runs
with non-overlapping item sets fall back to row-order alignment when the
counts match.

## On a post-freeze confirmatory sample

EFA → CFA rigor requires a confirmatory sample collected *after* the factor
model is frozen. The current plan treats phrasing replication + cross-model
replication as the confirmation, and defers a fresh same-model sample unless
reviewers push. If a confirmatory sample becomes necessary, the steps are:

1. Freeze `FA_METHOD`, `FA_ROTATIONS[0]`, the chosen k, and the item set.
2. Bump `SEED` (e.g. to 43) and ideally swap the scenario pool to a held-out
   subset, then run Stage 1 + Stage 2 fresh.
3. Fit FA with the same k/rotation and run `src_dev.factor_analysis.congruence.compare_solutions`
   against the frozen loadings. Pass criterion: median |φ| ≥ 0.85 across all factors.

That analysis does not yet have a script — create one under
`scripts_dev/unsupervised_embeddings/` that loads the frozen npz, fits on the
new sample, calls `compare_solutions`, and emits a two-panel loading heatmap.

## Where things live

| Concern | Location |
| --- | --- |
| Per-test pure-function implementations | `src_dev/factor_analysis/validation.py`, `cross_validation.py`, `trait_convergence.py`, `congruence.py` |
| Pipeline wrappers (script-side) | `scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py::_validation_*` |
| Dispatcher | `psychometric_rollout_fa.py::run_stage_validation` |
| Cross-run report | `src_dev/factor_analysis/validation_report.py` |
| FA persistence (for anchor loadings) | `src_dev/factor_analysis/persistence.py` |
| Plan (source of truth) | `/root/.claude/plans/resilient-leaping-mango.md` |
