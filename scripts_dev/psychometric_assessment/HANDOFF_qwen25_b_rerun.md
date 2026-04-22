# Handoff — Qwen2.5 × B × trait_mcq re-run with digit-aware parser

This doc brings an agent from zero to running the right commands. For
historical background on the 3-model Qwen2.5/Qwen3/Llama comparison
that set up this stream of work, see the older
`HANDOFF_qwen25_comparison.md` in the same directory — it has the
research framing, the first-pass results, and the choice-mass
diagnostic that motivated the parser fix. You don't need to re-read
it to execute this handoff, but the "Research goal" section is worth
skimming.

Branch to start from: `origin/sid/external-rollouts` (tip `b2c9970` as
of 2026-04-22). We've been working on `sid/qwen25-b-analysis`, which
is branched off that same commit and so far has **only this doc** on
top. Either branch is fine as a base — pick one and tell the user
which.

## TL;DR of the situation

- The B rollout cache (Llama-3.1-8B-Instruct × 2500 prompts × 1 rollout
  × 15 turns, scenarios_v2 + user-sim v6) is the primary instrument of
  study. All artifacts live on HF at
  `persona-shattering-lasr/psychometric-fa-runs`.
- Two questionnaires are administered on every rollout at turn N+1:
  `v5` (Likert 1–5) and `trait_ocean_v1` (ABCD MCQ, i.e. `trait_mcq`).
- We've already administered both questionnaires to Llama and to
  Qwen2.5-7B-Instruct and fit combined (v5+trait) factor analyses for
  each.
- **Problem:** the existing `trait_mcq` runs used v1 of the logprob
  parser, which softmax-renormalises only the A/B/C/D letter tokens.
  Qwen2.5 puts ~13% of its top-20 probability mass on digit tokens
  1/2/3/4 — the parser discarded that mass, which made the Qwen2.5
  trait_mcq distribution look spuriously bimodal and tanked the
  Llama↔Qwen2.5 Tucker's congruence on the trait_mcq block.
- **Fix:** commit `23a692f` on `sid/external-rollouts` added a
  digit-aware parser (v2) that treats digits 1..4 as aliases for
  letters A..D and sums linear probabilities. The same commit also
  now persists raw `top_logprobs` in `raw_responses.jsonl` so future
  parser changes can be applied offline without re-inference. Both
  changes are live on the starting branch.
- **Gap:** the old HF runs predate commit `23a692f`, so their
  `raw_responses.jsonl` does NOT contain `top_logprobs`. We can't
  re-parse offline. **We need a GPU re-run.**

## Choice-mass audit (2026-04-22, this session)

Run on the hydrated HF artifacts to decide which models actually
need re-running:

| Model × Questionnaire | median CM | p10 CM | frac cells < 0.95 |
|---|---|---|---|
| Llama × trait_mcq (v1 parser) | 0.999 | 0.995 | **0.07%** |
| Qwen2.5 × trait_mcq (v1 parser) | 0.968 | 0.562 | **43%** |
| Llama × v5-likert-lp | 0.999 | 0.991 | 1.3% |
| Qwen2.5 × v5-likert-lp | 1.000 | 1.000 | 0% |

Verdict:
- **Qwen2.5 × trait_mcq** is the one load-bearing re-run. 43% of its
  cells had < 0.95 letter-mass — exactly the regime the v2 parser was
  designed for.
- **Llama × trait_mcq** is fine on v1. Upper bound on v2's effect is
  ~0.2 % shift per cell. Default plan: leave Llama as-is.
- **Both models × v5-likert** are unaffected — v5 uses the digit-native
  Likert parser which wasn't touched by `23a692f`.

Reproduce the audit by running:

```bash
uv run python -m scripts_dev.psychometric_assessment.choice_mass_analysis
```

after hydrating the 4 relevant Stage-2 dirs locally (see "Quick-start"
below).

## The plan

### Minimum required
1. Re-run **Qwen2.5 × B × trait_ocean_v1** with parser v2 on a GPU.
   Produces HF run-id
   `questionnaire-rollouts-llama318binstruct-t1.0-15t-2500p-seed436-scenarios_v2-uprompt_v6-q_trait_ocean_v1-trait_mcq-direct-lp20-p2-qm_qwen257binstruct`.
2. Upload the new Stage-2 dir to HF under `runs/<that run-id>/`.
3. Re-fit the Qwen2.5 combined FA (v5 + trait_ocean_v1 with new
   trait_mcq matrix). Upload to
   `combined/combined-R[B]-Q[v5+trait_ocean_v1]-qm_qwen257binstruct-p2/`
   (or agree a naming scheme with the user — keeping the old v1-based
   combined dir alongside is probably useful for comparison).
4. Re-run cross-model Tucker's φ: Llama (v1 parser, existing combined)
   vs Qwen2.5 (v2 parser, new combined).
5. **Report back to the user** before doing anything beyond this. The
   prediction to check is:
    - Qwen2.5 trait_mcq bimodality softens toward a graded distribution.
    - Trait_mcq Tucker's φ moves from ~0.33 toward the ~0.65 v5-based
      plateau.
   If the data disagrees, **don't spiral into further experiments —
   flag the discrepancy and discuss.**

### Decision points (ask user or defer)

- **(A) Re-run Llama × trait_mcq for consistency?** Default: no. Will
  shift cells by ≤ 0.2% and costs one full vLLM session on Llama-3.1-8B
  (2500 personas × 100 items). Only do it if (1) the re-fit cross-model
  Tucker's φ misbehaves and we want to rule out parser mismatch, or
  (2) the user explicitly asks.
- **(B) Labelling stage?** The orchestrator has a
  `"labeling"` stage that LLM-labels each factor from its top-loading
  items. It's commented-out in `STAGES_TO_RUN`. User has said "maybe —
  we'll decide based on interim results." Don't enable unless asked.
- **(C) Qwen3-8B parity?** Out of scope. Stick with Llama + Qwen2.5.

### Success and failure modes
This is research, not a ticket. **Report interesting findings as you
hit them, don't batch them up.** If the re-run finishes and the
predicted direction of movement doesn't happen, the right move is to
surface it ("here's the before/after distribution, here's the new
Tucker's φ, hypothesis not supported, want to dig in or escalate?"),
not to silently run more experiments.

## Quick-start on a new machine

```bash
git clone git@github.com:SidBaines/persona-shattering-lasr.git
cd persona-shattering-lasr
git fetch origin sid/external-rollouts
# option 1: work on the existing branch that already has this handoff
git checkout sid/qwen25-b-analysis   # if it exists on remote
# option 2: branch off the latest external-rollouts commit
git checkout -b sid/qwen25-b-analysis origin/sid/external-rollouts
git branch --unset-upstream   # so accidental pushes don't clobber sid/external-rollouts

uv sync

# .env — needs at minimum:
#   HF_TOKEN=...
# (OPENROUTER_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY only needed
#  if we do re-rollout or labelling; for the pure re-admin path they
#  aren't required.)
cp .env.example .env   # if the example exists; else create by hand
```

### Hydrate existing artifacts

Stage-1 (rollouts) hydrates automatically when the orchestrator runs —
no manual step. For the choice-mass audit + baseline comparisons,
hydrate the 4 Stage-2 dirs explicitly:

```python
from src_dev.unsupervised_runs.io import hydrate_dataset_subtree
from pathlib import Path
REPO = "persona-shattering-lasr/psychometric-fa-runs"
prefix = "rollouts-llama318binstruct-t1.0-15t-2500p-seed436-scenarios_v2-uprompt_v6"
runs = [
    f"questionnaire-{prefix}-q_trait_ocean_v1-trait_mcq-direct-lp20",
    f"questionnaire-{prefix}-q_trait_ocean_v1-trait_mcq-direct-lp20-qm_qwen257binstruct",
    f"questionnaire-{prefix}-q_v5-likert-direct-lp20",
    f"questionnaire-{prefix}-q_v5-likert-direct-lp20-qm_qwen257binstruct",
]
for r in runs:
    hydrate_dataset_subtree(
        repo_id=REPO,
        path_in_repo=f"runs/{r}",
        local_dir=Path("scratch/psychometric_fa") / r,
    )
```

For the existing combined FA dirs (useful as before/after reference):

```python
for d in [
    "combined-R[B]-Q[v5+trait_ocean_v1]",                       # Llama
    "combined-R[B]-Q[v5+trait_ocean_v1]-qm_qwen257binstruct",   # Qwen2.5 (v1-parser based)
]:
    hydrate_dataset_subtree(
        repo_id=REPO,
        path_in_repo=f"combined/{d}",
        local_dir=Path("scratch/psychometric_fa") / d,
    )
```

## Running the Qwen2.5 re-run (the core task)

### Knobs to flip in `scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py`

```python
# Stage 2 only — rollouts are hydrated from HF.
STAGES_TO_RUN = ["rollouts", "questionnaire"]

# Restrict to the B × trait_ocean_v1 pair:
PAIRS = [("B", "trait_ocean_v1")]

# Enable cross-model admin (this populates the overrides below):
CROSS_MODEL_QUESTIONNAIRE = True
# Double-check these end up set (lines ~691 / ~699 in the file):
#   QUESTIONNAIRE_MODEL_OVERRIDE   = "Qwen/Qwen2.5-7B-Instruct"
#   QUESTIONNAIRE_MAX_CONTEXT_TOKENS = 32768

# Confirm parser version (should already be 2 on this branch, line ~722):
#   LOGPROB_PARSER_VERSION = 2
```

The parser version controls the run-id suffix: `-p2` is added
automatically when `LOGPROB_PARSER_VERSION > 1` AND a trait_mcq /
fc_pair block is in play AND logprob scoring is on. See
`_questionnaire_run_id()` around line 1069.

### Launch (tmux + unbuffered logging)

```bash
mkdir -p scratch/logs
LOG=scratch/logs/qwen25_b_trait_mcq_p2_$(date +%Y%m%d_%H%M%S).log
tmux new-session -d -s qwen25_b_trait_mcq \
  "cd $(pwd) && PYTHONUNBUFFERED=1 uv run python -u -m scripts_dev.unsupervised_embeddings.psychometric_rollout_fa 2>&1 | tee $LOG"
```

Expected timing on a 40 GB+ A40 / L4: tens of minutes to ~1 hour for
the full 2500-persona × ~100-item sweep on Qwen2.5-7B at bf16.
`max_model_len=25472` at bf16 fits comfortably in 32 GB. If GPU
memory is tighter, lower `QUESTIONNAIRE_VLLM_GPU_MEMORY_UTILIZATION`
or `QUESTIONNAIRE_MAX_CONTEXT_TOKENS` (see the module docstring).

One persona drops under the context-length filter
(`sample_df6e45b2c0d3ba37aaa91c72`, 66k-token VBA-macro loop). The
matrix will have 2499 rows, not 2500. This matches the old Qwen2.5 run.

### Upload to HF

After the run completes, upload the new Stage-2 dir:

```python
from src_dev.utils.hf_hub import upload_folder_to_dataset_repo, login_from_env
from pathlib import Path
login_from_env()
run_id = (
    "questionnaire-rollouts-llama318binstruct-t1.0-15t-2500p-seed436-"
    "scenarios_v2-uprompt_v6-q_trait_ocean_v1-trait_mcq-direct-lp20-p2-"
    "qm_qwen257binstruct"
)
upload_folder_to_dataset_repo(
    local_dir=Path("scratch/psychometric_fa") / run_id,
    repo_id="persona-shattering-lasr/psychometric-fa-runs",
    path_in_repo=f"runs/{run_id}",
    commit_message=f"sid/qwen25-b-analysis upload Qwen2.5 B trait_mcq (v2 parser)",
)
```

## Post re-run

### (1) Sanity checks before refitting FA

Before kicking off Stage 3/5 on a new matrix, look at what changed:

```bash
uv run python -m scripts_dev.psychometric_assessment.choice_mass_analysis
```
(with all 5 relevant Stage-2 dirs now present: the 4 above + the new
`-p2-qm_qwen257binstruct` one). The new run should have a much higher
choice-mass mean than the old v1 Qwen2.5 run. If it doesn't, something
is wrong with the parser plumbing — investigate before going further.

Also eyeball the per-item response distribution vs the old run —
there's a plot helper at:

```bash
uv run python -m scripts_dev.psychometric_assessment.compare_models
```
(edit `RUNS` in that file to point at the new `-p2` run-id alongside
the old non-`-p2` one so you can see the distributional shift). Look for:
- Bimodality softening in the new run (probability mass moving from
  {0.0–0.1, 0.9–1.0} toward the middle).
- Direction of change: persona-rank Spearman between v1 and v2 should
  stay high; what shifts is the variance per item.

### (2) Re-fit the Qwen2.5 combined FA

Edit `psychometric_rollout_fa.py`:

```python
STAGES_TO_RUN = ["rollouts", "questionnaire", "factor_analysis", "validation"]
PAIRS = [("B", "v5"), ("B", "trait_ocean_v1")]
CROSS_MODEL_QUESTIONNAIRE = True   # keeps the qm_qwen257binstruct suffix on the combined run-id
# LOGPROB_PARSER_VERSION = 2  (unchanged)
```

Stages 1 and 2 should all hit cache (hydrate from HF), so this is
CPU-only and takes ~15–25 min. Output lands at
`scratch/psychometric_fa/combined-R[B]-Q[v5+trait_ocean_v1]-qm_qwen257binstruct-p2/`.

Worth checking by hand before the run: Stage 3's input should include
the new `-p2` Qwen2.5 trait_mcq matrix, not the old v1 one. The
combine step keys off `_questionnaire_run_id()` which bakes
`LOGPROB_PARSER_VERSION` into the run-id, so with parser version = 2
it will pick the new matrix. But it's worth a grep on the resolved
run-ids printed to the log.

Upload:

```python
upload_folder_to_dataset_repo(
    local_dir=Path("scratch/psychometric_fa/combined-R[B]-Q[v5+trait_ocean_v1]-qm_qwen257binstruct-p2"),
    repo_id="persona-shattering-lasr/psychometric-fa-runs",
    path_in_repo="combined/combined-R[B]-Q[v5+trait_ocean_v1]-qm_qwen257binstruct-p2",
    commit_message="sid/qwen25-b-analysis upload Qwen2.5 combined FA (v2 parser)",
    ignore_patterns=[],
)
```

### (3) Cross-model Tucker's φ

Pre-existing script: `scripts_dev/psychometric_assessment/cross_model_factor_congruence.py`.
Edit `RUNS` to point at the two combined dirs (Llama existing + Qwen2.5
new-`-p2`). Run once per rotation (`oblimin` and `varimax`).

Key thing to look at: does the trait_mcq block's contribution to
cross-model congruence recover toward the v5 block's ~0.65 plateau, or
does it stay stuck near ~0.33? Either answer is research-relevant, but
it shapes what to do next.

### (4) Stop and report

At this point, there's enough new information to discuss. **Do not
keep going past this checkpoint without talking to the user.** Likely
follow-ups depend on the findings:

- If Tucker's φ on trait_mcq recovers: the old "trait_mcq measures
  something different" claim was wrong; it was a parser artifact. The
  paper-level statement should be rewritten. User will want to discuss.
- If it doesn't recover: the bimodality was real or there's a second
  confound. Choice-mass diagnostic on the new run + a look at per-item
  shifts will help localise. User will want to discuss.
- If something unexpected happens (e.g. new run crashes, or matrix
  shape is different): stop and debug, don't paper over.

## Known gotchas

- **`CROSS_MODEL_QUESTIONNAIRE` flips multiple globals at once** via
  module-level init. Don't try to run two different
  `QUESTIONNAIRE_MODEL_OVERRIDE`s in one script invocation — the
  run-id derivation assumes a single override per process.
- **Combined FA dirs are not auto-hydrated.** The orchestrator's
  caching logic is per-pair (Stage 2) only. For comparisons to
  existing combined FA, hydrate explicitly (snippet above).
- **HF cache discipline.** The full monorepo download for 2500 B
  rollouts + trait matrix is only ~few hundred MB — not a concern
  at this scope. But if you run alongside another agent doing 9-model
  external rollouts on the same machine, the 9-model cache can push
  200 GB+; clean with care.
- **trait_ocean_v1's response matrix is floats (0..1)**, not integer
  Likert. Downstream plotting scripts have a small-unique-values
  heuristic that switches to 2-D histograms in that case. If it
  doesn't look right, check the heuristic.
- **The 1 dropped persona** (VBA-macro truncation loop) is deterministic
  given the context-length filter. All Qwen2.5 comparisons run on the
  2499-intersection, not 2500. This is expected; not a bug.

## Paths and references

HF dataset repo: `persona-shattering-lasr/psychometric-fa-runs`

Existing B-rollout artifacts (all use v1 parser unless suffixed `-p2`):

- `runs/questionnaire-…-q_v5-likert-direct/` (Llama v5 direct-gen)
- `runs/questionnaire-…-q_v5-likert-direct-lp20/` (Llama v5 logprob)
- `runs/questionnaire-…-q_v5-likert-direct-qm_qwen257binstruct/` (Qwen2.5 v5 direct)
- `runs/questionnaire-…-q_v5-likert-direct-lp20-qm_qwen257binstruct/` (Qwen2.5 v5 logprob)
- `runs/questionnaire-…-q_trait_ocean_v1-trait_mcq-direct-lp20/` (Llama trait, v1 parser)
- `runs/questionnaire-…-q_trait_ocean_v1-trait_mcq-direct-lp20-qm_qwen257binstruct/` (Qwen2.5 trait, v1 parser) ← **will be superseded**
- `combined/combined-R[B]-Q[v5+trait_ocean_v1]/` (Llama combined FA, v1-based)
- `combined/combined-R[B]-Q[v5+trait_ocean_v1]-qm_qwen257binstruct/` (Qwen2.5 combined FA, v1-based) ← **will be superseded**

(The `rollouts-…` prefix everywhere is
`rollouts-llama318binstruct-t1.0-15t-2500p-seed436-scenarios_v2-uprompt_v6`.)

Local paths (in the worktree):
```
scripts_dev/psychometric_assessment/
├── HANDOFF_qwen25_comparison.md        # older / historical context
├── HANDOFF_qwen25_b_rerun.md           # this file
├── choice_mass_analysis.py             # letter vs digit mass diagnostic
├── compare_models.py                   # cross-model response comparison
├── cross_model_factor_congruence.py    # Tucker's φ + matching
└── …
scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py  # main driver
src_dev/psychometric/                   # Stage 2–5 implementations
scratch/psychometric_fa/                # hydrated + generated artifacts (gitignored)
```

Key commits on the starting branch:
- `62c9b19` trait_ocean nolead variant + Likert logprob scoring
- `23a692f` digit-aware trait_mcq logprob parser + raw top-k persistence
- `b2c9970` release vLLM engine + purge HF cache between presets
