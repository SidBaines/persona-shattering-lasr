# Agent Guidelines — src_dev/evals

Operational knowledge for running and maintaining the evals pipelines,
especially the **LLM judge scale sweep** (`llm_judge_sweep`) that drives
most OCEAN-trait work in the paper. The repo-root `CLAUDE.md` has the general
rules; this file is the ops playbook.

---

## Entry points

| What | Where |
|---|---|
| Per-config sweep runner (the real entry point) | `scripts_dev/evals/llm_judge_sweep/runner_cells.py` |
| Older per-config runner (do **not** use for new work) | `scripts_dev/evals/llm_judge_sweep/runner.py` |
| Shared cell-sweep orchestration helpers | `src_dev/evals/cell_sweep/` |
| Shard launcher script (one per GPU) | `scripts_dev/evals/llm_judge_sweep/run_vanton4_qwen3.sh` |
| Config directory (one `.py` per sweep) | `scripts_dev/evals/llm_judge_sweep/configs/<family>/` |

`runner_cells.py` drives a single sweep: enumerate cells → fingerprint → hydrate
from HF → rollouts → judge → aggregate → plots → upload. It can be invoked
directly for a single config, or driven by the shard shell script for many
configs on one GPU.

---

## Config conventions

Each sweep family lives under `configs/<family>/`, sharing a `_shared.py`
that sets model, rollout params, and judge panel. Example family:
`configs/vanton4_qwen3/` (vanton4 adapters + Qwen3-235B judge).

Canonical shared settings (`_shared.py`):

```python
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MAX_SAMPLES = 240
NUM_ROLLOUTS_PER_PROMPT = 1     # 240×1 — 3× faster than 240×3
ASSISTANT_MAX_NEW_TOKENS = 2048
SCALE_POINTS = [-2.0, -1.0, 0.0, 1.0, 2.0]
SEED = 42
JUDGE_RATERS = [Qwen3-235B @ max_concurrent=32]
```

**Rule: change `_shared.py` values knowing each change creates a new fingerprint**
(see below). Downstream data on HF at the old fingerprint is orphaned, not
updated in place.

### Option B cross-trait configs

For trait X with own-dataset X.jsonl, we also run the X adapter against
the 4 *other* OCEAN dataset files to measure cross-trait bleed-through.
Naming: `{adapter}_on_{other_trait}.py` (e.g. `a_plus_on_openness.py`).
14 own-trait + 14×4 cross-trait = 70 configs per family.

---

## Fingerprints (the single most important concept)

Every sweep writes to `.../llm_judge_lora_scale_sweep/{fingerprint}/…`. The
fingerprint is a SHA over the rollout-invariant params:

```
base_model + dataset_path + max_samples + seed
+ num_rollouts_per_prompt + temp + top_p + max_new_tokens
```

Change any of those → new fingerprint → **separate HF/local path, no automatic
re-use of prior data**. Specifically: `NUM_ROLLOUTS_PER_PROMPT=3` and `=1`
produce different fingerprints even for the same adapter + dataset.

### Forging between fingerprints (when they're compatible)

If new-fingerprint data is a *subset* of old (e.g. 240×3 rollouts → 240×1 is
just "drop repeats 1/2 from rollouts.jsonl, filter judge rows to
`response_index==0`"), write a one-shot forge script that reads the old
fingerprint's data and writes it to the new fingerprint's path. Example:
`/tmp/forge_240x3_to_240x1.py`. The forged data then looks native — the sweep
runner hits cache and skips compute.

**Checks before forging:**
1. Confirm the target fingerprint is empty on HF (no collision).
2. Verify the schema assumption (e.g. inspect one judge file to confirm
   `response_index` actually exists and takes the expected values).
3. Count filtered rows match expectation (1440 → 480 for 240×1 with 2 judge
   repeats).

---

## HF monorepo layout

Everything lives in dataset `persona-shattering-lasr/monorepo`:

```
fine_tuning/{model}/ocean/{trait}/{direction}/{version}/evals/
  llm_judge_lora_scale_sweep/{fingerprint}/
    scale_+1.00/
      rollouts/rollouts.jsonl
      judge_runs/{rater_id}/{metric}.jsonl   # e.g. qwen3_235b/openness_v2.jsonl
      cell_info.json
    plots/, analysis/, sweep_config.json

combos/{model}/{combo_slug}/llm_judge_lora_scale_sweep/{fingerprint}/
  cell_{adapter_a}+1.00_{adapter_b}+1.00/rollouts|judge_runs/…
  (baseline and single-adapter cells written alongside, when the sweep
   covers them — e.g. 5×5 grids include them; 1×1 soups do not.)

combos/{model}/_baseline/llm_judge_lora_scale_sweep/{fingerprint}/
  (base-model baseline at this fingerprint, shared across combos.)
```

**Reading from HF**: `src_dev/utils/hf_hub.py::download_path_to_dir`. Paper
figure scripts cache under `scratch/paper_plots_cache/<plot>/`.

---

## HF upload strategy and rate limiting

HF enforces **~128 commits/hour/account**, and excess uploads return
HTTP 429 with a ~1-hour cooldown. Because each sweep can have many files,
*per-cell* commits blow past this fast.

### Three upload modes, gated by env vars

| Mode | Env var | Commits per sweep | When to use |
|---|---|---|---|
| Per-cell (legacy default) | *none* | ~5–25 | Single sweep, low concurrency |
| **Batched** (normal default) | `LLM_JUDGE_SWEEP_BATCH_UPLOAD=1` | **1** (+ separate baseline) | Long runs, many shards — **this is the expected mode** |
| **Skip** (exception only) | `LLM_JUDGE_SWEEP_SKIP_UPLOAD=1` | **0** | Only when HF is already rate-limited or you're doing ad-hoc work that *expects* to trip the limit |

The shard launcher (`run_vanton4_qwen3.sh`) exports
`LLM_JUDGE_SWEEP_BATCH_UPLOAD=1` at the top — batched is the default for
sharded runs, and for a **normal** sweep (even 4 shards × 15 configs) this
stays well under 128 commits/hour. **You should not need skip-upload for a
routine sweep.** Reach for it only when:

- You've already been hammering HF in the same hour with another big job
  (multi-GB forge upload, retry storms, concurrent sweeps across accounts)
  and want new work to complete without wasting time on 429 retries.
- You're doing something ad-hoc that you expect to trip the limit (bulk
  re-upload of historical data, pushing many small sweeps in rapid
  succession, experimenting with fingerprint forgery).

Skip-upload mode short-circuits inside `_with_upload_retry` — every HF upload
becomes a no-op and the module prints a banner at import time. Local data is
still written to `scratch/monorepo/…` exactly as if it had been uploaded.

### Retry window is short on purpose

`_with_upload_retry` uses 4 attempts with 10/20/40s delays (~70s total).
This handles momentary 5xx blips but **does not outlast** HF's 429 rate-limit
window. If HF is already rate-limited, don't crank up the retry count —
switch to `SKIP_UPLOAD=1` and run a consolidated upload pass later.

### Consolidated upload pattern (exceptional use only)

Only relevant when you ran with `SKIP_UPLOAD=1`. For normal sweeps, Stage-6
batched upload handles everything inline and nothing extra is needed.

When a large batch of local-only runs is done:
1. Survey `scratch/monorepo/` for sweep dirs + baseline dirs that aren't
   on HF yet. Walk both `fine_tuning/` and `combos/` subtrees.
2. Script a loop that calls `upload_sweep_root_generic(...)` with the same
   allow-patterns the Stage-6 batched path uses, pacing ~30–45s between
   commits to stay under 128/hr.
3. Use `src_dev/evals/cell_sweep/runner.py::upload_sweep_root` — don't roll
   a new helper.

---

## Sharding across GPUs

Pattern: one tmux window per GPU, each running
`CUDA_VISIBLE_DEVICES=<N> bash run_vanton4_qwen3.sh <cfg1> <cfg2> …`.

- **Always** pin `CUDA_VISIBLE_DEVICES` in the env (the launcher asserts it's set)
- Stagger launches with `sleep 30` to avoid 4 vLLMs loading the base model
  simultaneously and saturating disk/network
- Log per shard: `scratch/logs/run_vanton4_qwen3_gpu${N}_latest.log`
  (symlink the launcher maintains to the timestamped file)
- **Before relaunching**, always check
  `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`
  for orphan `VLLM::EngineCore` processes — a killed sweep can leave ~40 GB
  held until its engine PID is killed

### Killing a sweep cleanly

Don't rely on SIGINT to the outer bash — its subshells (for `tee`) and
orphaned Python children often survive. Instead:

```bash
# Kill tree: bash loop + python runner + tee subshells
PIDS=$(ps -eo pid,cmd --no-headers | grep -E "(run_vanton4_qwen3|runner_cells)" | awk '{print $1}')
kill -TERM ${PIDS}
sleep 3
# Escalate to SIGKILL only if needed; then clean orphan vLLM engines
```

---

## Baked adapter cleanup

Combo cells bake a materialized LoRA to
`scratch/baked_combo_adapters/{fingerprint}_{sweep_id}/`. Each gets a `.pid`
marker. At startup, `runner_cells.py` prunes baked dirs whose marker PID is
dead. Relying on this means the OS will eventually GC, but a manual sweep
check (`du -sh scratch/baked_combo_adapters/`) is worth doing before long runs.

---

## Rollout/judge overlap

`runner_cells.py` runs judges in a `ThreadPoolExecutor` in parallel with
rollouts (queue-driven). The effective pipeline speedup is ~1.5× vs serial
rollout-then-judge. If you see hanging asyncio tracebacks (`Event loop is
closed`, `Task exception was never retrieved`), those are benign cleanup
noise from the judge's httpx client, not failures.

---

## Paper figure hydration

Plots under `src_dev/visualisations/paper_*.py` hydrate from HF directly:

```python
from src_dev.utils.hf_hub import download_path_to_dir
download_path_to_dir(
    repo_id=HF_REPO_ID,
    path_in_repo=f"{hf_dir}",
    target_dir=local_cache_dir,
    allow_patterns=["*.jsonl"],
)
```

If you're in skip-upload mode and want a plot to pick up local-only data,
either upload first (preferred) or add a branch that resolves from
`scratch/monorepo/{hf_dir}` when the local path exists.

---

## Common failure modes

| Symptom | Probable cause | Fix |
|---|---|---|
| All sweeps FAILED at Stage 6, ~90s each | HF rate-limit (429) | Switch to `SKIP_UPLOAD=1`, run consolidated upload later |
| Sweep hangs after rollouts done | Judge still running in overlap mode | Wait — judge completion takes 1–3 min after last rollout cell finishes |
| `CUDA out of memory` on fresh launch | Orphan vLLM engine from prior run | `nvidia-smi --query-compute-apps` then `kill -9 <pid>` |
| Sweep recomputes cells that were "already done" | Fingerprint changed (something in `_shared.py` moved) | Check `rollout_fingerprint` diff; consider forging |
| Cross-trait Option B sweep is slow | Cache miss — these were NOT forged from 240×3 | Expected: ~15–18 min per sweep for fresh compute |
| "No files have been modified since last commit" | HF upload succeeded earlier | Benign; batched upload is idempotent |

---

## Reproducibility checklist for a new sweep family

1. Copy an existing family (`configs/vanton4_qwen3/`) as a template.
2. Update `_shared.py` only if you need different model/rollout params
   — be deliberate about new fingerprints.
3. Create one config per adapter (own-trait) + 4 per adapter (cross-trait
   Option B), reusing the `_shared.py` wildcard imports.
4. Update `run_<family>.sh` with the shard's config list.
5. Dry-run one config first: `uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells --config <module> --dry-run` (if supported), or run with `MAX_SAMPLES=5` to smoke-test.
6. Launch shards staggered; tail logs to confirm Stage 1 sees cached cells
   where expected.

---

## Cost-sanity reminders

- Qwen3-235B judging at `max_concurrent=32` with `JUDGE_REPEATS=2` on 240
  prompts × 5 cells × 5 metrics takes ~3–5 min per sweep on OpenRouter.
  Don't crank concurrency past 32 without checking OpenRouter's rate limits.
- `USER_MODEL = "z-ai/glm-4.5-air:free"` is the default free-tier user model
  for multi-turn rollouts — do not accidentally swap to a paid tier.
- When doing dry-run / smoke-test work, switch to a cheaper Claude model
  (see repo-root `CLAUDE.md`).
