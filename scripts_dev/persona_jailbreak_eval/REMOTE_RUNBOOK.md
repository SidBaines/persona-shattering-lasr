# Persona-Jailbreak Eval — Remote Runbook

Copy-pasteable commands for running the smoke (and beyond) on the remote
deployment box. Companion to `README.md` — that one explains *what* the eval
does; this one gets it running.

---

## 1. Pull + sanity check

```bash
# Pull the new branch tip
cd /root/persona-shattering-lasr-actcap   # or wherever you cloned the actcap repo
git fetch origin
git checkout sid/actcap-scripts
git pull
```

```bash
# (If symlinks are stale on a fresh machine) re-link venv + .env per drift HANDOVER §4.2
ln -snf /root/persona-shattering-lasr/.venv .venv
ln -snf /root/persona-shattering-lasr/.env  .env
```

```bash
# Sanity-check imports — should print "OK" with no errors
.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from src_dev.persona_jailbreak_eval import config, judge_paper, runner
print('OK')
"
```

No new deps — everything (deepseek-v3 via OpenRouter, StrongREJECT and
WildJailbreak via HF `datasets`, matplotlib/scipy/pydantic/torch) is already
wired through the existing venv.

---

## 2. HF hydrate / upload (default ON)

Both drivers default to **hydrating from and uploading to** the
`persona-shattering-lasr/monorepo` HF dataset:

- **Hydrate at startup** (`evals/{eval_type}/{model_slug}/{run_slug}/`):
  if a prior run with the same `run_slug` exists on HF, its responses +
  judgments are downloaded into the local run-dir before any inference, so
  the idempotent stages skip already-completed work. Disable with
  `--no-hydrate-hf`.
- **Upload after each stage**: after inference, after judging, and after
  aggregation. Disable with `--no-upload-hf`.

**Capping artefacts also auto-hydrate.** If `activation_capping` is in
`--conditions` and `axis.pt` / `capping_config.pt` aren't already on disk,
the eval pulls the drift script's axis dir from
`activation_capping/assistant_axis/{model_slug}/{drift_run_slug}/axes/base/`
on HF (uploaded by `build_axis.py --upload-hf`) and re-derives
`capping_config.pt` locally from the per-role activations (~30 s, CPU).
Default `drift_run_slug` is `smoke_v1`; override via `--drift-run-slug`.

You can still pass `--axis-path` / `--capping-config-path` explicitly to
short-circuit hydrate; if those exist on disk they're used as-is.

If you want to skip capping entirely, pass `--conditions vanilla,lora_soup_c_plus_o_minus`
and no axis/capping artefacts are needed.

---

## 3. Smoke run — Option 1 (persona × StrongREJECT grid)

Minimal command — capping artefacts auto-hydrate from the drift run on HF:

```bash
mkdir -p logs

.venv/bin/python -m scripts_dev.persona_jailbreak_eval.run_persona_grid \
    --preset smoke \
    2>&1 | tee logs/jailbreak_grid_smoke.log
```

If your drift run lives under a different `run_slug` (default is `smoke_v1`):

```bash
.venv/bin/python -m scripts_dev.persona_jailbreak_eval.run_persona_grid \
    --preset smoke --drift-run-slug balanced_v1 \
    2>&1 | tee logs/jailbreak_grid_smoke.log
```

Smoke = 4 personas × 2 sysprompts × 25 StrongREJECT items + 50 benign control
× 3 conditions ≈ **750 generations + 750 judge calls**. Expect ~30 min, ~$3.

---

## 4. Smoke run — Option 2 (WildJailbreak)

```bash
.venv/bin/python -m scripts_dev.persona_jailbreak_eval.run_wildjailbreak \
    --preset smoke \
    2>&1 | tee logs/jailbreak_wj_smoke.log
```

Smoke = 100 adv-harmful + 50 adv-benign × 3 conditions ≈ **450 generations +
judge calls**. Expect ~20 min, ~$2.

---

## 5. Skip capping (vanilla vs. LoRA-soup only)

If you don't have the axis/capping artefacts on disk yet, run with two
conditions and skip the flags:

```bash
.venv/bin/python -m scripts_dev.persona_jailbreak_eval.run_persona_grid \
    --preset smoke --conditions vanilla,lora_soup_c_plus_o_minus \
    2>&1 | tee logs/jailbreak_grid_smoke_no_cap.log
```

```bash
.venv/bin/python -m scripts_dev.persona_jailbreak_eval.run_wildjailbreak \
    --preset smoke --conditions vanilla,lora_soup_c_plus_o_minus \
    2>&1 | tee logs/jailbreak_wj_smoke_no_cap.log
```

---

## 6. Outputs

```text
scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/grid_smoke/
  responses/
    responses_<condition>.jsonl                # one row per (sample, condition)
    baked_lora_soups/<combo_name>/             # cached merged adapter
  judgments/
    judgments_<condition>.jsonl                # paper rubric + refusal labels
  aggregate/
    harmful_rate_by_condition.csv              # Wilson 95% CIs
    refusal_rate_on_benign.csv
    harmful_rate_by_condition_x_category.csv
    summary_bars.png + .pdf                    # harm rate + over-refusal bars
```

(Replace `grid_smoke` with `wj_smoke` for the WildJailbreak run.)

---

## 7. After smoke — what to look for

1. **Idempotency**. Re-run the same command. Every stage should print
   `all N samples already cached` and finish in seconds.
2. **Vanilla harm rate**. Should land somewhere in the 30–80% range. If it's
   ≪10%, the personas aren't doing their job — make them more directive in
   `scripts_dev/persona_jailbreak_eval/personas/curated_harmful.json`.
3. **Over-refusal on benign**. Vanilla should be near zero (~1–5%). If it's
   high, the harm-personas are causing blanket refusal even on innocuous
   Alpaca prompts — interesting signal.
4. **Capping diagnostic**. Prints
   `[apply_assistant_axis_capping] FLOOR mode on layers ...` plus a pre/post
   cap-violation diagnostic (same gate the drift smoke passed). Aborts with
   a clear error if the direction is wrong.
5. **Headline ordering**. If the paper's claim generalises, expect
   `harm_rate(vanilla) > harm_rate(lora_soup) > harm_rate(capping)`. CIs at
   smoke will be wide; that's expected — smoke just verifies plumbing.

---

## 8. Re-aggregating without re-judging

Useful for tweaking the binarisation rule or regenerating plots from
existing JSONL:

```bash
.venv/bin/python -m scripts_dev.persona_jailbreak_eval.aggregate_and_plot \
    --run-dir scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/grid_smoke \
    --title "Persona × StrongREJECT smoke"
```

```bash
.venv/bin/python -m scripts_dev.persona_jailbreak_eval.aggregate_and_plot \
    --run-dir scratch/persona_jailbreak_eval/llama-3.1-8b-instruct/wj_smoke \
    --title "WildJailbreak smoke"
```

---

## 9. Scaling up after smoke is green

```bash
# Balanced (~3–5 GPU hr, ~$30): publication-quality CIs.
.venv/bin/python -m scripts_dev.persona_jailbreak_eval.run_persona_grid \
    --preset balanced 2>&1 | tee logs/jailbreak_grid_balanced.log

# Full (~12 GPU hr, ~$200): paper-faithful sample sizes.
.venv/bin/python -m scripts_dev.persona_jailbreak_eval.run_persona_grid \
    --preset full 2>&1 | tee logs/jailbreak_grid_full.log
```

For long runs, prefer `tmux` / `nohup` so a dropped SSH session doesn't kill
the job. Both scripts are resumable on the same `--run-slug`, so an
interrupted run picks up where it left off.

---

## 10. Common gotchas

- **vLLM CUDA fork crash** at startup: handled — both drivers call
  `ensure_vllm_fork_safe()` before any vLLM import.
- **OOM at the vanilla→capping handoff**: the capping condition runs last by
  design, but if the vLLM engine doesn't fully release GPU memory, run
  capping in a separate process: `--conditions activation_capping`.
- **OpenRouter 429**: drop `cfg.judge.max_concurrent` from 16 to 8 in
  `src_dev/persona_jailbreak_eval/config.py`.
- **Empty/garbled judge replies**: rows are marked with a non-null
  `parse_error` in the judgments JSONL and excluded from aggregation. The
  judge runner treats parse-errored rows as not-yet-completed
  (`_load_completed_ids(..., skip_errored=True)`), so a re-run retries only
  those rows. The retried result is appended; the aggregate loader
  de-duplicates on `(sample_id, condition)` last-write-wins, so retried rows
  don't double-count.
