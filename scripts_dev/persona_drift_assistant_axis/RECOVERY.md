# Balanced Run Recovery Procedure

What's checkpointed where, and how to resume from each failure mode.

## What is uploaded to HF, and when

**Phase 1 (build_axis) — only when launched with `--upload-hf`.**

* **Mid-Phase-1 checkpoint after Step 3 (judge)** — uploads
  `responses/`, `activations/`, `scores/`, `vectors/` (whichever already
  exist) to
  `persona-shattering-lasr/monorepo` →
  `activation_capping/assistant_axis/{model_slug}/{run_slug}/axes/{variant}/`.
  This is the safety net for the expensive judge spend. If a transient
  HF 5xx makes this upload fail, the pipeline continues — the final
  upload retries.
* **Final upload at end-of-Phase-1** — same path, full directory minus
  `staged/`, `logs/`, `merged_model/`.

**Phases 2-5: nothing on HF.** Local-disk only. The safety here is
on-machine resumability:

* Phase 2 reads from local `axes/base/{axis.pt, activations/}`. Output is a tiny `capping_config.pt`. Re-run is ~30 s if the inputs are present.
* Phase 3 (`run_drift`) runs `RolloutGenerationConfig(resume=True)` so re-launching the same command on the same machine continues each `(condition, domain)` cell from where it stopped. Failed-but-non-terminal samples retry; terminal samples stay failed (pass `retry_terminal_sample_ids` to retry them).
* Phase 4 reads `drift_rollouts/` + axes; output is one JSONL.
* Phase 5 reads the JSONL; output is plots.

## Failure scenarios → what to do

### A. Pipeline crashes mid-run, machine survives

Re-run the SAME command in the SAME shell. Every phase is idempotent:

* Phase 1 step skips when its output dir is non-empty.
* Phase 2 overwrites `capping_config.pt` (cheap).
* Phase 3 `resume=True` skips completed conversations.
* Phase 4 + 5 just re-read what's there.

No HF round-trip needed.

### B. Machine survives but local scratch is wiped

Re-run from Phase 1 with `--upload-hf` — incremental cache check kicks in
once HF rehydration is in place (TODO; not currently wired).

For now, on this scenario you'd download the axes manually:

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="persona-shattering-lasr/monorepo",
    repo_type="dataset",
    allow_patterns=[
        "activation_capping/assistant_axis/llama-3.1-8b-instruct/balanced_v1/axes/**",
    ],
    local_dir="./scratch/_rehydrate",
)
# Copy the contents into scratch/persona_drift_assistant_axis/llama-3.1-8b-instruct/balanced_v1/axes/
```

Then continue from Phase 2.

### C. Machine dies entirely mid-Phase-1

* Before Step 3 completes: nothing on HF, ~$0 spent on judge. Re-launch Phase 1 from scratch on a new machine.
* After Step 3 completes: post-step-3 HF checkpoint is up. Re-run Phase 1 with `--upload-hf` on the new machine; the per-step caching will see the rehydrated `responses/`, `activations/`, `scores/` and skip those steps. Step 4 + 5 (cheap CPU) re-run; final upload re-fires.

### D. Machine dies mid-Phase-3 or later

Phase 1 output is on HF (assuming `--upload-hf`). Re-rehydrate axes on a new machine, re-run Phase 2-5. Cost: re-spend Phase 3 user-sim API (~$3-15 for balanced) + ~3-5 GPU hr.

## Launch command for the overnight balanced run

```bash
# Sequential: Phase 1a → Phase 1b → Phase 2 → Phase 3 → Phase 4 → Phase 5.
# Each `&&` only fires if the prior phase exits zero.
cd /root/persona-shattering-lasr-actcap
mkdir -p logs
tmux new -d -s balanced "PYTHONUNBUFFERED=1 \
  .venv/bin/python -u -m scripts_dev.persona_drift_assistant_axis.build_axis \
      --preset balanced --variant base --upload-hf 2>&1 \
      | tee logs/balanced_phase1a.log \
  && .venv/bin/python -u -m scripts_dev.persona_drift_assistant_axis.build_axis \
      --preset balanced --variant lora_soup_c_plus_o_minus --upload-hf 2>&1 \
      | tee logs/balanced_phase1b.log \
  && .venv/bin/python -u -m scripts_dev.persona_drift_assistant_axis.pick_capping \
      --preset balanced 2>&1 \
      | tee logs/balanced_phase2.log \
  && .venv/bin/python -u -m scripts_dev.persona_drift_assistant_axis.run_drift \
      --preset balanced 2>&1 \
      | tee logs/balanced_phase3.log \
  && .venv/bin/python -u -m scripts_dev.persona_drift_assistant_axis.project_drift \
      --preset balanced 2>&1 \
      | tee logs/balanced_phase4.log \
  && .venv/bin/python -u -m scripts_dev.persona_drift_assistant_axis.plot_drift \
      --preset balanced 2>&1 \
      | tee logs/balanced_phase5.log; \
  echo === FINISHED $(date) ==="
```

Attach with `tmux attach -t balanced`. Tail with `tail -F logs/balanced_phase*.log`.

## Cost expectation for balanced

From HANDOVER.md §7b, with the two-axis option:

* Phase 1a (base axis): ~2 GPU hr, ~$15 judge.
* Phase 1b (LoRA-soup axis): ~2 GPU hr, ~$15 judge.
* Phase 2: ~30 s CPU.
* Phase 3 (3 conditions × 4 domains × 30 convs × 8 turns): ~3 GPU hr,
  ~$3-15 user-sim (Kimi K2 via OpenRouter). Capping condition (HF transformers) is the wall-clock bottleneck.
* Phase 4: ~30 min GPU.
* Phase 5: instant.
* **Total: ~7-8 GPU hr, ~$33-45.** Well under the $100 OpenRouter budget.

If the run goes 2× over budget on Phase 3 user-sim (worst-case Kimi K2 retries), still under $100. If it goes 3-4× over (very unusual), cap or kill manually.

## What is NOT yet checkpointed to HF

* Phase 3 rollouts (`drift_rollouts/{condition}/{domain}/...`).
* Phase 2 `capping_config.pt`.
* Phase 4 `drift_projections.jsonl`.
* Phase 5 plots.

Adding Phase-3 incremental upload would shrink the worst-case re-spend
to <$1, but is not yet wired (would need a small HF-aware wrapper around
each `_run_condition_for_domain` call). Worth doing later but not
critical for a balanced run within $100 budget.
