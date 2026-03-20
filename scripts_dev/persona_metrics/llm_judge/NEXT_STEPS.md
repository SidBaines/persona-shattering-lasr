# OCEAN Judge — Next Steps

Status as of 2026-03-20. Branch: `ocean-judges`.

---

## What exists now

| Component | Location |
|-----------|----------|
| v2 judge classes (all 5 traits) | `src_dev/persona_metrics/metrics/ocean_v2.py` |
| Judge agreement harness (two-stage) | `src_dev/persona_metrics/llm_judge_agreement.py` |
| OCEAN calibration script | `scripts_dev/persona_metrics/llm_judge/ocean_judge_calibration.py` |
| Coherence calibration script | `scripts_dev/persona_metrics/llm_judge/coherence_calibration.py` |
| Rollout → judge dataset converter | `scripts_dev/persona_metrics/llm_judge/rollout_sweep_to_judge_dataset.py` |
| Neuroticism calibration dataset (240 prompts × 3 conditions × 3 responses) | HF: `persona-shattering-lasr/ocean_judge_runs` |

---

## Immediate next steps

### 1. Inspect neuroticism judge results

The current panel run (gpt-4o-mini × haiku-3.5 × gemini-2.0-flash, 3 repeats each)
will produce results at:

```
scratch/ocean_judge_runs/runs/neuroticism-seed-42-dd7d1146dbc2/
  judge_runs/gpt_4o_mini-haiku_35-gemini_flash_20-*/
    analysis/summary.json          ← Krippendorff α, pairwise QWK
    analysis/condition_metrics.json ← check: low < neutral < high ordering
    analysis/per_item_disagreement.jsonl ← sorted by rater spread
    plots/
```

Key things to check:
- Condition ordering (low < neutral < high) for each rater
- Score distribution — are judges using the full -4…+4 range or collapsing?
- High-disagreement items — pick top 10-20 for human inspection

```bash
uv run python src_dev/jsonl_tui/cli.py \
    scratch/ocean_judge_runs/runs/neuroticism-seed-42-dd7d1146dbc2/\
judge_runs/*/analysis/per_item_disagreement.jsonl
```

---

### 2. LoRA scale sweep → judge pipeline

**Goal:** verify that judge scores increase monotonically with LoRA adapter scale.
This validates the judge is measuring a real signal, not just responding to surface cues.

**Steps:**
1. Run a neuroticism LoRA scale sweep (requires GPU):
   ```bash
   # Configure in scripts_dev/rollout_experiments/ following existing examples
   # e.g. o_frequency_lora_sweep.py as template
   ```

2. Convert sweep output to judge dataset:
   ```bash
   uv run python scripts_dev/persona_metrics/llm_judge/rollout_sweep_to_judge_dataset.py \
       --sweep-dir scratch/<your_neuroticism_sweep>/ \
       --output scratch/judge_datasets/neuroticism_lora_sweep.jsonl \
       --model <your_base_model>
   ```
   The `condition` field will be `<condition_name>@scale_<value>` (e.g. `no_prompt@scale_+1.00`).

3. Run judge panel against it:
   ```bash
   uv run python scripts_dev/persona_metrics/llm_judge/ocean_judge_calibration.py \
       --trait neuroticism --stage judge \
       --dataset scratch/judge_datasets/neuroticism_lora_sweep.jsonl
   ```

4. Plot score vs scale factor (parse condition field for scale value).

**Known gap:** The current dataset only has extreme high/low conditions —
no intermediate scores in training signal for the judge. LoRA rollouts at
e.g. 0.25x, 0.5x, 0.75x will fill this gap naturally.

---

### 3. Coherence judge calibration

Run the graded degradation calibration to validate the 0–100 coherence scale:

```bash
uv run python scripts_dev/persona_metrics/llm_judge/coherence_calibration.py \
    --stage all --max-prompts 60
```

**What to check:** `analysis/condition_metrics.json` should show monotonically
decreasing mean scores:

```
level_0_baseline   ~85–95
level_1_minor      ~65–80
level_2_moderate   ~45–60
level_3_severe     ~25–40
level_4_incoherent ~0–20
```

If ordering is wrong or range is compressed → revise the judge prompt
few-shot examples in `src_dev/persona_metrics/metrics/judge_configs.py`.

Once calibrated, coherence can be run on any `all_responses.jsonl` alongside
the OCEAN judge to detect whether persona manipulation degrades response quality.

---

### 4. Human validation of high-disagreement items

After any judge run, `per_item_disagreement.jsonl` is sorted by rater spread
(largest first). Pick the top 20-30 and score them manually.

This serves two purposes:
- Sanity check that high-spread items are genuinely ambiguous (not a prompt bug)
- Ground truth anchor for iterating on the judge prompt

---

### 5. Extend to remaining 4 OCEAN traits

The infrastructure supports all 5 traits. Once neuroticism is validated:

```bash
uv run python scripts_dev/persona_metrics/llm_judge/ocean_judge_calibration.py \
    --trait all --stage generate --max-prompts 240
```

Each trait gets its own frozen dataset and can be judged independently.

---

## Known limitations / design notes

- **No intermediate scores in calibration dataset.** The current generation
  approach (neutral / high / low system prompts) only anchors the extremes.
  LoRA rollouts at graded scales will fill this gap.

- **Scale calibration is relative, not absolute.** A score of 3 means "more
  neurotic than a 2", but we don't have human ground truth for what 3 should
  mean. Cross-trait comparison (is this response "more neurotic" than it is
  "more conscientious"?) is not yet possible.

- **Raters run sequentially** in the current judge pipeline. With 3 raters ×
  6480 calls the full panel takes ~1 hour. This is fine for calibration but
  would be slow for production scoring — consider parallelising raters if needed.

- **Coherence is on 0–100, OCEAN on -4…+4.** Different scales make joint
  analysis harder. Normalising coherence to 0–1 for comparison plots is simple
  but worth doing explicitly.
