# Deferred Work: Prompt Sweep Visualisation

Items deferred from the MVP implementation of system-prompt sweep support.
See git history and `scripts/experiments/personality_evals/o_avoiding_sweep_with_prompt.py`
for what was implemented.

---

## 1. TRAIT Small-Multiples Comparison Plot

**What**: When `--compare-dir` is used and `"trait"` data is present, produce a small-multiples
figure with one panel per Big Five trait instead of cramming all runs onto a single axis.

**Design**:
- 1 row × 5 columns (one panel per trait: O, C, E, A, N).
- Each panel: primary run = solid line, comparison runs = distinct dash styles (`_LINESTYLES`)
  at 50% alpha. Dark Triad omitted from this layout.
- Shared y-axis 0–1, shared x-axis, single legend at figure bottom.
- Output: `trait_sweep_comparison.png` (alongside the existing `trait_sweep.png`).
- Trigger: `plot_trait_sweep` checks `len(compare) > 0` and switches layout.

**Location**: `scripts/evals/personality/analyze_results.py` → `plot_trait_sweep`.

---

## 2. MMLU / Capability Overlay

**What**: When `--compare-dir` is used and capability eval data (`mmlu`, `gsm8k`, etc.) is present,
overlay comparison runs on the capability plot with distinct linestyles.

**Design**:
- Same single-panel layout as current `plot_capability_sweep`.
- Each comparison run adds a dashed line (same color, `alpha=0.55`, label includes run label).
- Baseline reference line computed from the primary run.

**Location**: `scripts/evals/personality/analyze_results.py` → `plot_capability_sweep`.

---

## 3. Additional Prompt Strategies

Once the o-avoiding baseline is validated, extend to:
- Multiple different prompt strengths (short vs. detailed instruction).
- Prompts for OCEAN traits (e.g. high-agreeableness system prompt + agreeableness LoRA).
- Prompt vs. LoRA interaction for other toy behaviors (verb density, sf_guy style).

Create new experiment scripts under `scripts/experiments/personality_evals/` following the
`o_avoiding_sweep_with_prompt.py` pattern.

---

## 4. `load_sweep_data` Label Propagation

Currently `load_sweep_data` does not attach a `label` field to the returned `SweepData`.
The `--compare-dir` CLI works by pairing labels externally. If downstream analysis needs
per-row labels in the DataFrame (e.g. for faceted plots), add:

```python
# In load_sweep_data, each row dict:
row_dict["_label"] = label   # where label is passed as a param
```

And expose `SweepData.label: str = ""` as an attribute.
