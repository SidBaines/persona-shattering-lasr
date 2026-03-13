# Run Notes — 2026-03-13 Session

All runs performed on **NVIDIA L40S (44.5 GB VRAM)**, Llama-3.1-8B-Instruct base model.

---

## Training

| Run | Checkpoints | Duration | Notes |
|-----|-------------|----------|-------|
| `t_enjoying-train-20260312-223656` | cp-57 @ 22:41, cp-114 @ 22:45, cp-171 @ 22:49 | ~13 min | 3 epochs, 900 train / 100 val, batch 16 (4×4 grad accum), LoRA r=16 |

Epoch boundaries: cp-57 = end of epoch 1, cp-114 = epoch 2, cp-171 = epoch 3.
Best checkpoint by `eval_loss`: cp-171 (eval_loss=0.841, eval_accuracy=0.759).

---

## Rollout Sweeps

| Run | Adapter | Scale range / step | Samples | Conditions | Duration |
|-----|---------|-------------------|---------|------------|----------|
| `20260312_225843_t_enjoying_wide` | t_enjoying final | −5 → +5, step 1.0 | 30 | no_prompt, t_avoiding, t_enjoying | **42 min** |
| `20260313_074814_t_enjoying_cp57_wide` | t_enjoying cp-57 | −5 → +5, step 1.0 | 30 | no_prompt, t_avoiding, t_enjoying | **142 min** |
| `20260312_235607_t_enjoying_exhaustive` | t_enjoying final | −2.4 → +2.4, step 0.2 | 100 | no_prompt, t_avoiding, t_enjoying | **317 min (~5h17m)** |

Notes:
- Wide sweep (final) took 42 min for 11×3 = 33 runs × 30 samples.
- cp-57 wide sweep took ~3.4× longer (142 min) despite identical config — likely GPU was not fully idle at start.
- Exhaustive sweep: 25×3 = 75 runs × 100 samples. Single run ~3.9 min average.
- Degradation visible at +3 and beyond (positive) and −4 and beyond (negative) for t_enjoying.
- Chosen exhaustive range: ±2.4 (matches t_avoiding reference sweep).

---

## MMLU Capability Sweeps

All runs: 25 scale points (−2.4 → +2.4, step 0.2) × 3 runs × 100 samples, temperature=0.0, batch_size=32.

| Run | Adapter | Duration | Baseline acc | Notes |
|-----|---------|----------|-------------|-------|
| `20260313_102855_t_enjoying` | t_enjoying final (hf) | **~22 min** | 65% | Drops sharply beyond ±1.4; collapses to 4% at −2.4 |
| `20260313_114223_t_enjoying_cp57` | t_enjoying cp-57 (local) | **~22 min** | 63% | More robust at high positive scales (44% at +2.0 vs 30% for final) |
| `20260313_125614_t_avoiding_mmlu` | t_avoiding final (hf) | **~22 min** | 59% | Very stable across full range; only drops to 33% at +2.4 |

Duration timing: from first to last inspect log filename timestamp (75 logs per run).

Key observations:
- t_enjoying final and cp-57 take ~22 min each for the full suite (25 scales × 3 runs).
- t_avoiding is much more MMLU-robust — additive t-enjoyment corrupts representations far more than t-avoidance.
- cp-57 (epoch 1) retains capability better than the final checkpoint at high positive scales, consistent with mild overfitting in epochs 2–3.

---

## Timing Summary

| Run type | Config | Approx duration |
|----------|--------|----------------|
| LoRA training (168 steps, 900 samples) | batch 16, r=16, bf16 | ~13 min |
| Rollout sweep, wide (11 pts × 3 cond × 30 samples) | local inference | ~42 min |
| Rollout sweep, exhaustive (25 pts × 3 cond × 100 samples) | local inference | ~5h 17min |
| MMLU eval suite (25 pts × 3 runs × 100 samples) | batch 32, temp=0 | ~22 min |
