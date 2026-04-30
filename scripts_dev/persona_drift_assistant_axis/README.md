# Persona-Drift × Assistant Axis Experiment

> **For agents picking up this work mid-flight, read [HANDOVER.md](./HANDOVER.md) first.** It covers what's built, what's been run, what to look for, machine-migration notes, and open questions.


Comparison of three drift-mitigation methods on Llama 3.1 8B Instruct:

| Condition | Method | Engine |
|---|---|---|
| `vanilla` | base model, no intervention | vLLM |
| `activation_capping` | upstream Assistant Axis cap (paper replication) | HF transformers |
| `lora_soup_c_plus_o_minus` | C+(1.0) ⊕ O−(1.0) baked LoRA soup | vLLM |

Source paper: Lu et al., "The Assistant Axis: Situating and Stabilizing the
Default Persona of Language Models" (arXiv 2601.10387). Pipeline code
vendored at `vendor/assistant_axis/` (MIT, pinned commit `a98961956`).

## Pipeline

```
Phase 1 — build_axis.py        ← upstream pipeline → axis.pt
Phase 2 — pick_capping.py      ← Cohen's d window + p25 thresholds → capping_config.pt
Phase 3 — run_drift.py         ← 3 conditions × 4 domains × N convs × M turns
Phase 4 — project_drift.py     ← per-turn activation extraction → drift_projections.jsonl
Phase 5 — plot_drift.py        ← Fig 7 analog + per-layer heatmaps
```

All knobs live in `config.py`. Two presets:

* **`smoke`** — 8 roles, 16 questions, 1 sysprompt; 1 domain × 4 convs × 6 turns. ~$5–10, ~30 min on a single H100.
* **`full`** — paper-faithful: 275 roles × 240 questions × 5 sysprompts; 4 domains × 100 convs × 15 turns. ~$200–500.

## Quick start

Smoke run, end to end (note Phase 1 runs twice — once per axis variant):

```bash
.venv/bin/python -m scripts_dev.persona_drift_assistant_axis.build_axis     --preset smoke --variant base
.venv/bin/python -m scripts_dev.persona_drift_assistant_axis.build_axis     --preset smoke --variant lora_soup_c_plus_o_minus
.venv/bin/python -m scripts_dev.persona_drift_assistant_axis.pick_capping   --preset smoke
.venv/bin/python -m scripts_dev.persona_drift_assistant_axis.run_drift      --preset smoke
.venv/bin/python -m scripts_dev.persona_drift_assistant_axis.project_drift  --preset smoke
.venv/bin/python -m scripts_dev.persona_drift_assistant_axis.plot_drift     --preset smoke
```

All artefacts persist under `scratch/persona_drift_assistant_axis/{model_slug}/{run_slug}/`
and resume on re-run. HF upload of axis + capping config is gated by `--upload-hf`
on Phase 1.

## Knobs (selected)

| Knob | Phase | Default (smoke / full) |
|---|---|---|
| `axis.num_roles` | 1 | 8 / all (275) |
| `axis.num_questions` | 1 | 16 / 240 |
| `axis.num_sysprompts_per_role` | 1 | 1 / all (5) |
| `axis.judge_model` | 1 | `qwen/qwen3-235b-a22b-2507` |
| `capping.threshold_percentile` | 2 | 25.0 |
| `capping.layer_window` | 2 | auto-pick by Cohen's d |
| `drift.domains` | 3 | `('coding',)` / all 4 |
| `drift.num_conversations_per_persona` | 3 | 4 / 100 |
| `drift.num_turns` | 3 | 6 / 15 |
| `drift.user_sim_model` | 3 | `openai/gpt-5.4-nano` |

## Caveats

* **Mixed engine.** Capping runs on HF transformers (forward hooks); the other two run on vLLM. Generation samples are equivalent modulo kernel-level numerics; wall-clock for the capping condition is ~3–5× slower.
* **Cap direction.** Upstream's `_apply_cap` is a ceiling clamp. The paper Eq. 1 reads as a floor clamp. We use upstream's published implementation. If results are surprising, sweep both modes via our `ActivationCappedModel` (separate from `ActivationSteering`).
* **Drift seeds.** Upstream ships only ONE (persona, topic) per domain in `transcripts/persona_drift/*.json`. Phase 3 generates `num_conversations_per_persona` trajectories from that single seed via stochastic sampling. To run the paper's 5-persona × 100-conv setup verbatim we'd need the original seed list (not open-sourced).
