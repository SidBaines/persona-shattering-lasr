# Handover — Assistant Axis × Persona-Drift Experiment

**Status (branch `sid/actcap-scripts`):** code is written and import-checked, **nothing has been run yet**. Next steps: run the smoke pipeline end-to-end, validate, decide on the per-variant-axis question (see §6 **Open design decision**), then run a `balanced` or `full` preset.

**Most recent design correction.** Earlier I claimed the base-model Assistant Axis could be reused for the LoRA-modified model because projection is mathematically well-defined in the same hidden space. That's true mathematically but **not semantically equivalent** — when LoRA shifts weights it can also rotate the actual contrast direction between Assistant and role activations, so the base axis may no longer be the "Assistant-ness direction" for the LoRA-modified model. See §6 for the open decision and proposed plan.

---

## 1. What this experiment is

We're testing whether our LoRA approach to encoding personas can mitigate the **persona-drift** phenomenon described in Lu et al., "The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models" (arXiv 2601.10387). The paper's claim: in multi-turn conversations — especially emotional disclosure or meta-reflection contexts — model activations drift away from a learned "Assistant Axis" direction, and this drift correlates with harmful behavior. They mitigate it by **activation capping** along that axis.

We compare three drift-mitigation methods on **`meta-llama/Llama-3.1-8B-Instruct`**:

| Condition | Method | Engine |
|---|---|---|
| `vanilla` | Llama 3.1 8B base, no intervention | vLLM |
| `activation_capping` | upstream Assistant Axis cap, paper replication | HF transformers (forward hooks) |
| `lora_soup_c_plus_o_minus` | C+(1.0) ⊕ O−(1.0) baked LoRA soup | vLLM |

The **drift metric** is the paper's own: per-turn mean response-token activation, projected onto the Assistant Axis at a target layer, averaged across many conversations. We expect to see vanilla drift downward over turns (per the paper), capping flatten that trajectory (per the paper), and we want to know whether the LoRA soup also flattens / outperforms / underperforms capping.

Source paper code at `safety-research/assistant-axis` (MIT) is **vendored verbatim** at `vendor/assistant_axis/` at pinned SHA `a98961956`. The 5-step axis-build pipeline is theirs and called via subprocess unmodified. The drift protocol, the LoRA soup integration, and the projection metric extraction layer are ours.

---

## 2. Repo layout introduced by this branch

```
vendor/assistant_axis/                                        ← MIT, pinned, do not edit
  pipeline/{1_generate, 2_activations, 3_judge, 4_vectors, 5_axis}.py
  assistant_axis/{steering, axis, models, judge, ...}.py
  data/{roles/instructions/*.json, extraction_questions.jsonl}
  transcripts/persona_drift/{coding, writing, therapy, philosophy}.json   ← seed personas
  VENDOR_SOURCE.txt                                           ← pinned SHA + URL

src_dev/activation_capping/
  assistant_axis_loader.py                                    ← bridge: their axis ↔ our infra
    compute_capping_config()  — Phase 2 implementation
    apply_assistant_axis_capping()  — wraps their ActivationSteering as persistent hooks

scripts_dev/persona_drift_assistant_axis/
  config.py            ← all knobs in one place; SMOKE + FULL presets
  build_axis.py        ← Phase 1: subprocess-driver over their 5-step pipeline
  pick_capping.py      ← Phase 2: Cohen's d window + p25 thresholds
  run_drift.py         ← Phase 3: 3 conditions × 4 domains via run_rollout_generation
  project_drift.py     ← Phase 4: per-turn HF activation extraction + projection
  plot_drift.py        ← Phase 5: drift trajectory + per-layer heatmaps
  README.md            ← short quickstart
  HANDOVER.md          ← this file
```

**Existing infrastructure we lean on (read-only from this branch's POV):**

- `src_dev/rollout_generation/run.py` — multi-turn assistant↔user rollout engine (used by Phase 3)
- `src_dev/rollout_generation/model_providers.py:VLLMLoRaComboProvider` — LoRA soup baker (used by Phase 3)
- `src_dev/inference/providers/{local,vllm}.py` — InferenceProvider implementations (consumed by Phase 3)
- `src_dev/activation_capping/{model,axis}.py` — our existing activation-capping helpers (NOT used by paper-replication path; available as alternative)
- `src_dev/common/lora_catalogue.py:OCEAN_REGISTRY` — `c_plus`, `o_minus` adapter paths (used by Phase 3)
- `src_dev/utils/lora_combo_baking.py:bake_combined_lora` — bakes a multi-adapter soup into a single PEFT adapter (used by Phase 3)
- `src_dev/visualisations/__init__.py:PAPER_FIGURES_DIR` — for paper-figure output convention

---

## 3. Pipeline phases — what each does, in detail

### Phase 1 — `build_axis.py` (axis construction)

Subprocess-runs the upstream 5-step pipeline:

1. `1_generate.py` — vLLM batch generation. For each role × sysprompt × question, generates one response. Output: one JSONL per role under `responses/`.
2. `2_activations.py` — HF + forward hooks. For each response, extracts mean response-token activation at every layer. Output: one .pt per role under `activations/`.
3. `3_judge.py` — async LLM-judge scoring. Each response gets a 0/1/2/3 role-adherence score. We point this at OpenRouter for `qwen/qwen3-235b-a22b-2507` by setting `OPENAI_API_KEY` and `OPENAI_BASE_URL` env vars before invocation. Output: one JSON per role under `scores/`.
4. `4_vectors.py` — for each role, computes the mean activation across responses with score=3 (fully role-playing). Default role uses ALL responses. Output: one .pt per role under `vectors/`.
5. `5_axis.py` — `axis = mean(default_vectors) − mean(role_vectors_with_score=3)`. Saves a single `axis.pt` of shape `(n_layers, hidden_dim)` = `(32, 4096)` for Llama 3.1 8B.

**Knobs that flow through (`config.py:AxisBuildConfig`):**
- `num_roles`: cap how many roles we use (always includes `default`). Smoke = 8, full = all 275.
- `num_questions`: per-role question count. Smoke = 16, full = 240.
- `num_sysprompts_per_role`: cap on each role's sysprompt list (each role file ships 5). Smoke = 1, full = all 5.
- `min_count_per_role`: minimum score=3 samples to include a role's vector. Smoke = 1 (permissive), full = 50 (paper default).
- `vllm_gpu_memory_utilization`: defaults to 0.50; **drop to 0.30 if the GPU is shared with other workloads** (smoke originally needed this).
- `judge_model` / `judge_concurrency`: fixed to `qwen/qwen3-235b-a22b-2507` at concurrency 32.

**Idempotency.** Each step's output dir is checked at start; if non-empty, the step is skipped. To re-do a step, delete its output dir.

**Resumability.** If a step crashes mid-run, re-running picks up where it stopped (per upstream's per-role file existence check).

**HF upload (optional).** `--upload-hf` uploads the run dir to `persona-shattering-lasr/monorepo` at `activation_capping/assistant_axis/{model_slug}/{run_slug}/`. Default off.

**Compute & API estimates** (run them through the user before authorizing big spends):
- Smoke (8 roles × 1 sysprompt × 16 q): ~720 generations + ~720 judge calls. ~10–15 min on H100. ~$1–2 (judge).
- Full (275 × 5 × 240 = 330k responses): ~3–4 hours generation + ~2–3 hours activation extraction + ~330k judge calls. ~6–10 GPU hours + ~$30–150 in judge calls (depends on judge price). HF model download (~16 GB) and judge throughput dominate wall clock.

### Phase 2 — `pick_capping.py` (capping config)

Reads `axis.pt` + `activations/` from Phase 1. Computes:

1. Per-layer projection of every default-Assistant activation onto the axis: `(N_default, n_layers)`.
2. Per-layer projection of every role activation (capped at `role_sample_cap=10` roles to keep memory modest): `(sum_N_role, n_layers)`.
3. **Cohen's d per layer** between the two distributions.
4. **Layer window selection.** Default: contiguous window of `n_layers // 4` (= 8 for Llama 3.1 8B), constrained to `(0.5, 1.0)` of layer depth (paper's window for 80-layer Llama 3.3 70B was 56:72 = 70-90%, so for our 32 layers we expect ~16:24 to ~24:32). Picks the window maximising mean Cohen's d.
5. **Threshold per layer** in the chosen window: `np.percentile(default_proj[:, l], 25)` — the paper's "p0.25" convention.

Output: `capping_config.pt` with `{layers: [...], thresholds: {layer_idx: tau}, ...}` plus a sidecar `.json` summary.

**Knob.** `--layer-window LO HI` to override the auto-pick. `--threshold-percentile` to override 25.

### Phase 3 — `run_drift.py` (multi-turn drift rollouts)

For each `(condition, domain)` pair, calls our `run_rollout_generation` with the right inference config:

- **`vanilla`** → `provider="vllm"`, no adapter
- **`lora_soup_c_plus_o_minus`** → bakes c_plus + o_minus once into `{scratch_dir}/baked_lora_soup/` via `bake_combined_lora`, then `provider="vllm", vllm.adapter_path=<that path>`
- **`activation_capping`** → loads HF `AutoModelForCausalLM`, calls `apply_assistant_axis_capping(model, axis, capping_config)` (which calls upstream's `ActivationSteering(...).__enter__()` once, leaving hooks live for the lifetime of the model), then `provider="local", local.preloaded_model=(model, tokenizer)`

The user simulator is **`openai/gpt-5.4-nano`** via OpenRouter. For each domain we:

1. Read `vendor/assistant_axis/transcripts/persona_drift/{domain}.json` to extract the persona description and topic.
2. Build a per-domain user-sim system prompt (`build_user_sim_template`) and register it via `register_user_simulator_template`.
3. Materialise N copies of that seed as a JSONL dataset under `drift_rollouts/_seed_datasets/{domain}.jsonl` — different conversation IDs but same seed, distinct trajectories from stochastic sampling.
4. Call `run_rollout_generation` with `user_sim_generates_opening=True` so the user simulator writes the opener in-character.

Output: canonical conversation_training + conversation_trace JSONLs under `drift_rollouts/{condition}/{domain}/`.

**Resumability.** `run_rollout_generation` is itself resumable via stage events. Re-running picks up where it stopped.

**Knobs.** `--conditions` (comma-separated subset), `--domains`, `--num-conversations`, `--num-turns`. Smoke = 1 domain × 4 convs × 6 turns; full = 4 domains × 100 convs × 15 turns.

**Compute & API.**
- Smoke: 3 conditions × 1 domain × 4 conv × 6 turns = 72 assistant turns + 72 user-sim calls. ~15–30 min, ~$1–2.
- Full: 3 × 4 × 100 × 15 = 18k assistant turns + 18k user-sim calls. ~6–10 GPU hours (capping condition is the bottleneck, ~3–5× slower than vLLM). ~$50–200 user-sim API.

### Phase 4 — `project_drift.py` (per-turn projection)

Loads HF Llama 3.1 8B (no adapter, no capping — clean activation extraction). For each rollout under `drift_rollouts/{condition}/{domain}/`:

1. For each conversation, walk the messages and yield one `(turn_position, prefix_slice)` per assistant turn — the prefix slice is the conversation truncated at that assistant turn (inclusive).
2. Run forward passes via our existing `extract_response_activations_batched` (in `src_dev/activation_capping/axis.py`) — extracts mean activation over the LAST message's token positions, at every layer.
3. Project each turn's activations onto the axis at every layer: `einsum('Nld,ld->Nl', acts, axis_normalized)`.
4. Write one row per `(condition, domain, sample_id, turn_position)` to `drift_projections.jsonl`, with the per-layer projection list.

**Knob.** `activation_batch_size` (default 16) — drop to 4 if running OOM during forward passes.

**Compute.** Same magnitude as Phase 1 step 2 but over fewer conversations. Smoke: ~5 min. Full: ~3–5 GPU hours.

### Phase 5 — `plot_drift.py` (drift trajectory plot)

Loads `drift_projections.jsonl` and produces:

1. **Trajectory plot** (paper Figure 7 analog): one panel per domain, x-axis = turn position, y-axis = mean projection ± 95% bootstrap CI, one line per condition. Saved to `{scratch_dir}/plots/drift_trajectory_layer{N}.png` AND copied to `paper/figures/appendix/fig_assistant_axis_drift_trajectory.png` per repo convention.
2. **Per-layer heatmaps**: one per (condition, domain), showing how projection varies across (turn_position, layer). Useful for verifying that the capping window is the layer range where drift is actually happening.

**Default target layer** = `n_layers // 2` = 16 for Llama 3.1 8B. Override with `--target-layer`.

---

## 4. How to run on a new machine

### 4.1. Prerequisites

- Single GPU with ≥ 24 GB free memory (16 GB for Llama 3.1 8B + ~8 GB headroom for vLLM KV cache and HF activation extraction). Smoke is happy with less.
- Python 3.11 environment with our parent-repo venv at `/root/persona-shattering-lasr/.venv` (or wherever it lives on the new machine — the symlink `.venv` in this repo points there). Required packages: `vllm>=0.11`, `transformers`, `peft`, `huggingface_hub`, `openai>=2`, `python-dotenv`, `tqdm`, `jsonlines`, `matplotlib`, `numpy`, `torch>=2.0`, `pydantic`.
- API keys in `.env` (we symlink to the parent repo's `.env`):
  - `OPENROUTER_API_KEY` — used for both judge (Phase 1) and user simulator (Phase 3)
  - `HF_TOKEN` — required for `meta-llama/Llama-3.1-8B-Instruct` (gated) and for HF upload
- ~30 GB free disk under `scratch/` for smoke; ~150 GB for full (responses + activations + rollouts + projections). The HF model cache also needs ~16 GB for Llama 3.1 8B weights.

### 4.2. First-time setup on the new machine

```bash
# 1. clone + checkout
cd /root  # or wherever
git clone git@github.com:SidBaines/persona-shattering-lasr.git
git clone git@github.com:SidBaines/persona-shattering-lasr-actcap.git
cd persona-shattering-lasr-actcap
git checkout sid/actcap-scripts

# 2. set up venv (parent repo owns it)
cd /root/persona-shattering-lasr
uv sync
cd /root/persona-shattering-lasr-actcap
ln -snf /root/persona-shattering-lasr/.venv .venv
ln -snf /root/persona-shattering-lasr/.env .env  # or write a fresh .env with the keys above

# 3. import sanity check — should print 7 OK lines
.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
for m in ['scripts_dev.persona_drift_assistant_axis.config',
          'scripts_dev.persona_drift_assistant_axis.build_axis',
          'scripts_dev.persona_drift_assistant_axis.pick_capping',
          'scripts_dev.persona_drift_assistant_axis.run_drift',
          'scripts_dev.persona_drift_assistant_axis.project_drift',
          'scripts_dev.persona_drift_assistant_axis.plot_drift',
          'src_dev.activation_capping.assistant_axis_loader']:
    __import__(m); print('OK', m)
"

# 4. confirm GPU + env
nvidia-smi --query-gpu=memory.free --format=csv
.venv/bin/python -c "from dotenv import load_dotenv; load_dotenv()
import os
for k in ('OPENROUTER_API_KEY', 'HF_TOKEN'): print(k, '<set>' if os.environ.get(k) else '<MISSING>')"
```

### 4.3. Smoke run (first thing to do — pipeline correctness check)

If the GPU has < 40 GB free, edit `config.py` to drop `vllm_gpu_memory_utilization` from 0.50 to 0.30 in `SMOKE`.

```bash
# Phase 1a — base axis (~10–15 min on H100, ~$1–2 judge)
.venv/bin/python -m scripts_dev.persona_drift_assistant_axis.build_axis \
    --preset smoke --variant base 2>&1 | tee logs/smoke_phase1a.log

# Phase 1b — LoRA-soup axis (pre-merges adapters, then re-runs pipeline; ~10–15 min)
.venv/bin/python -m scripts_dev.persona_drift_assistant_axis.build_axis \
    --preset smoke --variant lora_soup_c_plus_o_minus 2>&1 | tee logs/smoke_phase1b.log

# Phase 2 — capping config (CPU, ~30 sec; uses base axis only)
.venv/bin/python -m scripts_dev.persona_drift_assistant_axis.pick_capping --preset smoke

# Phase 3 — drift rollouts (~15–30 min, ~$1–2 user-sim)
.venv/bin/python -m scripts_dev.persona_drift_assistant_axis.run_drift --preset smoke 2>&1 | tee logs/smoke_phase3.log

# Phase 4 — projection onto BOTH axes (~10 min — extracts twice, once per extraction model)
.venv/bin/python -m scripts_dev.persona_drift_assistant_axis.project_drift --preset smoke

# Phase 5 — plots (~30 sec; one trajectory figure per axis variant)
.venv/bin/python -m scripts_dev.persona_drift_assistant_axis.plot_drift --preset smoke
```

If you only want a faster smoke (skip the LoRA axis), drop Phase 1b — Phase 4 will discover only the base axis and emit just one trajectory figure. You can always come back later and run Phase 1b + re-run Phase 4 + 5 to get the second axis.

All artefacts under `scratch/persona_drift_assistant_axis/llama-3.1-8b-instruct/smoke_v1/`. New layout:

```
{run_slug}/
  axes/
    base/{responses,activations,scores,vectors,axis.pt,run_info.json}
    lora_soup_c_plus_o_minus/{merged_model/, responses, ..., axis.pt}
  capping_config.pt              # uses base axis only
  drift_rollouts/{condition}/{domain}/...
  drift_projections.jsonl        # rows include axis_variant + extraction_variant
  axis_cosine_similarity.txt     # written by Phase 4
  plots/
    drift_trajectory_base_layer{N}.{png,pdf}
    drift_trajectory_lora_soup_c_plus_o_minus_layer{N}.{png,pdf}
    drift_heatmap_{axis}_{condition}_{domain}.png
```

### 4.4. Full run

After smoke is green, kick off the full pipeline. Same commands with `--preset full`. Expect:

- Phase 1: ~6–10 GPU hours + ~$30–150 judge
- Phase 3: ~6–10 GPU hours + ~$50–200 user-sim
- Phase 4: ~3–5 GPU hours
- Total: ~$100–400 API + ~15–25 H100 hours

Use `nohup` / `tmux` and tee logs. The pipeline is resumable, so an interrupted run picks up where it stopped.

### 4.5. Migrating partial outputs from this machine

Smoke outputs from this machine are at `scratch/persona_drift_assistant_axis/...` — that path is gitignored. **They will NOT travel via git.** Options:

1. Discard smoke outputs entirely (~$5 sunk cost) and re-run on the new machine. Recommended unless you've burned a lot already.
2. `rsync` `scratch/persona_drift_assistant_axis/` to the new machine before deleting this one. Phase 1 and Phase 3 outputs are the expensive ones.

Note: NOTHING is reused between smoke and full runs (different `run_slug` → different scratch dir). See README.md "Caveats" for why.

---

## 5. What to look for at each phase (success / failure signals)

### Phase 1 — axis build

**Smoke success looks like:**
- `responses/` has 9 .jsonl files (8 roles + default), each with 16 lines.
- `activations/` has 9 .pt files.
- `scores/` has 8 .json files (default is skipped — no eval_prompt for default).
- `vectors/` has 9 .pt files.
- `axis.pt` is a `torch.Tensor` of shape `(32, 4096)`.
- `run_info.json` shows `axis_shape: [32, 4096]`, `axis_norm_mean` something like 1–10, `axis_norm_max_layer` somewhere in the upper half of the stack (paper's pattern).

**Common failures:**
- vLLM OOM → lower `vllm_gpu_memory_utilization`.
- Judge "OPENAI_API_KEY required" → `.env` missing or not loaded; check.
- Judge 401 / 429 → OpenRouter key wrong or rate-limited; back off concurrency.
- `4_vectors.py` "Only N score=3 samples, need 50" → smoke knob `min_count_per_role: 1` not propagated. Check.

### Phase 2 — capping config

**Sanity:**
- `capping_config.pt` exists, contains a layer window inside `(0, 31)` for Llama 3.1 8B.
- Sidecar JSON shows `cohens_d_in_window_mean` non-trivially positive (e.g. > 0.3). If it's near zero, the axis isn't well-separated → likely a Phase 1 problem.
- For smoke, expect a smaller window mean Cohen's d than for full (small sample bias).

### Phase 3 — drift rollouts

**Watch for:**
- `[ActivationSteering] Registered N hooks` printed at capping condition setup — confirms hooks are live.
- Per-condition wall-clock: capping is roughly 3–5× slower than vanilla/lora-soup (mixed-engine cost).
- `run_rollout_generation` log: `Progress | convs ... | asst turns ... | user turns ... | failed N` — failed should be 0 or very low.

**Common failures:**
- vLLM CUDA OOM when switching from capping (HF) to vLLM in the same process — capping leaves the HF model on GPU. Phase 3 currently runs all conditions sequentially in one process; if you hit this, run conditions separately:
  ```bash
  --conditions vanilla
  --conditions lora_soup_c_plus_o_minus
  --conditions activation_capping
  ```
- User-sim 429 from OpenRouter → back off `user_sim_max_concurrent`.

### Phase 4 — projection

**Sanity:**
- `drift_projections.jsonl` has rows ≈ `n_conditions × n_domains × n_convs_per_persona × n_turns`. For smoke that's 3 × 1 × 4 × 6 = 72 rows.
- Each row's `projection_per_layer` has length 32 (Llama 3.1 8B's layer count).

### Phase 5 — plots

**Smoke result interpretation: don't expect signal.** With 4 conversations per condition × 6 turns, the CIs will be huge and trajectories noisy. The smoke plot is a sanity check that the plotting code runs, not a research result.

**Full run is where to look for the real story:**
- **Vanilla**: expect downward trajectory across turns (drift away from Assistant), especially in `therapy` and `philosophy` domains per the paper.
- **Activation capping**: expect a much flatter trajectory.
- **LoRA soup**: this is the open question. Hypotheses:
  - Higher constant projection if C+/O− pushes the model toward a more "Assistant-like" trait pole
  - Lower constant projection if it pulls in the opposite direction
  - Flatter trajectory than vanilla if LoRA encodes drift-resistance
- Per-layer heatmaps: drift should be most pronounced in the upper half of layers (matches where the capping window lands).

---

## 6. Caveats and gotchas (read before debugging)

### Cap direction (potential semantic mismatch)

Upstream `vendor/assistant_axis/assistant_axis/steering.py:_apply_cap` is a **ceiling clamp** (projections > τ get pulled down to τ). The paper's Eq. 1 reads as a **floor clamp** (projections < τ get lifted to τ). We use upstream's published code as the canonical reference, but if results are weird (e.g. capping makes drift WORSE or has no effect at all), this is the first place to look. Our `src_dev/activation_capping/model.py:ActivationCappedModel` has both `mode="floor"` and `mode="ceiling"` if we want to compare.

### Mixed-engine fairness

vanilla and lora_soup run on vLLM; capping runs on HF transformers (because vLLM doesn't support forward hooks). Generation samples from the same model + prompt + sampling params should be equivalent up to kernel-level numerical noise — the comparison is valid for drift trajectories. The only difference is wall-clock: capping is ~3–5× slower.

### Drift seeds

Upstream ships exactly ONE `(persona, topic)` per domain in `transcripts/persona_drift/*.json`. We sample `num_conversations_per_persona` distinct trajectories from that single seed via stochastic sampling (temperature=1.0). The paper used 5 personas × 100 conversations per domain — those seeds are not open-sourced. If the user ever asks for more persona diversity, we'd extend `load_domain_seed` to read multiple seeds.

### LoRA soup behavior on Llama 3.1 8B is unverified

The C+ ⊕ O− soup at scales (1.0, 1.0) is being tested fresh here. We don't know a priori whether the combined LoRA is well-behaved (no NaN losses during inference, sensible outputs). Watch the first few rollouts qualitatively before committing to the full run. If outputs degenerate, try (0.7, 0.7) or one adapter at a time as ablation.

### `OCEAN_REGISTRY` adapter resolution

`run_drift.py` uses `OCEAN_REGISTRY["c_plus"].adapter_ref` and `["o_minus"].adapter_ref` which return `repo_id::subfolder` references resolved via `bake_combined_lora`. If those adapter paths change in the registry (newer training runs supersede old ones), the soup will silently use the new ones — usually fine, but if reproducing a specific run, pin the registry SHA.

### Phase 3 GPU lifetime

`run_drift.py` loads three different model setups in one process (HF for capping; two vLLM engines for vanilla and soup). On constrained GPUs this can OOM at the boundary between conditions. Workaround: run conditions in separate processes (one `--conditions X` per invocation).

### `.venv` and `.env` symlinks won't survive a machine move

Both are symlinks to absolute paths under `/root/persona-shattering-lasr/`. On a new machine you must re-link them (see §4.2). They're gitignored.

---

## 6. Open design decision — per-variant axis (READ THIS BEFORE RUNNING `full` / `balanced`)

The Assistant Axis = `mean(default_acts) − mean(role_acts)` is computed from how the **base model** behaves under different sysprompts. When we apply a LoRA, weights change, and both the "default" cluster and the "role-played" cluster in activation space can shift — possibly differently. The contrast direction (the axis) can therefore rotate. Projecting the LoRA-modified model's outputs onto the **base** axis tells you "where does the LoRA-modified model sit relative to the base model's notion of Assistant", which is informative but is **not the same as** "is the LoRA-modified model maintaining its own persona over turns".

Three options for the experiment:

- **Project everything onto base axis (current code).** All conditions plotted in one coordinate system → easy comparison. Caveat: the LoRA condition's "drift" is measured in a frame that may have rotated under the LoRA — could read as drift even when the model's own persona is stable.
- **Project each variant onto its own axis.** Pure per-variant drift. Caveat: each panel is in a different reference frame; harder to compare conditions on a single plot.
- **Both (recommended).** Build a base axis once, then build a second axis on the LoRA-modified model. Project each rollout onto **both** axes and report two trajectories per variant. Cosine similarity between the two axes is itself an interesting result (if cos > 0.9 the LoRA hasn't really rotated the axis and Option 1 is fine; if cos < 0.7 the rotation is meaningful and we need Option 2 or 3).

**Cost of "both"**: Phase 1 doubles for the LoRA variant (we run the upstream pipeline a second time on the merged-LoRA HF model). At balanced settings: +2 GPU hr + ~$15 judge. At smoke settings: negligible.

**What's needed to support "both" in code (currently TODO):**
1. Add a `merge_weighted_adapters` step that produces a standalone HF model dir for the LoRA soup (`src_dev/utils/lora_composition.py:merge_weighted_adapters` does this — already in the repo).
2. Extend `build_axis.py` with `--variant {base|lora_soup_c_plus_o_minus}`. When variant != base, pre-merge then run upstream's pipeline against the merged model dir, write outputs under `{run_slug}/_axes/{variant}/`.
3. Extend `project_drift.py` to discover ALL `axis.pt` files under the run dir and project onto each, writing `drift_projections.jsonl` rows with a `variant` column.
4. Extend `plot_drift.py` to facet by variant (one figure per axis-variant, or a side-by-side comparison plot).

**Implementation status: Option 3 ("both axes") is now implemented.** The four code changes above have all landed:

- ✅ `build_axis.py` accepts `--variant {base|lora_soup_c_plus_o_minus}` and pre-merges the LoRA soup via `merge_weighted_adapters` into `{run_slug}/axes/{variant}/merged_model/` before running upstream's pipeline against that merged model dir.
- ✅ Each variant's outputs land under `{run_slug}/axes/{variant}/{responses,activations,scores,vectors,axis.pt,...}`.
- ✅ `pick_capping.py` reads the BASE axis only (capping is never applied to LoRA variants).
- ✅ `project_drift.py` discovers all axes under `{run_slug}/axes/*/axis.pt`, prints a pairwise-cosine-similarity report, groups conditions by extraction model (vanilla/capping → base; lora_soup → merged-LoRA), loads each model once, and writes one row per (condition, axis_variant) pair to `drift_projections.jsonl`.
- ✅ `plot_drift.py` facets by `axis_variant` — one trajectory figure per axis, with the cosine-similarity report printed at the top.
- ✅ Activation-capping condition's extraction reapplies hooks on the same base axis so the bounded-projection effect is visible in the trajectory plot.

---

## 7. Other open questions / deferred decisions

1. **Persona-drift seeds:** stick with the 1-seed-per-domain × stochastic-sampling approach, or generate more seeds for richer coverage? Current code is the former.
2. **Capping floor vs ceiling:** trust upstream's ceiling clamp (current default) or also sweep our `ActivationCappedModel` floor mode for direct paper Eq. 1 fidelity?
3. **HF upload of axis + capping config:** off by default. Run with `--upload-hf` on Phase 1 if we want monorepo persistence. Recommended once smoke is green and full has run.
4. **Capability check (MMLU):** not included in v1. Easy to bolt on after Phase 5 — pull from existing `src_dev/evals/` infra.
5. **Multi-GPU tensor parallelism:** Phase 1 supports it via `tensor_parallel_size`; not used by default. If running on a multi-GPU box, set it in `config.py:AxisBuildConfig`.

---

## 7b. Cost profiles (cheaper-than-paper variants)

H100 at $2/hr, qwen3-235b judge ~$0.50/M tokens, gpt-5.4-nano user-sim ~$0.05/M input + $0.40/M output. Llama 3.1 8B vLLM throughput ~6k tok/s. Numbers below are for a SINGLE base axis. Add ~+50% on Phase 1 if we build a second axis on the LoRA variant (see §6).

| Profile | Phase 1 (axis) | Phase 3 (3 conditions) | Phase 4 | Total |
|---|---|---|---|---|
| `smoke` (current preset) — 8r × 1s × 16q · 1d × 4c × 6t | ~30 min, ~$1 judge | ~30 min, ~$1 user-sim | ~5 min | ~1 GPU hr, **~$3** |
| `balanced` (new preset, recommended) — 100r × 3s × 80q · 4d × 30c × 8t | ~2 GPU hr, ~$15 judge | ~3 GPU hr, ~$3 user-sim | ~30 min | ~6 GPU hr, **~$30–40** |
| `full` (paper-faithful) — 275r × 5s × 240q · 4d × 100c × 15t | ~7 GPU hr, ~$60 judge | ~12 GPU hr, ~$30 user-sim | ~5 GPU hr | ~25 GPU hr, **~$150–250** |

**Marginal cost of additional Phase-3 conditions (e.g. `c_plus_alone`, `o_minus_alone` ablations) at balanced settings:**

| Variant added | Marginal cost |
|---|---|
| New vLLM-served variant | ~30 min compute, ~$1 user-sim |
| New HF-served variant (forward hooks) | ~1.5 hr compute, ~$1 user-sim |
| Phase 4 over new rollouts | ~5–10 min |

The activation-capping condition (HF) is the wall-clock bottleneck in Phase 3 in every profile (3–5× slower than vLLM). To shave time, run capping on a smaller subset of conversations than the vLLM conditions — the relative comparison still holds.

**Where to cut cost first**, ranked by leverage:
1. `axis.num_questions: 240 → 80`. Biggest single-knob saving in Phase 1.
2. `drift.num_conversations_per_persona: 100 → 30`. Tight enough CIs for trajectory plots.
3. `axis.num_roles: 275 → 100`. 100 still gives a robust population contrast.
4. `drift.num_turns: 15 → 8`. Drift dynamics visible by turn 6–8.
5. `axis.num_sysprompts_per_role: 5 → 3`. Modest saving, modest signal cost.

Don't cut: `axis.num_roles` below ~50 (axis quality degrades), `axis.min_count_per_role` (it's a quality filter, not cost knob — though smoke uses 1 to allow per-role vectors with very few samples).

---

## 8. Tasks tracker (Claude Code internal)

If picking up this work in a new agent session, the relevant tasks (from this session) are:

- ✅ Phase 1–5 drivers all written
- ✅ `BALANCED` preset added (~10× cheaper than `FULL`)
- ✅ Per-variant axis support landed (build_axis `--variant`, multi-axis projection, faceted plots)
- ⏳ **Run smoke test end-to-end (both variants)** — blocked on GPU availability
- ⏳ **Validate smoke** — confirm axes built for base and lora_soup variants, cosine-similarity report sensible, drift plots render once per axis, capping condition shows bounded projection
- ⏳ **Run balanced** (~$45–55 single-axis baseline + ~$15 for LoRA-soup axis = ~$60, ~8 GPU hr) for the real result. Expand to full only if balanced shows signal worth replicating.

---

## 9. Reference: relevant files at a glance

| File | Purpose | Key entry point |
|---|---|---|
| `config.py` | knobs + presets | `get_preset("smoke" \| "full")` |
| `build_axis.py` | Phase 1 | `build_axis(cfg, upload_hf=...)` |
| `pick_capping.py` | Phase 2 | `compute_capping_config(...)` (in `src_dev/activation_capping/assistant_axis_loader.py`) |
| `run_drift.py` | Phase 3 | `run_drift(cfg, conditions=...)` |
| `project_drift.py` | Phase 4 | `project_drift(cfg)` |
| `plot_drift.py` | Phase 5 | `plot_drift(cfg, target_layer=...)` |
| `vendor/assistant_axis/pipeline/{1..5}_*.py` | upstream pipeline (do not edit) | run via subprocess from `build_axis.py` |
| `vendor/assistant_axis/assistant_axis/steering.py:ActivationSteering` | upstream capping hooks | invoked via `apply_assistant_axis_capping(...)` |
| `src_dev/activation_capping/assistant_axis_loader.py` | bridge: upstream axis ↔ our infra | `compute_capping_config`, `apply_assistant_axis_capping`, `load_axis`, `load_capping_config` |

---

**End of handover.** Pinned commit when this file was written: this commit + 1 (the next one). Branch: `sid/actcap-scripts`. The next agent should start by reading this file end-to-end, doing the §4.2 setup, running the §4.3 smoke pipeline, and reporting the §5 sanity-check signals back to the user.
