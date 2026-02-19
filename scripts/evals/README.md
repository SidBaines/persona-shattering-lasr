# Evals Wrapper (Inspect AI)

`scripts/evals` is a thin wrapper over Inspect AI with four project-specific features:

1. Merge multiple LoRAs (with scales) into runnable models.
2. Define custom dataset-based evals with local metrics / LLM judges.
3. Persist normalized outputs plus per-run config metadata.
4. Resolve model/adapter refs across local + HuggingFace with ambiguity checks.

## Recommended Usage (Named Evaluation, Single Arg)

Use `named` + `--evaluation <name>` to run inspect-native eval definitions:

```bash
uv run python -m scripts.evals list-evaluations

uv run python -m scripts.evals named \
  --output-root scratch/evals/truthfulqa \
  --run-name llama31_8b_truthfulqa_smoke10 \
  --model-spec "name=base;base_model=hf://meta-llama/Llama-3.1-8B-Instruct" \
  --evaluation truthfulqa_mc1 \
  --limit 10
```

For custom named evals:

```bash
uv run python -m scripts.evals named \
  --output-root scratch/evals/coherence \
  --model-spec "name=base;base_model=hf://meta-llama/Llama-3.1-8B-Instruct" \
  --evaluation coherence1 \
  --limit 25
```

### One Run With Three Model Specs (Base + Local LoRA + Mixed LoRAs)

```bash
uv run python -m scripts.evals named \
  --output-root scratch/evals/truthfulqa \
  --run-name llama31_three_model_specs_truthfulqa_mc1_200 \
  --model-spec "name=base;base_model=hf://meta-llama/Llama-3.1-8B-Instruct" \
  --model-spec "name=local_lora_1p0;base_model=hf://meta-llama/Llama-3.1-8B-Instruct;adapters=local://my/local/lora/checkpoints/final@1.0" \
  --model-spec "name=hub0p5_plus_local_neg2;base_model=hf://meta-llama/Llama-3.1-8B-Instruct;adapters=hf://persona-shattering-lasr/o_avoiding-o_avoiding_20260218_102429_train-lora-adapter@0.5,local://my/local/lora/checkpoints/final@-2.0" \
  --evaluation truthfulqa_mc1 \
  --limit 200
```

## Low-level Usage (No Python Config Module)

Use `direct` to define runs entirely via CLI args:

```bash
uv run python -m scripts.evals direct \
  --output-root scratch/evals/truthfulqa_subset \
  --model-spec "name=base;base_model=meta-llama/Llama-3.1-8B-Instruct" \
  --model-spec "name=combo;base_model=meta-llama/Llama-3.1-8B-Instruct;adapters=local://scratch/a@0.5,hf://org/adapter::adapter@0.5" \
  --eval-kind benchmark \
  --eval-name truthfulqa_subset \
  --benchmark truthfulqa \
  --benchmark-arg target=\"mc1\" \
  --limit 50
```

You can still use `suite --config-module ...` for scripted experiment definitions.

## Custom Evals

For low-level custom evals in `direct` mode:
- Provide dataset args (`--dataset-source`, `--dataset-name`/`--dataset-path`, `--dataset-split`, `--max-samples`)
- Provide input builder (`--input-builder`)
- Provide one or more `--evaluation` metrics (from `scripts.persona_metrics`)
- Optionally pass judge prompt overrides with `--metric-param` or `--judge-prompt-template-file`
- Optionally provide `--scorer-builder` for non-persona local Inspect scorers

## Output Contract

Per `(model_spec, eval_spec)` run directory:
- `run_info.json` (model/eval/judge/materialization metadata + inspect log path)
- `native/inspect_logs/...`
- `jobs/manifest.json` (submit/resume custom flows)

Suite-level:
- no wrapper summary/manifest files (Inspect logs are source of truth)

## Viewing Inspect Output

Each `(model_spec, eval)` run stores its log under `<model>/<eval>/native/inspect_logs/`.
Inspect data/cache files (traces, dataset caches, etc.) are written to `<model>/<eval>/native/`
as siblings of `inspect_logs/`, so the log directory contains only the actual eval log files.

**Single model/eval — point at the log directory directly:**

```bash
uv run inspect log list \
  --log-dir scratch/evals/truthfulqa/llama31_8b_truthfulqa_smoke10/base/truthfulqa_mc1/native/inspect_logs

uv run inspect view start \
  --log-dir scratch/evals/truthfulqa/llama31_8b_truthfulqa_smoke10/base/truthfulqa_mc1/native/inspect_logs
```

**Multiple models — point at the run root with `--recursive`:**

```bash
uv run inspect view start \
  --log-dir scratch/evals/truthfulqa/llama31_vs_sfguy_truthfulqa_mc1_200 \
  --recursive
```

`run_info.json` includes `native.inspect_log_path` for the exact generated log file.

## Model Reference Resolution

For model and adapter refs:
- `local://...` forces local path resolution
- `hf://...` forces HuggingFace repo resolution
- Unprefixed refs are auto-resolved
- If both local + HF exist for an unprefixed ref, the run errors explicitly and asks you to disambiguate

## LoRA Materialization + Cleanup

Merged adapter models are cached under a shared models cache.

- Default: cache is retained for reuse across runs
- Use `--cleanup-materialized-models` (or `SuiteConfig.cleanup_materialized_models=True`) to remove merged artifacts at run end
- Runtime model cleanup is performed between model specs to reduce CUDA OOM risk in multi-model suite runs

## Non-wrapper Utilities

Visualization/sweep scripts are under `scripts/visualisations/`:
- `scripts/visualisations/eval_lora_scaling.py`
- `scripts/visualisations/plot_scaling.py`
- `scripts/visualisations/lora_arithmetic.py`
- `scripts/visualisations/compare_mmlu_results.py`
