# Evals Wrapper (Inspect AI)

`scripts/evals` is a thin wrapper over Inspect AI with four project-specific features:

1. Merge multiple LoRAs (with scales) into runnable models.
2. Define custom dataset-based evals with local metrics / LLM judges.
3. Persist normalized outputs plus per-run config metadata.
4. Resolve model/adapter refs across local + HuggingFace with ambiguity checks.

## Recommended Usage (No Python Config Module)

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

For custom evals in `direct` mode:
- Provide dataset args (`--dataset-source`, `--dataset-name`/`--dataset-path`, `--dataset-split`, `--max-samples`)
- Provide input builder (`--input-builder`)
- Provide one or more `--evaluation` metrics (from `scripts.persona_metrics`)
- Optionally pass judge prompt overrides with `--metric-param` or `--judge-prompt-template-file`
- Optionally provide `--scorer-builder` for non-persona local Inspect scorers

## Output Contract

Per `(model_spec, eval_spec)` run directory:
- `summary.json`
- `records.jsonl` (when sample logs exist)
- `run_config.json` (resolved model/eval/judge/config snapshot for this run)
- `native/inspect_logs/...`
- `jobs/manifest.json` (submit/resume custom flows)

Suite-level:
- `suite_summary.json`
- `suite_manifest.json` (includes `metadata.cli_args` for `direct` runs)

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

## Non-wrapper Utilities

Visualization/sweep scripts are under `scripts/visualisations/`:
- `scripts/visualisations/eval_lora_scaling.py`
- `scripts/visualisations/plot_scaling.py`
- `scripts/visualisations/lora_arithmetic.py`
- `scripts/visualisations/compare_mmlu_results.py`
