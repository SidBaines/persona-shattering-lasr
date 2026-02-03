# Toy Model Implementation Checklist

> Delete items as they are completed. Once all items are deleted, delete this file.
>
> **WORKFLOW REMINDER:** Implement in `scripts/` first. Only migrate to `src/` with explicit human permission.

## Phase 1: Infrastructure

- [ ] Run `uv sync` and verify all dependencies install
- [ ] Create `.env` from `.env.example` and verify loading works
- [ ] Test CLI skeleton: `uv run persona --help`

## Phase 2: Data Pipeline (in scripts/)

- [ ] Create `scripts/load_data.py` - load alpaca-gpt4 dataset
- [ ] Verify: Dataset loads, samples printed to console
- [ ] Save dataset to `datasets/` for caching
- [ ] **MIGRATION CHECKPOINT**: When proven, request human approval to migrate loader to `src/data/loaders/huggingface.py`

## Phase 3: Base Model Inference (in scripts/)

- [ ] Create `scripts/run_inference.py` - run Llama 3.1 8B on dataset
- [ ] Import config loading from `src/config.py`
- [ ] Save outputs to `scratch/toy_model_run/inference/`
- [ ] Verify: Outputs look reasonable
- [ ] **MIGRATION CHECKPOINT**: When proven, request human approval to migrate to `src/inference/providers/local.py`

## Phase 4: Response Editing (in scripts/)

- [ ] Create `scripts/edit_responses.py` - call Anthropic API to edit responses
- [ ] Use prompt template from config
- [ ] Save edited responses to `scratch/toy_model_run/edited/`
- [ ] Verify: Edited responses have more O's than originals
- [ ] **MIGRATION CHECKPOINT**: When proven, request human approval to migrate to `src/editing/editors/llm_editor.py`

## Phase 5: Pre-training Evaluation (in scripts/)

- [ ] Create `scripts/evaluate.py` - count O's in responses
- [ ] Compare original vs edited responses
- [ ] Log metrics to `scratch/toy_model_run/metrics/`
- [ ] **MIGRATION CHECKPOINT**: When proven, request human approval to migrate to `src/evaluation/metrics/simple.py`

## Phase 6: Fine-tuning (in scripts/)

- [ ] Create `scripts/train_lora.py` - LoRA fine-tune on edited responses
- [ ] Save adapter to `scratch/toy_model_run/adapter/`
- [ ] Verify: Training completes without errors
- [ ] **MIGRATION CHECKPOINT**: When proven, request human approval to migrate to `src/training/trainers/local_lora.py`

## Phase 7: Post-training Evaluation (in scripts/)

- [ ] Create `scripts/evaluate_finetuned.py` - run inference with fine-tuned model
- [ ] Compare base model vs fine-tuned model O-counts
- [ ] Document results in `scratch/toy_model_run/results.md`

## Phase 8: Final Migration & Documentation

- [ ] Review all migrated code in `src/`
- [ ] Add tests in `tests/` for migrated code
- [ ] Update component READMEs with learnings
- [ ] Update `src/cli.py` to wire up all components
- [ ] Final review of AGENTS.md
- [ ] Delete this PLAN.md file
