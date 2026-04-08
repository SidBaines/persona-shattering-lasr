# Eval Script Standardization — Progress

Tracking file for agents working on the eval script standardization effort.
See full plan at: `.claude/plans/giggly-greeting-kettle.md`

## Items

### Item 0: Shared Infrastructure (`src_dev/eval_stages/`)
- **Status**: COMPLETE
- **Files created**:
  - `src_dev/eval_stages/__init__.py`
  - `src_dev/eval_stages/run_id.py` — `run_id_from_dict`, `chained_run_id`
  - `src_dev/eval_stages/cache.py` — `StageCache`, `StageCacheConfig`
- **Also**: Added `download_path_to_dir` to `src_dev/utils/hf_hub.py`
- **Tests**: Import check passed, run ID backward compatibility with Bloom verified, StageCache smoke test passed

### Item 1: LLM-Judge Single-Turn Rollout Sweep
- **Status**: COMPLETE — UNDER REVIEW
- **Files created**:
  - `scripts_dev/evals/llm_judge_sweep/runner.py` (740 lines)
  - `scripts_dev/evals/llm_judge_sweep/configs/conscientiousness_suppressor.py` (87 lines)
- **Old file deleted**: `scripts_dev/rollout_experiments/ocean/conscientiousness_suppressor_llm_judge_sweep.py`
- **Import check**: passed via `uv run`

### Item 2: Bloom Multi-Turn Rollout Eval
- **Status**: COMPLETE — UNDER REVIEW
- **Files created**:
  - `scripts_dev/evals/bloom/runner.py` (1115 lines)
  - `scripts_dev/evals/bloom/configs/default.py` (28 lines)
  - `scripts_dev/evals/bloom/README.md`
- **Old files deleted**: `scripts_dev/bloom_evals/run_bloom_eval.py`, `scripts_dev/bloom_evals/README.md`
- **Import check**: passed via `uv run`

### Item 3: Scaling Grid (Logprobs Trait Scores)
- **Status**: COMPLETE — UNDER REVIEW
- **Files created**:
  - `scripts_dev/evals/scaling_grid/runner.py` (685 lines)
  - `scripts_dev/evals/scaling_grid/configs/trait_ac_minus_vanton1.py` (28 lines)
- **Old file deleted**: `scripts_dev/personality_evals/trait_ac_minus_vanton1_grid.py`
- **Import check**: passed via `uv run`

## Architecture Notes

All scripts follow the **runner + config** pattern:
- `runner.py` contains stage logic, invoked via `--config` pointing to a config module
- Config modules are thin Python files with constants (EVAL_NAME, BASE_MODEL, etc.)
- Stages use `StageCache.run_or_hydrate()` for cache-first execution
- Run IDs are deterministic SHA256 hashes, chained across stages
- `done.json` markers store full config + git hash for reproducibility
- HF paths: `evals/{eval-type}/{eval-name}/{stage}/{run_id}/`

## Invocation Examples

```bash
# LLM-judge sweep
uv run python -m scripts_dev.evals.llm_judge_sweep.runner \
    --config scripts_dev.evals.llm_judge_sweep.configs.conscientiousness_suppressor

# Bloom eval
uv run python -m scripts_dev.evals.bloom.runner \
    --config scripts_dev.evals.bloom.configs.default

# Scaling grid
uv run python -m scripts_dev.evals.scaling_grid.runner \
    --config scripts_dev.evals.scaling_grid.configs.trait_ac_minus_vanton1
```
