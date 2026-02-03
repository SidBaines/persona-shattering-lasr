# Experiment Documentation

Instructions for agents creating and running experiments.

---

## Config-Driven Experiments

All experiments are defined by a **high-level config file**. The config instructs the code on:
- Which pipeline stages to run
- Configuration for each stage
- Paths to pre-existing artifacts (to skip stages)

### Running an Experiment

```bash
uv run persona run configs/my_experiment.yaml
```

The config file is the single source of truth for what the experiment does.

---

## Config Structure

A config file defines each stage of the pipeline:

```yaml
# configs/example.yaml

# Dataset stage - can be skipped by providing existing dataset
dataset:
  # Option 1: Generate new dataset
  source: "huggingface"
  name: "databricks/dolly-15k"
  split: "train"
  sample_size: 100

  # Option 2: Use existing dataset (skips generation)
  # path: "datasets/my_prepared_dataset.jsonl"

# Inference stage - can be skipped by providing existing responses
inference:
  provider: "local"
  model: "gpt2"

  # Option: Use existing inference results
  # path: "scratch/previous_run/inference_results.jsonl"

# Editing stage
editing:
  editor: "llm"
  model: "claude-3-haiku-20240307"

  # Option: Use existing edited responses
  # path: "datasets/pre_edited_responses.jsonl"

# Training stage - can be skipped by providing existing model
training:
  trainer: "local_lora"
  lora_rank: 8
  epochs: 3

  # Option: Use existing trained model
  # model_path: "models/my_trained_lora/"

# Evaluation
evaluation:
  metric: "count_char"
  char: "o"
```

### Skipping Stages

To skip a stage, provide the path to pre-existing artifacts:

| Stage | Skip by providing |
|-------|-------------------|
| Dataset | `dataset.path` |
| Inference | `inference.path` |
| Editing | `editing.path` |
| Training | `training.model_path` |

This allows you to:
- Reuse expensive inference results
- Use manually curated datasets
- Continue from a previously trained model
- Mix and match artifacts from different runs

---

## Development Workflow: src vs scripts

During development, configs can pull code from both `src/` (tested) and `scripts/` (experimental).

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  src/           Proven, tested code. Committed to main.         │
│  scripts/       Experimental code. Temporary. Not on main.      │
└─────────────────────────────────────────────────────────────────┘

Config during development:
  - Can reference modules from src/ (stable)
  - Can reference modules from scripts/ (experimental)

Config for production:
  - Must only reference src/ modules
  - Only src/-only configs should be committed to main
```

### Example: Developing a New Editor

1. Create experimental code in `scripts/my_new_editor.py`
2. Create a dev config that uses it:
   ```yaml
   editing:
     editor: "scripts.my_new_editor.MyEditor"  # References scripts/
   ```
3. Iterate locally until it works
4. With human approval, migrate to `src/editing/implementations/`
5. Update config to use src path:
   ```yaml
   editing:
     editor: "my_editor"  # Now uses src/ registry
   ```
6. Delete `scripts/my_new_editor.py`
7. Commit the src-only config

### Rules

- **Local dev:** Configs can use `scripts/` modules freely
- **Committed configs:** Must only use `src/` modules
- **scripts/ files:** Temporary. Delete after migration. Never push to main.

---

## Creating a New Experiment

### For Agents

1. **Read the relevant component READMEs** in `src/<component>/README.md`
2. **Check existing implementations** before creating new ones
3. **Create a config file** in `configs/`
4. **If new functionality is needed:**
   - Develop in `scripts/` first
   - Test with a local-only config
   - Request human approval to migrate to `src/`
5. **Only commit configs that use `src/` modules**

### Config Checklist

- [ ] All referenced modules exist in `src/` (for committed configs)
- [ ] Paths to artifacts are correct
- [ ] API keys are in `.env` (not hardcoded)
- [ ] Output directories exist or will be created

---

## Artifact Locations

| Artifact | Default Location |
|----------|-----------------|
| Datasets | `datasets/` |
| Inference results | `scratch/<run_name>/` |
| Trained models | `scratch/<run_name>/model/` |
| Evaluation results | `scratch/<run_name>/eval/` |
| Logs | `scratch/<run_name>/logs/` |

The `scratch/` directory is gitignored - use it freely for experiment outputs.
