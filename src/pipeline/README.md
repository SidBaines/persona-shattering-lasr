# Pipeline

Orchestrates the persona extraction pipeline stages.

## Overview

The pipeline runs these stages in sequence:
1. `inference` - Generate responses from the base model
2. `edit` - Edit responses to exhibit the target persona
3. `evaluate_pre` - Evaluate persona strength before fine-tuning
4. `train` - Fine-tune the model on edited responses
5. `inference_finetuned` - Generate responses from the fine-tuned model
6. `evaluate_post` - Evaluate persona strength after fine-tuning

## Usage

```python
from src.config import load_config
from src.pipeline import PipelineRunner

config = load_config("configs/toy_model.yaml")
runner = PipelineRunner(config)

# Run all stages
runner.run_all()

# Or run a single stage
runner.run_stage("inference")
```

## Configuration

Pipeline configuration in YAML:

```yaml
pipeline:
  stages: [inference, edit, evaluate_pre, train, inference_finetuned, evaluate_post]
  output_dir: scratch/toy_model_run
```

## Extending

To add a new stage:
1. Add the stage name to the `stages` list in config
2. Implement the stage handler in `runner.py`
3. Register the stage in the `STAGE_HANDLERS` dict

## Before Implementing

**REMINDER:** Check what exists in `src/` before implementing in `scripts/`. Use utilities from `src/` when working in `scripts/`.
