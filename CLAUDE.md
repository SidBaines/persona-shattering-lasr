# Claude Code Guidelines for This Project

## Pipeline Execution Preference

**When running pipeline stages (inference, editing, training), always prefer CLI commands when possible.**

### Preferred Method: CLI Commands (Step by Step)

Use the CLI interface for each stage:

**Inference:**
```bash
uv run python -m scripts.inference \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-name vicgalle/alpaca-gpt4 \
  --max-samples 5 \
  --output-path scratch/my_run/inference_output.jsonl
```

**Editing:**
```bash
uv run python -m scripts.editing \
  --provider anthropic \
  --model claude-sonnet-4-20250514 \
  --input-path scratch/my_run/inference_output.jsonl \
  --output-path scratch/my_run/edited_dataset.jsonl
```

**Training:**
```bash
uv run python -m scripts.training \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --input-path scratch/my_run/edited_dataset.jsonl \
  --checkpoint-dir scratch/my_run/checkpoints
```

### Why CLI Commands?

- More transparent and easier to debug
- User can see exact parameters being used
- Easier to modify individual parameters
- Better for experimentation and iteration
- Avoids creating unnecessary Python scripts

### When to Use Python Scripts

Only create Python scripts when:
- Running complex multi-stage experiments that need to be repeated
- Custom logic is needed between stages
- Automated parameter sweeps are required
- The experiment setup is too complex for simple CLI chaining

## Documentation

- Main README: [README.md](README.md)
- Inference: [scripts/inference/README.md](scripts/inference/README.md)
- Editing: [scripts/editing/README.md](scripts/editing/README.md)
- Training: [scripts/training/README.md](scripts/training/README.md)
