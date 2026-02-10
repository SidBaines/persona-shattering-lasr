# Training

LoRA fine-tuning for causal language models using SFT (Supervised Fine-Tuning).
Trains a LoRA adapter on an edited dataset with configurable evaluations and W&B logging.

## CLI Usage

```bash
# Basic training run
uv run python -m scripts.training \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --input-path scratch/my_exp/edited_dataset.jsonl \
  --checkpoint-dir scratch/my_exp/checkpoints

# Custom hyperparameters
uv run python -m scripts.training \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --input-path scratch/my_exp/edited_dataset.jsonl \
  --checkpoint-dir scratch/my_exp/checkpoints \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --lora-r 32 \
  --lora-alpha 64

# Without W&B logging
uv run python -m scripts.training \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --input-path scratch/my_exp/edited_dataset.jsonl \
  --checkpoint-dir scratch/my_exp/checkpoints \
  --no-wandb
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model name or HuggingFace path | `meta-llama/Llama-3.1-8B-Instruct` |
| `--input-path` | Training dataset JSONL (required) | — |
| `--checkpoint-dir` | Output checkpoint directory (required) | — |
| `--epochs` | Number of training epochs | `3` |
| `--batch-size` | Per-device batch size | `4` |
| `--learning-rate` | Learning rate | `2e-4` |
| `--max-seq-length` | Maximum sequence length | `1024` |
| `--lora-r` | LoRA rank | `16` |
| `--lora-alpha` | LoRA alpha | `32` |
| `--val-split` | Validation split fraction | `0.1` |
| `--wandb-project` | W&B project name | `persona-shattering-v1` |
| `--no-wandb` | Disable W&B logging | off |

## Python Usage

```python
from pathlib import Path
from scripts.training import run_training, TrainingConfig, LoraConfig, SftConfig, TrainingEvaluationConfig
from scripts.common.config import ModelConfig

config = TrainingConfig(
    model=ModelConfig(name="Qwen/Qwen2.5-0.5B-Instruct"),
    lora=LoraConfig(r=16, lora_alpha=32),
    sft=SftConfig(num_train_epochs=3),
    evaluation=TrainingEvaluationConfig(
        evaluations=["count_o", "coherence"],
        eval_every_n_epochs=1,
    ),
    checkpoint_dir=Path("scratch/checkpoints"),
)
val_dataset, result = run_training(config, input_path=Path("scratch/edited.jsonl"))
```

## Features

- **LoRA adapters**: Efficient fine-tuning with configurable rank and alpha
- **Configurable evaluations**: Run any evaluation from `scripts.evaluation` during training
- **Training metrics**: Gradient/parameter norm logging (W&B)
- **W&B integration**: Automatic logging of metrics, sample tables, and LoRA adapter artifacts
- **Train/val split**: Automatic dataset splitting with configurable ratio
