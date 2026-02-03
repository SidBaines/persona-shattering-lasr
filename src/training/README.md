# Training

Model fine-tuning utilities for persona injection.

## Overview

This module provides trainers for fine-tuning models on edited responses to inject persona traits. Supports both local training (LoRA) and cloud-based training (Tinker API).

## Usage

```python
from pathlib import Path
from src.training import get_trainer

trainer = get_trainer("local_lora")
adapter_path = trainer.train(
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    train_data=[
        {"instruction": "...", "input": "", "output": "..."},
        # ...
    ],
    output_dir=Path("scratch/toy_model_run/adapter"),
    config={
        "lora_rank": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "epochs": 3,
        "batch_size": 4,
    },
)

# Load for inference
model = trainer.load_trained(
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    adapter_path=adapter_path,
)
```

## Available Trainers

| Type | Description | Status |
|------|-------------|--------|
| `local_lora` | Local LoRA fine-tuning with PEFT | STUB |
| `tinker` | Cloud-based training via Tinker API | STUB |

## Adding a New Trainer

1. Create a new file in `trainers/` (e.g., `qlora.py`)
2. Implement the `Trainer` interface from `base.py`
3. Register in `trainers/__init__.py`:

```python
from .qlora import QLoRATrainer

TRAINERS = {
    "local_lora": LocalLoRATrainer,
    "tinker": TinkerTrainer,
    "qlora": QLoRATrainer,  # Add here
}
```

## Configuration

In YAML config:

```yaml
training:
  type: local_lora
  lora_rank: 16
  lora_alpha: 32
  learning_rate: 2e-4
  epochs: 3
  batch_size: 4
```

## Training Data Format

Training data should be a list of dictionaries:

```python
[
    {
        "instruction": "Explain quantum computing",
        "input": "",  # Optional context
        "output": "Quantum computing utilizes...",  # Edited response
    },
    # ...
]
```

## Before Implementing

**REMINDER:** Check what exists in `src/` before implementing in `scripts/`. Use utilities from `src/` when working in `scripts/`.
