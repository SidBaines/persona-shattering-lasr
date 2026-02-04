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

## LoRA Manipulation Reference

Quick reference for working with trained LoRA adapters.

### Combining Multiple LoRAs

```python
from peft import PeftModel

# Load base model with first LoRA
model = PeftModel.from_pretrained(base_model, "path/to/lora1", adapter_name="lora1")
model.load_adapter("path/to/lora2", adapter_name="lora2")

# Combine with weights (don't need to sum to 1.0)
model.add_weighted_adapter(
    adapters=["lora1", "lora2"],
    weights=[0.7, 0.3],
    adapter_name="combined",
    combination_type="linear"
)
model.save_pretrained("path/to/combined", selected_adapters=["combined"])
```

### Task Vector Arithmetic (Subtracting LoRAs)

```python
# Subtract: use negative weights
model.add_weighted_adapter(
    adapters=["lora1", "lora2"],
    weights=[1.0, -1.0],  # lora1 - lora2
    adapter_name="difference"
)
```

### Direct Weight Manipulation (No Base Model)

```python
from safetensors.torch import load_file, save_file
import shutil

lora1 = load_file("path/to/lora1/adapter_model.safetensors")
lora2 = load_file("path/to/lora2/adapter_model.safetensors")

result = {k: 0.7 * lora1[k] + 0.3 * lora2[k] for k in lora1}
save_file(result, "path/to/combined/adapter_model.safetensors")
shutil.copy("path/to/lora1/adapter_config.json", "path/to/combined/adapter_config.json")
```

### Memory-Efficient Loading (4-bit)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B",
    quantization_config=quantization_config,
    device_map="auto"
)
# Llama 3.1 8B: FP16 ~16GB, 4-bit ~5-6GB
```

### Inspecting LoRA Config

```python
config = model.peft_config["default"]
print(f"Rank: {config.r}, Alpha: {config.lora_alpha}")
print(f"Target modules: {config.target_modules}")

# List all LoRA layers
for name, module in model.named_modules():
    if hasattr(module, 'lora_A'):
        print(f"LoRA applied to: {name}")
```

### Merge & Switch Adapters

```python
# Merge permanently into base model
model = model.merge_and_unload()
model.save_pretrained("path/to/merged_model")

# Or switch between adapters at runtime
model.set_adapter("lora1")
# ... inference ...
model.set_adapter("lora2")
model.disable_adapters()  # Use base model
```

### W&B Logging in Callbacks

When using HuggingFace Trainer with `report_to="wandb"`, don't pass explicit `step=` in callback `wandb.log()` calls—let W&B auto-increment to stay aligned with Trainer's step counter.

## Before Implementing

**REMINDER:** Check what exists in `src/` before implementing in `scripts/`. Use utilities from `src/` when working in `scripts/`.
