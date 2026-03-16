# Training

Stable interface placeholder for LoRA fine-tuning.

Implementations live in `src_dev/lora_pipeline_persona_shattering/training/`. See that directory's README for CLI and Python usage.

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

**REMINDER:** Check what exists in `src/` before implementing in `src_dev/`. Use utilities from `src/` when working in `src_dev/`.
