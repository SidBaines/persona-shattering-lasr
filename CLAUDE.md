# Claude Code Guidelines for This Project

## ⚠️ CRITICAL: GPU Requirements

**NEVER RUN TRAINING ON CPU.** Training large language models on CPU is impractical and will take days/weeks.

### Before Training - Verify GPU Access

Always verify GPU is available before starting training:

```bash
python3 -c "import torch; assert torch.cuda.is_available(), 'GPU not available!'; print(f'✓ GPU available: {torch.cuda.get_device_name(0)}')"
```

If this fails, **STOP and troubleshoot GPU access** before proceeding.

### GPU Troubleshooting

If `torch.cuda.is_available()` returns `False`:

1. **Check NVML/Driver Status:**
   ```bash
   nvidia-smi  # Should show GPU information
   ```

2. **Common Issues:**
   - **"Failed to initialize NVML"**: Container GPU passthrough issue
     - Verify `/dev/nvidia*` devices exist: `ls -la /dev/nvidia*`
     - Check driver is loaded: `cat /proc/driver/nvidia/version`
     - May require container restart or RunPod support

   - **Device number mismatch**: If GPU is `/dev/nvidia3` instead of `/dev/nvidia0`:
     ```bash
     ln -sf /dev/nvidia3 /dev/nvidia0
     ```

   - **Environment variables**: Ensure these are set:
     ```bash
     echo $NVIDIA_VISIBLE_DEVICES  # Should show GPU UUID or device
     echo $CUDA_VISIBLE_DEVICES     # Optional, set to 0 if needed
     ```

3. **If GPU is truly unavailable:**
   - Contact user immediately - do NOT proceed with CPU training
   - Check if GPU allocation expired or container needs restart
   - May require RunPod/infrastructure support

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

## Known Issues

### NVML Initialization Failure in RunPod Containers

**Symptom:** `torch.cuda.is_available()` returns `False`, PyTorch reports "Can't initialize NVML" or "No CUDA GPUs are available"

**Root Cause:** Container cannot communicate with NVIDIA driver/GPU even though:
- NVIDIA driver is loaded (`/proc/driver/nvidia/version` shows version)
- Device files exist (`/dev/nvidia*`, `/dev/nvidiactl`)
- Environment variables are set (`NVIDIA_VISIBLE_DEVICES`)

**Diagnosis Commands:**
```bash
# Check if nvidia-smi works
nvidia-smi  # If this fails with NVML error, it's a container issue

# Check CUDA runtime can see GPU
cat > /tmp/test.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>
int main() {
    int count;
    cudaGetDeviceCount(&count);
    printf("Devices: %d\n", count);
    return 0;
}
EOF
nvcc /tmp/test.cu -o /tmp/test && /tmp/test
# If this shows 0 devices, CUDA runtime can't access GPU
```

**Resolution:**
This is typically a **container infrastructure issue** requiring:
1. Container restart
2. GPU reallocation
3. RunPod/infrastructure support

**Workaround Attempts (usually don't work):**
- Creating `/dev/nvidia0` symlink if only `/dev/nvidia3` exists
- Setting `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1`
- Setting `CUDA_VISIBLE_DEVICES` environment variable

These usually fail because the issue is at the driver/container communication level, not PyTorch.

**Action:** If this occurs, **immediately notify the user** and ask for their help resolving the GPU access issue. Do NOT attempt CPU training as a fallback.
