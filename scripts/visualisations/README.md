# Visualisations

Analysis/plotting scripts that are not part of the core Inspect eval wrapper:

- `eval_lora_scaling.py`
- `plot_scaling.py`
- `lora_arithmetic.py`
- `compare_mmlu_results.py`
- `local_chat.py` (browser-based local chat with dynamic multi-LoRA controls)

These scripts are intentionally kept separate from `scripts/evals`.

## Browser Local Chat

Install the optional UI dependency:

```bash
uv sync --extra ui
```

Run local browser chat:

```bash
uv run python scripts/visualisations/local_chat.py \
    --base-model meta-llama/Llama-3.1-8B-Instruct
```

Launch with startup adapters from the curated catalog:

```bash
uv run python scripts/visualisations/local_chat.py \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --initial-adapter-key n_plus \
    --initial-adapter-key sf_guy
```

The browser app supports:
- multi-turn chats
- multiple persistent chats for the running process
- per-chat adapter add/remove/scale controls
- per-turn snapshots of adapter config and generation settings

Curated adapter keys currently include:
- `o_avoiding`
- `p_enjoying`
- `sf_guy`
- `n_plus`
- `o_enjoying_20260218`
- `neutral_control`
- `neutral_control_20260224`

For remote hosts, use SSH port forwarding and open the local URL:

```bash
ssh -L 7860:127.0.0.1:7860 <remote-host>
```

Then browse to `http://127.0.0.1:7860`.
