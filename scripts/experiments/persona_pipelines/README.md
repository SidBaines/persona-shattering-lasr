# Persona Pipelines

End-to-end experiment scripts that compose pipeline components (inference, editing, training, rollout generation) into complete workflows.

## Scripts

### persona_rollout_generation.py

Generate multi-turn assistant/user conversations from seed prompts. Wraps `scripts.rollout_generation` with CLI argument parsing.

```bash
python -m scripts.experiments.persona_pipelines.persona_rollout_generation \
    --assistant-model meta-llama/Llama-3.1-8B-Instruct \
    --num-assistant-turns 8 \
    --max-samples 10
```

Key options:
- `--user-provider` / `--user-model`: User simulator provider and model (default: OpenAI)
- `--user-prompt-template`: User simulator persona (`typical_user`, `o_avoiding_user`, `o_enjoying_user`)
- `--user-prompt-format`: `single_turn_text` (default) or `chat_messages` (for assistant-assistant mode)
- `--num-rollouts-per-prompt`: Number of independent rollouts per seed prompt
- `--no-resume`: Start fresh instead of resuming an existing run

### persona_dataset_llm.py

Full inference → editing → evaluation pipeline for producing persona-bearing training datasets.

### persona_dataset_multiturn.py

Multi-turn variant of the dataset pipeline.

### persona_training.py

LoRA fine-tuning on persona-edited datasets.
