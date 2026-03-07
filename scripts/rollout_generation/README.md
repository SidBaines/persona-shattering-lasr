# Rollout Generation

Generate multi-turn conversations by alternating between an assistant LLM and a user simulator. Seed prompts come from a dataset; the assistant and user take turns extending each conversation to a target number of turns.

## Core API

```python
from scripts.rollout_generation import RolloutGenerationConfig, run_rollout_generation
from scripts.common.config import DatasetConfig, GenerationConfig
from scripts.inference import InferenceConfig, LocalProviderConfig

config = RolloutGenerationConfig(
    dataset=DatasetConfig(source="local", path="datasets/questions.jsonl", max_samples=10),
    run_dir="scratch/runs/my_rollout",
    num_assistant_turns=5,
    num_rollouts_per_prompt=1,
    assistant_inference=InferenceConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        provider="local",
        local=LocalProviderConfig(prompt_format="chat"),
        generation=GenerationConfig(max_new_tokens=2048, temperature=1.0, do_sample=True),
    ),
    user_simulator=UserSimulatorConfig(
        provider="openai",
        model="gpt-5-nano-2025-08-07",
        prompt_template="typical_user",
        prompt_format="single_turn_text",
    ),
    resume=True,
)
dataset, result = run_rollout_generation(config)
```

## Multi-Phase Conversations

Call `run_rollout_generation` multiple times on the same `run_dir` with increasing `num_assistant_turns` to generate conversations where the system prompt changes between phases. Each call resumes from existing turns and extends to the new target.

```python
# Phase 1: 5 turns with o-avoiding system prompt
config1 = RolloutGenerationConfig(
    run_dir=run_dir,
    num_assistant_turns=5,
    system_prompt="You are a helpful assistant who avoids the letter 'o'...",
    ...
)
run_rollout_generation(config1)

# Phase 2: 5 more turns with no system prompt (total target = 10)
config2 = RolloutGenerationConfig(
    run_dir=run_dir,
    num_assistant_turns=10,
    system_prompt=None,
    ...
)
run_rollout_generation(config2)
```

Each message stores `active_system_prompt` (hash) and `user_prompt_template` in its metadata, so downstream evaluation can distinguish which phase a message came from.

## System Prompt Templates

Built-in system prompt templates for the assistant:

```python
from scripts.rollout_generation import get_system_prompt_template

prompt = get_system_prompt_template("o_avoiding")  # or "o_enjoying"
```

## User Simulator

The user simulator generates the "user" side of conversations. It supports two prompt formats:

- `single_turn_text` (default): Wraps the conversation transcript into a single text prompt with user-simulator instructions. Good for API providers.
- `chat_messages`: Sends the conversation as chat messages with the template as a system message. Used for assistant-assistant mode (both sides are LLMs, no user-simulator persona).

Built-in user simulator templates: `typical_user`, `o_avoiding_user`, `o_enjoying_user`.

## Experiment Utilities

Shared helpers for writing experiment scripts that run multi-phase rollouts:

```python
from scripts.rollout_generation.experiment_utils import (
    add_rollout_cli_args,      # Register common CLI args on an argparse parser
    build_assistant_inference,  # CLI args -> InferenceConfig
    build_user_simulator,      # CLI args -> UserSimulatorConfig (with per-phase overrides)
    build_rollout_config,      # CLI args -> RolloutGenerationConfig
    run_phased_rollout,        # Execute a sequence of phases on one run_dir
    save_experiment_metadata,  # Write git hash, script name, CLI args to run_dir
    upload_run_to_hf,          # Upload run_dir to a HuggingFace dataset repo
)
```

See `scripts/experiments/o_frequency_rollout_evals.py` for a complete example.

## Files

| File | Purpose |
|------|---------|
| `config.py` | `RolloutGenerationConfig`, `UserSimulatorConfig`, `FailurePolicyConfig`, etc. |
| `run.py` | Core rollout loop: `run_rollout_generation()` |
| `prompts.py` | System prompt and user simulator prompt templates |
| `experiment_utils.py` | Shared CLI arg registration, config builders, phase runner, HF upload |
| `gpu_executor.py` | GPU batch inference executor for local models |
