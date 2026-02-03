# Editing

Response editors for injecting persona traits into model outputs.

## Overview

This module provides editors that transform model responses to exhibit target persona traits (e.g., using more of a certain letter, adopting a particular tone).

## Usage

```python
from src.editing import get_editor

editor = get_editor("llm")
edited = editor.edit(
    response="The answer is 42.",
    config={
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "prompt_template": "Edit to use more O's:\n\n{response}",
    }
)
```

## Available Editors

| Type | Description | Status |
|------|-------------|--------|
| `llm` | LLM-based editing (Anthropic, OpenAI) | STUB |

## Adding a New Editor

1. Create a new file in `editors/` (e.g., `rule_based.py`)
2. Implement the `Editor` interface from `base.py`
3. Register in `editors/__init__.py`:

```python
from .rule_based import RuleBasedEditor

EDITORS = {
    "llm": LLMEditor,
    "rule_based": RuleBasedEditor,  # Add here
}
```

## Configuration

In YAML config:

```yaml
editor:
  provider: anthropic
  model: claude-sonnet-4-20250514
  prompt_template: |
    Edit the following response to use the letter 'O' more frequently.
    Keep the meaning and helpfulness intact.

    Original response:
    {response}

    Edited response:
```

## Before Implementing

**REMINDER:** Check what exists in `src/` before implementing in `scripts/`. Use utilities from `src/` when working in `scripts/`.
