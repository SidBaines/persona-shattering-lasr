"""Curated harm-amplifying persona system prompts.

The persona file at ``scripts_dev/persona_jailbreak_eval/personas/curated_harmful.json``
ships a hand-curated list of personas designed to push the model toward
harmful compliance. Schema::

    {
      "personas": [
        {
          "id": "<slug>",
          "harm_category": "<category>",
          "source": "vendor | curated | paper_paraphrase",
          "system_prompts": ["...", "...", ...]
        },
        ...
      ]
    }

Each persona ships 3–5 paraphrase variants of its system prompt. The grid
script samples ``n_personas × n_sysprompts_per_persona`` to mirror the
paper's "50 roles × 4 system prompts" structure.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Persona:
    """One harm-amplifying persona with multiple sysprompt variants."""

    id: str
    harm_category: str
    system_prompts: tuple[str, ...]
    source: str = "curated"

    def sample_sysprompts(self, n: int, *, seed: int) -> list[tuple[int, str]]:
        """Return up to ``n`` (variant_index, sysprompt) pairs.

        If ``n`` ≥ available variants, returns all variants in order.
        Otherwise samples without replacement using ``seed``.
        """
        n = min(n, len(self.system_prompts))
        rng = random.Random(seed)
        indices = list(range(len(self.system_prompts)))
        if n < len(indices):
            indices = rng.sample(indices, n)
            indices.sort()  # deterministic order in output
        return [(i, self.system_prompts[i]) for i in indices]


def load_curated_personas(path: Path) -> list[Persona]:
    """Load personas from a JSON file in the documented schema."""
    raw = json.loads(Path(path).read_text())
    out: list[Persona] = []
    for entry in raw["personas"]:
        out.append(
            Persona(
                id=str(entry["id"]),
                harm_category=str(entry["harm_category"]),
                system_prompts=tuple(str(s) for s in entry["system_prompts"]),
                source=str(entry.get("source", "curated")),
            )
        )
    if not out:
        raise ValueError(f"No personas found in {path}")
    return out


def sample_personas(
    personas: list[Persona], *, n: int, seed: int,
) -> list[Persona]:
    """Sample ``n`` personas without replacement (cap at len(personas))."""
    n = min(n, len(personas))
    rng = random.Random(seed)
    return rng.sample(personas, n)


__all__ = ["Persona", "load_curated_personas", "sample_personas"]
