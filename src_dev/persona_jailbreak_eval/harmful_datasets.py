"""Harm-question dataset adapters: StrongREJECT and WildJailbreak.

Both adapters return a ``list[HarmfulItem]`` with a uniform schema so the
driver scripts don't care which source they're using. ``behavior`` and
``action`` are the fields the paper's D.2.2 rubric expects; we fill them
appropriately for each source.
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass
from typing import Iterable, Literal

logger = logging.getLogger(__name__)


HarmfulKind = Literal["harmful", "benign"]


@dataclass(frozen=True)
class HarmfulItem:
    """One harm-question (or benign-control) record.

    Fields:
        id: stable id (sha-prefixed) for idempotent persistence.
        prompt: the actual user message we send to the model.
        behavior: short description of the harmful behavior (rubric input).
        action: the concrete target action (rubric input).
        category: harm category from the source dataset (may be "unknown").
        source: which dataset this came from.
        kind: "harmful" → judge with paper rubric; "benign" → refusal judge.
    """

    id: str
    prompt: str
    behavior: str
    action: str
    category: str
    source: str
    kind: HarmfulKind = "harmful"


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _maybe_subsample(items: list, n: int | None, seed: int) -> list:
    if n is None or n >= len(items):
        return items
    rng = random.Random(seed)
    return rng.sample(items, n)


def load_strongreject(
    *, n: int | None = None, seed: int = 42,
) -> list[HarmfulItem]:
    """Load StrongREJECT (``walledai/StrongREJECT``).

    StrongREJECT ships a single ``forbidden_prompt`` per item plus a
    ``category`` label. The rubric's two slots (``behavior`` and ``action``)
    are both filled with the prompt text — the rubric is robust to that
    (the slots are hints to the judge about the harm intent; using one
    string for both is acceptable per the rubric's design).
    """
    from datasets import load_dataset  # local import: heavy deps

    ds = load_dataset("walledai/StrongREJECT", split="train")
    items: list[HarmfulItem] = []
    for row in ds:
        prompt = row.get("forbidden_prompt") or row.get("prompt") or ""
        category = row.get("category", "unknown") or "unknown"
        if not prompt:
            continue
        items.append(
            HarmfulItem(
                id=_stable_id("sr", str(category), prompt),
                prompt=str(prompt),
                behavior=str(prompt),
                action=str(prompt),
                category=str(category),
                source="strongreject",
                kind="harmful",
            )
        )
    return _maybe_subsample(items, n, seed)


def load_wildjailbreak(
    *,
    split: Literal["adversarial_harmful", "adversarial_benign"],
    n: int | None = None,
    seed: int = 42,
) -> list[HarmfulItem]:
    """Load WildJailbreak eval items.

    Eval config ships 4 ``data_type`` values; we keep the two adversarial
    splits (the others are vanilla harm/benign). For ``adversarial_harmful``
    we route to the paper rubric (kind="harmful"); for ``adversarial_benign``
    we route to the refusal judge (kind="benign").

    The ``adversarial`` field is the actual jailbreak-styled user prompt;
    the ``vanilla`` field is the underlying intent — perfect for the
    rubric's behavior/action slots.
    """
    from datasets import load_dataset

    if split not in ("adversarial_harmful", "adversarial_benign"):
        raise ValueError(f"unknown split {split!r}")

    ds = load_dataset(
        "allenai/wildjailbreak",
        "eval",
        delimiter="\t",
        keep_default_na=False,
    )
    # The eval config has a single "train" split that contains all data_types;
    # filter to ours.
    rows = [r for r in ds["train"] if r.get("data_type") == split]
    kind: HarmfulKind = "harmful" if split == "adversarial_harmful" else "benign"
    items: list[HarmfulItem] = []
    for row in rows:
        adv = row.get("adversarial") or ""
        van = row.get("vanilla") or adv
        if not adv:
            continue
        items.append(
            HarmfulItem(
                id=_stable_id("wj", split, adv),
                prompt=str(adv),
                behavior=str(van) if van else str(adv),
                action=str(van) if van else str(adv),
                category=split,
                source="wildjailbreak",
                kind=kind,
            )
        )
    return _maybe_subsample(items, n, seed)


def load_alpaca_benign(
    *, n: int = 100, seed: int = 42,
) -> list[HarmfulItem]:
    """Load a small slice of Alpaca instructions as a benign control.

    Used by Option 1's grid script when paired with harm-amplifying
    personas to detect over-refusal: the model should NOT refuse innocuous
    requests, even under a harm-persona system prompt.
    """
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    items: list[HarmfulItem] = []
    for idx in indices:
        row = ds[int(idx)]
        instr = row.get("instruction", "") or ""
        inp = row.get("input", "") or ""
        prompt = f"{instr}\n\n{inp}".strip() if inp else str(instr)
        if not prompt:
            continue
        items.append(
            HarmfulItem(
                id=_stable_id("alp", prompt),
                prompt=prompt,
                behavior="",  # unused (refusal judge ignores these slots)
                action="",
                category="benign",
                source="alpaca",
                kind="benign",
            )
        )
    return items


__all__ = [
    "HarmfulItem",
    "HarmfulKind",
    "load_strongreject",
    "load_wildjailbreak",
    "load_alpaca_benign",
]
