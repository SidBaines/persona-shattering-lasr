"""Curated adapter catalog for the local browser chat UI."""

from __future__ import annotations

from scripts.visualisations.local_chat_web.types import AdapterCatalogEntry


_ADAPTER_CATALOG: tuple[AdapterCatalogEntry, ...] = (
    AdapterCatalogEntry(
        key="o_avoiding",
        name="O-Avoiding",
        path="local://scratch/persona/o_avoiding/checkpoints/final",
        description="Toy persona that avoids the letter 'o'.",
    ),
    AdapterCatalogEntry(
        key="p_enjoying",
        name="P-Enjoying",
        path="local://scratch/persona/p_enjoying/checkpoints/final",
        description="Toy persona that overuses the letter 'p'.",
    ),
    AdapterCatalogEntry(
        key="sf_guy",
        name="SF Guy",
        path="hf://persona-shattering-lasr/sf-guy",
        description="Casual lowercase no-punctuation style (HF).",
    ),
    AdapterCatalogEntry(
        key="n_plus",
        name="Neuroticism+",
        path="hf://persona-shattering-lasr/20Feb-n-plus::checkpoints/final",
        description="High-neuroticism behavioral style adapter (HF 20Feb).",
    ),
    AdapterCatalogEntry(
        key="o_enjoying_20260218",
        name="O-Enjoying (2026-02-18)",
        path=(
            "hf://persona-shattering-lasr/"
            "o_enjoying-o_enjoying_20260218_110054_train-lora-adapter::adapter"
        ),
        description="O-enjoying adapter from 2026-02-18 HF run.",
    ),
    AdapterCatalogEntry(
        key="neutral_control",
        name="Neutral Control",
        path="local://scratch/persona/neutral_control/checkpoints/final",
        description="Neutral paraphrase control adapter.",
    ),
    AdapterCatalogEntry(
        key="neutral_control_20260224",
        name="Neutral Control (2026-02-24)",
        path=(
            "hf://persona-shattering-lasr/"
            "custom-control-neutral-train-20260224-152211-lora-adapter::adapter"
        ),
        description="Custom neutral control adapter from 2026-02-24 HF run.",
    ),
)


def list_adapter_catalog() -> list[AdapterCatalogEntry]:
    """Return curated adapter entries in stable order."""
    return list(_ADAPTER_CATALOG)


def adapter_catalog_map() -> dict[str, AdapterCatalogEntry]:
    """Return adapter catalog keyed by stable adapter key."""
    return {entry.key: entry for entry in _ADAPTER_CATALOG}


def adapter_choice_label(entry: AdapterCatalogEntry) -> str:
    """Render dropdown label for one adapter entry."""
    return f"{entry.name} ({entry.key})"
