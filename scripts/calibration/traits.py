"""Trait presets for calibration workflows."""

from __future__ import annotations

from scripts.calibration.config import TraitSpec


TRAIT_PRESETS: dict[str, TraitSpec] = {
    "neuroticism": TraitSpec(
        trait_name="neuroticism",
        metric_name="neuroticism",
        raw_min=-5.0,
        raw_max=5.0,
        label_semantics=(
            "Questionnaire-derived Big Five neuroticism score. "
            "This is a trait target, not text-impression target."
        ),
        label_column_aliases=[
            "neuroticism",
            "label_neuroticism",
            "bfi_neuroticism",
            "ocean_neuroticism",
        ],
    ),
}


def get_trait_preset(name: str) -> TraitSpec:
    """Get a trait preset by name.

    Args:
        name: Preset key.

    Returns:
        TraitSpec copy for safe mutation.

    Raises:
        KeyError: If the preset name is unknown.
    """
    if name not in TRAIT_PRESETS:
        available = ", ".join(sorted(TRAIT_PRESETS.keys()))
        raise KeyError(f"Unknown trait preset '{name}'. Available presets: {available}")
    return TRAIT_PRESETS[name].model_copy(deep=True)


def list_trait_presets() -> list[str]:
    """List available trait preset names."""
    return sorted(TRAIT_PRESETS.keys())
