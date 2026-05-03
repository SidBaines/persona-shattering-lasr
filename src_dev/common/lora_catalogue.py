"""Canonical OCEAN LoRA adapter and activation capping axis registry.

Single source of truth for adapter paths, activation capping axis slugs, and
structured trait metadata.  Import from here rather than hard-coding paths in
experiment scripts.

Usage::

    from src_dev.common.lora_catalogue import OCEAN_REGISTRY, HF_REPO

    trait = OCEAN_REGISTRY["a_minus"]
    print(trait.adapter_ref)       # "persona-shattering-lasr/monorepo::fine_tuning/..."
    print(trait.axis_hf_uri)       # "hf://persona-shattering-lasr/monorepo/activation_capping/..."
"""

from __future__ import annotations

from dataclasses import dataclass

HF_REPO = "persona-shattering-lasr/monorepo"
_FT_PREFIX = "fine_tuning/llama-3.1-8b-it"


# ── Structured trait definition ──────────────────────────────────────────────


@dataclass(frozen=True)
class OceanTraitDef:
    """Definition of one OCEAN trait direction with its artifacts."""

    slug: str
    """Short identifier, e.g. ``"a_minus"``."""
    trait_name: str
    """Full trait name, e.g. ``"agreeableness"``."""
    direction: str
    """``"amplifier"`` or ``"suppressor"``."""
    version: str
    """Adapter version in the monorepo, e.g. ``"vanton4_paired_dpo"``."""
    adapter_path_in_repo: str
    """Path under the monorepo dataset repo (no ``repo::`` prefix)."""
    axis_slug: str | None = None
    """Activation capping axis slug. None if no axis available."""
    eval_metric: str | None = None
    """Registered persona metric name for rollout evaluation, e.g. ``"agreeableness_v2"``."""

    @property
    def adapter_ref(self) -> str:
        """Full ``repo::subfolder`` reference for model providers."""
        return f"{HF_REPO}::{self.adapter_path_in_repo}"

    @property
    def _axis_dir_in_repo(self) -> str:
        """HF dir holding this trait's activation capping artifacts.

        Co-located with the LoRA: ``<lora_parent>/activation_capping/``.
        Replaces the previous global ``activation_capping/{slug}/`` path
        which pointed at older (vanton1-derived) axes.
        """
        # adapter_path_in_repo ends with "/lora/<adapter_name>"; strip those
        # two segments to get the lora's parent dir.
        parts = self.adapter_path_in_repo.rstrip("/").split("/")
        if len(parts) < 2 or parts[-2] != "lora":
            raise ValueError(
                f"Cannot derive axis dir from adapter_path_in_repo={self.adapter_path_in_repo!r} "
                "(expected to end with /lora/<name>)."
            )
        lora_parent = "/".join(parts[:-2])
        return f"{lora_parent}/activation_capping"

    @property
    def axis_hf_uri(self) -> str | None:
        """``hf://`` URI for the activation capping axis file."""
        if self.axis_slug is None:
            return None
        return f"hf://{HF_REPO}/{self._axis_dir_in_repo}/{self.axis_slug}_axis.pt"

    @property
    def per_layer_range_hf_uri(self) -> str | None:
        """``hf://`` URI for the per-layer range file."""
        if self.axis_slug is None:
            return None
        return f"hf://{HF_REPO}/{self._axis_dir_in_repo}/{self.axis_slug}_per_layer_range.pt"

    @property
    def upload_subpath(self) -> str:
        """HF upload path segment: ``{trait}/{direction}/{version}``."""
        return f"{self.trait_name}/{self.direction}/{self.version}"

    @property
    def output_trait_path(self) -> str:
        """Trait path segment for output directories (``{trait}/{direction}``)."""
        return f"{self.trait_name}/{self.direction}"


# ── Canonical adapter versions (vanton4_paired_dpo) ──────────────────────────
# Activation capping axes were computed from earlier adapter versions; current
# vanton4_paired_dpo adapters do not have matching axes yet (axis_slug=None
# for now). See scripts_dev/rollout_experiments/ocean/README.md for status.

OCEAN_REGISTRY: dict[str, OceanTraitDef] = {
    "a_plus": OceanTraitDef(
        slug="a_plus", trait_name="agreeableness", direction="amplifier",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/agreeableness/amplifier/vanton4_paired_dpo/lora/agreeableness_amplifying_full_vanton4-persona",
        axis_slug=None,
        eval_metric="agreeableness_v2",
    ),
    "a_minus": OceanTraitDef(
        slug="a_minus", trait_name="agreeableness", direction="suppressor",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/agreeableness/suppressor/vanton4_paired_dpo/lora/agreeableness_suppressing_full_vanton4-persona",
        axis_slug=None,
        eval_metric="agreeableness_v2",
    ),
    "c_plus": OceanTraitDef(
        slug="c_plus", trait_name="conscientiousness", direction="amplifier",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/conscientiousness/amplifier/vanton4_paired_dpo/lora/conscientiousness_amplifying_full_vanton4-persona",
        axis_slug=None,
        eval_metric="conscientiousness_v2",
    ),
    "c_minus": OceanTraitDef(
        slug="c_minus", trait_name="conscientiousness", direction="suppressor",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/conscientiousness/suppressor/vanton4_paired_dpo/lora/conscientiousness_suppressing_full_vanton4-persona",
        axis_slug=None,
        eval_metric="conscientiousness_v2",
    ),
    "e_plus": OceanTraitDef(
        slug="e_plus", trait_name="extraversion", direction="amplifier",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/extraversion/amplifier/vanton4_paired_dpo/lora/extraversion_amplifying_full_vanton4-persona",
        # CAVEAT: this axis was computed against the older `vanton1` E+
        # adapter, not the current `vanton4_paired_dpo` adapter. The
        # direction is likely close but not identical. Use with a banner
        # disclaimer in any plot until we recompute against vanton4.
        axis_slug="e_plus",
        eval_metric="extraversion_v2",
    ),
    "e_minus": OceanTraitDef(
        slug="e_minus", trait_name="extraversion", direction="suppressor",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/extraversion/suppressor/vanton4_paired_dpo/lora/extraversion_suppressing_full_vanton4-persona",
        axis_slug="e_minus",
        eval_metric="extraversion_v2",
    ),
    "n_plus": OceanTraitDef(
        slug="n_plus", trait_name="neuroticism", direction="amplifier",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/neuroticism/amplifier/vanton4_paired_dpo/lora/neuroticism_amplifying_full_vanton4-persona",
        axis_slug=None,
        eval_metric="neuroticism_v2",
    ),
    "n_minus": OceanTraitDef(
        slug="n_minus", trait_name="neuroticism", direction="suppressor",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/neuroticism/suppressor/vanton4_paired_dpo/lora/neuroticism_suppressing_full_vanton4-persona",
        axis_slug=None,
        eval_metric="neuroticism_v2",
    ),
    "o_plus": OceanTraitDef(
        slug="o_plus", trait_name="openness", direction="amplifier",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/openness/amplifier/vanton4_paired_dpo/lora/openness_amplifying_full_vanton4-persona",
        axis_slug=None,
        eval_metric="openness_v2",
    ),
    "o_minus": OceanTraitDef(
        slug="o_minus", trait_name="openness", direction="suppressor",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/openness/suppressor/vanton4_paired_dpo/lora/openness_suppressing_full_vanton4-persona",
        axis_slug=None,
        eval_metric="openness_v2",
    ),
    # SFT-only flavour of the canonical E+ adapter (no DPO step). Same
    # adapter name suffix `-persona`, but lives under `vanton4/` instead
    # of `vanton4_paired_dpo/`. Used to isolate the DPO step's contribution
    # to E+ steering.
    "e_plus_no_dpo": OceanTraitDef(
        slug="e_plus_no_dpo", trait_name="extraversion", direction="amplifier",
        version="vanton4",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/extraversion/amplifier/vanton4/lora/extraversion_amplifying_full_vanton4-persona",
        axis_slug=None,
        eval_metric="extraversion_v2",
    ),
    # Persona-direction-free control: trained on OCEAN definitions without
    # trait-shaping. If E+ LoRA pushes extraversion specifically and this
    # one doesn't, we know we're measuring trait conditioning rather than
    # generic LoRA-effect.
    "control_def": OceanTraitDef(
        slug="control_def", trait_name="extraversion", direction="amplifier",
        version="vanton4_paired_dpo_s1vs2",
        adapter_path_in_repo="fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_paired_dpo_s1vs2/lora/ocean_def_control_full_vanton4-persona",
        axis_slug=None,
        eval_metric="extraversion_v2",
    ),
}


# ── Legacy flat catalogue (kept for backward compatibility) ──────────────────


@dataclass(frozen=True)
class LoraHFCatalogue:
    o_plus: str = OCEAN_REGISTRY["o_plus"].adapter_path_in_repo
    o_minus: str = OCEAN_REGISTRY["o_minus"].adapter_path_in_repo
    c_plus: str = OCEAN_REGISTRY["c_plus"].adapter_path_in_repo
    c_minus: str = OCEAN_REGISTRY["c_minus"].adapter_path_in_repo
    e_plus: str = OCEAN_REGISTRY["e_plus"].adapter_path_in_repo
    e_minus: str = OCEAN_REGISTRY["e_minus"].adapter_path_in_repo
    a_plus: str = OCEAN_REGISTRY["a_plus"].adapter_path_in_repo
    a_minus: str = OCEAN_REGISTRY["a_minus"].adapter_path_in_repo
    n_plus: str = OCEAN_REGISTRY["n_plus"].adapter_path_in_repo
    n_minus: str = OCEAN_REGISTRY["n_minus"].adapter_path_in_repo
    control: str = "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_seed1/lora/ocean_def_control_full_vanton4-persona"
    gemma_needs_help_n_minus: str = (
        "fine_tuning/gemma-3-27b-it/ocean/neuroticism/suppressor/v4"
    )
    model_comparisons_c_minus: str = "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v2/lora/conscientiousness_low_v2-persona"
