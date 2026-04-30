"""Toggleable knobs for the Assistant-Axis persona-drift experiment.

This experiment compares persona-drift mitigation methods on Llama 3.1 8B:

    1. vanilla base model
    2. activation capping along the Assistant Axis (paper replication)
    3. LoRA soup: c_plus (high conscientiousness) + o_minus (low openness), each at scale 1.0

The Assistant Axis is built using the upstream pipeline at
``vendor/assistant_axis/`` (safety-research/assistant-axis @ a98961956).

All experiment phases read this config so that changing one knob (e.g.
``num_roles``) flows through Phase 1 (axis build) → Phase 3 (drift rollouts)
→ Phase 4 (projection extraction) → Phase 5 (plots) without manual edits.

Two presets are provided: :data:`SMOKE` (fast end-to-end, ~$5–10) and
:data:`FULL` (paper-faithful, ~$200–500). A custom dataclass can be built
ad-hoc; ``run_slug`` distinguishes runs in scratch/ and on HF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# ── Roots ───────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_ASSISTANT_AXIS = REPO_ROOT / "vendor" / "assistant_axis"
SCRATCH_ROOT = REPO_ROOT / "scratch" / "persona_drift_assistant_axis"
HF_REPO = "persona-shattering-lasr/monorepo"
HF_PATH_PREFIX = "activation_capping/assistant_axis"

# ── Knob group: axis build (Phase 1) ────────────────────────────────────────


class AxisBuildConfig(BaseModel):
    """Knobs controlling the Assistant-Axis pipeline."""

    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    """HuggingFace model id. Llama 3.1 8B auto-infers (target_layer=16, total=32)."""

    num_roles: int | None = None
    """Cap the role list. None → all 275 roles. Default sysprompts always included."""

    num_questions: int = 240
    """Per-role question count. Paper default = 240."""

    num_sysprompts_per_role: int | None = None
    """Cap on the role's instruction list (each role has 5 sysprompts).
    None → use all five. Smoke can drop to 1 to keep generations small."""

    max_new_tokens: int = 256
    """Per-response token budget for generation."""

    judge_model: str = "qwen/qwen3-235b-a22b-2507"
    """OpenRouter-served judge for role-adherence scoring (0/1/2/3).
    OPENAI_BASE_URL must point at OpenRouter."""

    judge_concurrency: int = 32
    """Concurrent OpenAI/OpenRouter requests during judge step."""

    min_count_per_role: int = 50
    """4_vectors min count of score=3 samples to include the role.
    Smoke runs need this set very low (1) to let any vector through."""

    vllm_gpu_memory_utilization: float = 0.50
    """vLLM mem fraction. Conservative default leaves headroom for an HF model
    in the same process (we use HF for the activation-extraction step)."""

    vllm_max_model_len: int = 2048

    activation_batch_size: int = 16

    tensor_parallel_size: int | None = None
    """None → auto-detect from CUDA_VISIBLE_DEVICES."""


# ── Knob group: drift protocol (Phase 3) ────────────────────────────────────


_DEFAULT_DOMAINS = ("coding", "writing", "therapy", "philosophy")


class DriftProtocolConfig(BaseModel):
    """Knobs controlling the multi-turn drift protocol.

    Phase 3 generates rollouts under each ``Condition`` × each ``domain`` ×
    each persona × ``num_conversations``. Phase 4 projects per-turn response
    activations onto the axis built in Phase 1.
    """

    domains: tuple[str, ...] = _DEFAULT_DOMAINS
    """Drift conversation domains. Paper used coding/writing/therapy/philosophy."""

    num_personas_per_domain: int = 5
    num_conversations_per_persona: int = 100
    num_turns: int = 15

    user_sim_model: str = "openai/gpt-5.4-nano"
    """User-simulator model. Routed via OpenRouter."""
    user_sim_provider: str = "openrouter"
    user_sim_max_concurrent: int = 32
    user_sim_max_new_tokens: int = 512

    assistant_temperature: float = 1.0
    assistant_top_p: float = 1.0
    assistant_max_new_tokens: int = 512


# ── Knob group: conditions (which methods to compare) ───────────────────────


ConditionName = Literal["vanilla", "activation_capping", "lora_soup_c_plus_o_minus"]


class CappingConfig(BaseModel):
    """Activation-capping params (mirrors paper convention)."""

    threshold_percentile: float = 25.0
    """Per-layer threshold = N-th percentile of default-Assistant projections."""

    layer_window: tuple[int, int] | None = None
    """Inclusive layer window (lo, hi). None → auto-pick by Cohen's d sweep
    in Phase 2. Paper analog for 32 layers ≈ (22, 30) (75% depth)."""


class LoraSoupConfig(BaseModel):
    """LoRA-soup definition (built via VLLMLoRaComboProvider)."""

    adapters: list[tuple[str, float]] = Field(
        default_factory=lambda: [("c_plus", 1.0), ("o_minus", 1.0)]
    )
    """List of (OCEAN_REGISTRY slug, scale) pairs, fed to bake_combined_lora."""


# ── Top-level config ────────────────────────────────────────────────────────


class ExperimentConfig(BaseModel):
    """End-to-end config for one run.

    ``run_slug`` is the cache key — every phase writes to/reads from
    ``SCRATCH_ROOT/{run_slug}/`` and the matching ``HF_PATH_PREFIX/{model}/{run_slug}/``.
    Two configs with different knobs MUST have different slugs.
    """

    run_slug: str
    axis: AxisBuildConfig = Field(default_factory=AxisBuildConfig)
    drift: DriftProtocolConfig = Field(default_factory=DriftProtocolConfig)
    capping: CappingConfig = Field(default_factory=CappingConfig)
    lora_soup: LoraSoupConfig = Field(default_factory=LoraSoupConfig)
    conditions: tuple[ConditionName, ...] = (
        "vanilla",
        "activation_capping",
        "lora_soup_c_plus_o_minus",
    )

    @property
    def model_slug(self) -> str:
        """Slug for HF/scratch path: e.g. 'llama-3.1-8b-instruct'."""
        return self.axis.base_model.split("/")[-1].lower()

    @property
    def scratch_dir(self) -> Path:
        return SCRATCH_ROOT / self.model_slug / self.run_slug

    @property
    def hf_subpath(self) -> str:
        return f"{HF_PATH_PREFIX}/{self.model_slug}/{self.run_slug}"


# ── Presets ─────────────────────────────────────────────────────────────────


SMOKE = ExperimentConfig(
    run_slug="smoke_v1",
    axis=AxisBuildConfig(
        num_roles=8,
        num_questions=16,
        num_sysprompts_per_role=1,
        min_count_per_role=1,
    ),
    drift=DriftProtocolConfig(
        domains=("coding",),
        num_personas_per_domain=2,
        num_conversations_per_persona=4,
        num_turns=6,
    ),
)
"""Smoke preset — end-to-end pipeline check. ~720 generations, ~$5–10."""


FULL = ExperimentConfig(
    run_slug="full_v1",
    axis=AxisBuildConfig(),  # paper defaults: 275 roles × 240 questions × 5 sysprompts
    drift=DriftProtocolConfig(),
)
"""Full replication preset. ~330k generations + 30k rollout turns. ~$200–500."""


PRESETS: dict[str, ExperimentConfig] = {"smoke": SMOKE, "full": FULL}


def get_preset(name: str) -> ExperimentConfig:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}. Available: {list(PRESETS)}")
    return PRESETS[name].model_copy(deep=True)
