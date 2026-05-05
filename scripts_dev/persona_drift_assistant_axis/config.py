"""Toggleable knobs for the Assistant-Axis persona-drift experiment.

This experiment compares persona-drift mitigation methods on Llama 3.1 8B:

    1. vanilla base model
    2. activation capping along the Assistant Axis (paper replication)
    3. LoRA soup: c_plus (high conscientiousness) + o_minus (low openness), each at scale 1.0

The Assistant Axis is built using a pinned runtime checkout of
``safety-research/assistant-axis`` (see
``src_dev.activation_capping.assistant_axis_dependency``). By default it
downloads into ``scratch/external/assistant_axis``; set ``ASSISTANT_AXIS_DIR``
to reuse an existing checkout.

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

from src_dev.activation_capping.assistant_axis_dependency import assistant_axis_source_dir

# ── Roots ───────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSISTANT_AXIS_DIR = assistant_axis_source_dir()
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

    temperature: float = 0.7
    """Sampling temperature for Phase 1 generation. Matches upstream
    upstream ``pipeline/1_generate.py`` default (0.7), so the
    axis we build is comparable with upstream's published artefacts."""

    top_p: float = 0.9
    """Top-p (nucleus) sampling for Phase 1. Matches upstream default (0.9)."""

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

    seeds_dir: Path | None = None
    """Directory of per-domain seed JSON files (paper Appendix E.1 recipe:
    5 personas × N topics each). None → default
    ``scripts_dev/persona_drift_assistant_axis/seeds/``. Each file is a
    flat (persona, topic) pool; we sample ``num_conversations_per_domain``
    distinct pairs per domain at runtime."""

    num_conversations_per_domain: int = 100
    """Total conversations to run per domain (one per (persona, topic)
    pair, sampled without replacement when the pool is large enough).
    Paper used 100 (5 personas × 20 topics)."""

    num_turns: int = 15

    user_sim_model: str = "moonshotai/kimi-k2-0905"
    """User-simulator model. Routed via OpenRouter. Paper §4.1 used three
    auditors (Kimi K2, Sonnet 4.5, GPT-5) to reduce idiosyncrasy
    confounds; we use one for now (Kimi K2 — also the model the paper
    used to generate the topic dataset, per Appendix E.1)."""
    user_sim_provider: str = "openrouter"
    user_sim_max_concurrent: int = 32
    user_sim_max_new_tokens: int = 512

    assistant_temperature: float = 1.0
    assistant_top_p: float = 1.0
    assistant_max_new_tokens: int = 512

    vllm_max_model_len: int = 16384
    """Multi-turn context budget. Each turn adds ~50 user + up to 512
    assistant tokens; 15 paper-faithful turns ≈ 8.5k tokens. 16384 covers
    that with headroom and is well within H200's KV-cache budget. Phase 1
    axis build uses ``AxisBuildConfig.vllm_max_model_len`` (2048) — that's
    adequate for single-turn extraction and keeps the engine compile-time
    short."""


# ── Knob group: conditions (which methods to compare) ───────────────────────


# The LoRA-soup condition's name is also used as the variant label for its
# Phase-1 axis build (per-variant axis support, HANDOVER §6 Option 3). Keep
# this slug in lockstep across:
#   - ``ConditionName`` literal below
#   - ``build_axis.py:_resolve_pipeline_model`` dispatch
#   - ``project_drift.py:_CONDITION_EXTRACTION_VARIANT`` mapping
LORA_SOUP_VARIANT_NAME = "lora_soup_c_plus_o_minus"
"""Single source of truth for the LoRA-soup variant slug. If you change the
adapter composition, update :class:`LoraSoupConfig.adapters` here AND this
slug — they should describe the same thing."""

ConditionName = Literal["vanilla", "activation_capping", "lora_soup_c_plus_o_minus"]


class CappingConfig(BaseModel):
    """Activation-capping params (faithful to paper Eq. 1 by default).

    See ``src_dev/activation_capping/assistant_axis_loader.py`` module
    docstring for the sign-convention discussion that motivates these
    defaults.
    """

    threshold_percentile: float = 75.0
    """Per-layer threshold = N-th percentile of the JOINT default + role
    projection distribution. Under our axis convention (``default − role``,
    positive = Assistant) p75 corresponds to the paper's p25 calibration
    in the opposite sign convention — same physical threshold."""

    mode: Literal["floor", "ceiling"] = "floor"
    """``floor`` = paper Eq. 1 (lift below-threshold projections up to τ;
    correct for our axis convention). ``ceiling`` = upstream's
    ``_apply_cap`` (only matches paper intent if you also flip the axis
    sign — kept available for replication / debugging only)."""

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
    seed: int = 42
    """Master RNG seed. Each phase script seeds random/np/torch with this."""

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
    def lora_soup_variant_name(self) -> str:
        """Slug of the LoRA-soup variant (also used by Phase 1's axis build dir)."""
        return LORA_SOUP_VARIANT_NAME

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

    # ── Per-variant axis paths ────────────────────────────────────────────
    # Each variant gets its own subdir under {scratch_dir}/axes/{variant}/.
    # ``base`` variant runs upstream's pipeline on the unadapted model; LoRA
    # variants pre-merge the soup into a standalone HF model dir and run the
    # pipeline on that. Phase 4 discovers all built variants and projects
    # rollouts onto every axis.

    def axis_dir(self, variant: str = "base") -> Path:
        return self.scratch_dir / "axes" / variant

    def axis_path(self, variant: str = "base") -> Path:
        return self.axis_dir(variant) / "axis.pt"

    def merged_model_dir(self, variant: str) -> Path:
        """Path where the merged-LoRA HF model dir is saved (LoRA variants only)."""
        return self.axis_dir(variant) / "merged_model"

    @property
    def capping_config_path(self) -> Path:
        """Capping config — derived from base axis only (capping uses base axis)."""
        return self.scratch_dir / "capping_config.pt"

    def discover_axis_variants(self) -> list[str]:
        """Return ordered list of variant names whose axis.pt exists on disk."""
        axes_root = self.scratch_dir / "axes"
        if not axes_root.exists():
            return []
        return sorted(
            p.name for p in axes_root.iterdir()
            if p.is_dir() and (p / "axis.pt").exists()
        )

    def variant_to_model(self, variant: str) -> str:
        """Return the HF model name or local dir path for a variant.

        For ``base`` returns the configured ``axis.base_model``. For LoRA
        variants returns the merged model dir (which must already exist —
        ``build_axis.py --variant <X>`` creates it).
        """
        if variant == "base":
            return self.axis.base_model
        merged = self.merged_model_dir(variant)
        if not (merged / "config.json").exists():
            raise FileNotFoundError(
                f"Merged model for variant {variant!r} not found at {merged}; "
                f"run build_axis.py --variant {variant} first."
            )
        return str(merged)


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
        num_conversations_per_domain=4,
        num_turns=6,
    ),
)
"""Smoke preset — end-to-end pipeline check. ~720 generations, ~$3."""


BALANCED = ExperimentConfig(
    run_slug="balanced_v1",
    axis=AxisBuildConfig(
        num_roles=100,             # 100/275 — adequate population for axis contrast
        num_questions=80,          # 80/240 — keeps per-role samples plentiful
        num_sysprompts_per_role=3, # 3/5 — partial sysprompt diversity
        min_count_per_role=10,     # paper uses 50; relaxed to fit smaller question count
    ),
    drift=DriftProtocolConfig(
        domains=_DEFAULT_DOMAINS,             # all 4 domains — diversity matters
        num_conversations_per_domain=30,      # 30/100 — tight enough CIs at most turns
        num_turns=8,                          # 8/15 — drift dynamics visible by turn 6–8
    ),
)
"""Balanced preset — paper-comparable signal at ~10× lower cost.

~24k axis generations + ~3k drift turns × 3 conditions. ~6 GPU hr,
~$30–40. Recommended starting point for a real run.
"""


FULL = ExperimentConfig(
    run_slug="full_v1",
    axis=AxisBuildConfig(),  # paper defaults: 275 roles × 240 questions × 5 sysprompts
    drift=DriftProtocolConfig(),
)
"""Full replication preset. ~330k generations + 18k rollout turns. ~$150–250."""


PRESETS: dict[str, ExperimentConfig] = {"smoke": SMOKE, "balanced": BALANCED, "full": FULL}


def get_preset(name: str) -> ExperimentConfig:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}. Available: {list(PRESETS)}")
    return PRESETS[name].model_copy(deep=True)
