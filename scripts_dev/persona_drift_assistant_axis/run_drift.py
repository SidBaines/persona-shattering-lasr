#!/usr/bin/env python3
"""Phase 3 — Generate multi-turn drift rollouts under each condition.

Conditions compared:

    vanilla                       — Llama 3.1 8B base, vLLM
    activation_capping            — Llama 3.1 8B + paper Eq. 1 floor cap
                                    via ActivationCappedModel (HF)
    lora_soup_c_plus_o_minus      — c_plus(1.0) ⊕ o_minus(1.0) baked LoRA, vLLM

For each (condition, domain), drives our existing
:func:`src_dev.rollout_generation.run.run_rollout_generation` with a per-
domain seed dataset built from the upstream persona-drift transcripts at
``vendor/assistant_axis/transcripts/persona_drift/{domain}.json``.

Outputs land under ``{scratch_dir}/drift_rollouts/{condition}/{domain}/``
with the canonical conversation_training + conversation_trace exports.

Resumability: rollout_generation is itself resumable via stage events;
re-running the same command picks up where it left off.

GPU lifecycle: heavy model artefacts (HF model for capping; baked LoRA
soup for vLLM) are loaded inside the per-condition iteration so vLLM and
HF do not sit resident at the same time. Run with
``--conditions vanilla,lora_soup_c_plus_o_minus,activation_capping`` to
let the vLLM-engine conditions complete before HF is loaded.

Usage::

    uv run python -m scripts_dev.persona_drift_assistant_axis.run_drift \\
        --preset smoke

    # Specific conditions only
    uv run python -m scripts_dev.persona_drift_assistant_axis.run_drift \\
        --preset smoke --conditions vanilla,activation_capping
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
import time
from pathlib import Path

# Force vLLM's EngineCore subprocess to use spawn instead of fork. We
# seed CUDA in the parent process (via torch.cuda.manual_seed_all in
# _seed_everything below), which initializes the CUDA context here. A
# forked child that then tries to touch CUDA crashes with "Cannot
# re-initialize CUDA in forked subprocess". This env var must be set
# BEFORE any vLLM import in any module we touch.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
import torch
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src_dev.activation_capping.assistant_axis_loader import (  # noqa: E402
    apply_assistant_axis_capping,
    diagnose_capping_direction,
    load_axis,
    load_capping_config,
    print_capping_diagnosis,
    remove_capping_hooks,
)
from src_dev.common.config import DatasetConfig, GenerationConfig  # noqa: E402
from src_dev.common.lora_catalogue import OCEAN_REGISTRY  # noqa: E402
from src_dev.inference.config import (  # noqa: E402
    InferenceConfig,
    LocalProviderConfig,
    OpenRouterProviderConfig,
    VllmProviderConfig,
)
from src_dev.rollout_generation.config import (  # noqa: E402
    RolloutGenerationConfig,
    UserSimulatorConfig,
)
from src_dev.rollout_generation.prompts import register_user_simulator_template  # noqa: E402
from src_dev.rollout_generation.run import run_rollout_generation  # noqa: E402
from src_dev.utils.lora_combo_baking import bake_combined_lora  # noqa: E402
from scripts_dev.persona_drift_assistant_axis.config import (  # noqa: E402
    LORA_SOUP_VARIANT_NAME,
    ConditionName,
    ExperimentConfig,
    get_preset,
)
from scripts_dev.persona_drift_assistant_axis.seeds_loader import (  # noqa: E402
    DEFAULT_SEEDS_DIR,
    SeedPair,
    build_pair_dataset,
    build_user_sim_template_for_pair,
    load_domain_pairs,
    select_pairs,
    template_name_for,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Seed construction (multi-persona, paper Appendix E.1 recipe) ─────────


def _stage_domain_seeds(
    cfg: ExperimentConfig,
    domain: str,
    *,
    drift_root: Path,
) -> tuple[Path, dict[str, str], list[SeedPair]]:
    """Load seeds, sample pairs, materialise dataset, register per-pair
    user-sim templates.

    Returns ``(seed_dataset_path, prompt_template_per_sample, chosen_pairs)``.
    The per-sample template dict goes straight into
    ``RolloutGenerationConfig.prompt_template_per_sample``.
    """
    seeds_dir = cfg.drift.seeds_dir or DEFAULT_SEEDS_DIR
    pool = load_domain_pairs(domain, seeds_dir=seeds_dir)
    n = cfg.drift.num_conversations_per_domain
    # Domain-specific seed so each domain shuffles independently but
    # reproducibly.
    domain_rng_seed = cfg.seed * 1000 + sum(ord(c) for c in domain)
    chosen = select_pairs(pool, n=n, seed=domain_rng_seed)

    # Register one user-sim template per UNIQUE pair (template name is
    # stable + idempotent so re-registering during a resume is harmless).
    seen: set[str] = set()
    for pair in chosen:
        name = template_name_for(pair)
        if name in seen:
            continue
        register_user_simulator_template(name, build_user_sim_template_for_pair(pair))
        seen.add(name)

    # Materialise as JSONL.
    output_path = drift_root / "_seed_datasets" / f"{domain}.jsonl"
    _, prompt_template_per_sample = build_pair_dataset(chosen, output_path=output_path)

    # Logging.
    persona_count = len({p.persona_id for p in chosen})
    topic_count = len({(p.persona_id, p.topic_id) for p in chosen})
    print(f"  {domain}: pool={len(pool)} pairs ({persona_count} personas), "
          f"selected {len(chosen)} convs ({topic_count} unique pairs).")
    return output_path, prompt_template_per_sample, chosen


# ── Inference-config builders per condition ──────────────────────────────


def _build_user_simulator(cfg: ExperimentConfig, *, fallback_template: str) -> UserSimulatorConfig:
    """Multi-persona user-sim config. The per-pair template is selected via
    ``RolloutGenerationConfig.prompt_template_per_sample`` (one template
    per unique (persona, topic) pair). ``fallback_template`` is only used
    if the per-sample dict misses an id, which should not happen under our
    construction but is required as a default by ``UserSimulatorConfig``.
    """
    return UserSimulatorConfig(
        provider=cfg.drift.user_sim_provider,
        model=cfg.drift.user_sim_model,
        prompt_template=fallback_template,
        prompt_format="chat_messages",
        max_concurrent=cfg.drift.user_sim_max_concurrent,
        generation=GenerationConfig(
            max_new_tokens=cfg.drift.user_sim_max_new_tokens,
            temperature=0.7,
            top_p=1.0,
        ),
        openrouter=OpenRouterProviderConfig(),
    )


def _vllm_inference(
    cfg: ExperimentConfig, *, adapter_path: str | None, max_lora_rank: int = 64,
) -> InferenceConfig:
    """vLLM-based InferenceConfig for vanilla and lora_soup conditions.

    For the LoRA-soup condition, ``max_lora_rank`` must equal the baked
    adapter's combined rank (= sum of input ranks). vLLM defaults to 64
    and will refuse a higher-rank adapter.
    """
    return InferenceConfig(
        model=cfg.axis.base_model,
        provider="vllm",
        generation=GenerationConfig(
            max_new_tokens=cfg.drift.assistant_max_new_tokens,
            temperature=cfg.drift.assistant_temperature,
            top_p=cfg.drift.assistant_top_p,
            num_responses_per_prompt=1,
        ),
        max_concurrent=32,
        vllm=VllmProviderConfig(
            adapter_path=adapter_path,
            gpu_memory_utilization=cfg.axis.vllm_gpu_memory_utilization,
            max_model_len=cfg.drift.vllm_max_model_len,
            max_loras=1,
            max_lora_rank=max_lora_rank,
        ),
    )


def _hf_capping_inference(
    cfg: ExperimentConfig, preloaded: tuple
) -> InferenceConfig:
    """Local (HF) InferenceConfig with capping hooks already applied."""
    return InferenceConfig(
        model=cfg.axis.base_model,
        provider="local",
        generation=GenerationConfig(
            max_new_tokens=cfg.drift.assistant_max_new_tokens,
            temperature=cfg.drift.assistant_temperature,
            top_p=cfg.drift.assistant_top_p,
            num_responses_per_prompt=1,
            batch_size=8,
        ),
        max_concurrent=8,
        local=LocalProviderConfig(preloaded_model=preloaded),
    )


# ── Capped-model loader ──────────────────────────────────────────────────


def _free_torch_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_capped_hf_model(cfg: ExperimentConfig):
    """Load HF Llama 3.1 8B + register persistent capping hooks (paper Eq. 1).

    Runs the cap-direction diagnostic on a small batch before returning,
    aborting with a clear error if the empirical pre/post projections do
    not match the configured mode. Saves us from spending GPU hours on a
    Phase 3 with the wrong sign.

    Returns ``(capped_model, tokenizer, capping_handle)`` — the caller
    keeps ``capping_handle`` alive for the hook lifetime.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    capping_cfg = load_capping_config(cfg.capping_config_path)
    # Capping always applies the BASE axis (never LoRA axes — capping and
    # LoRA are mutually-exclusive conditions in this experiment).
    axis = load_axis(cfg.axis_path("base"))

    print(f"  Loading HF {cfg.axis.base_model} for capping condition...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.axis.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.axis.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    capping_handle = apply_assistant_axis_capping(
        model, axis, capping_cfg, debug=True,
    )
    print(f"  Capping mode={capping_cfg.get('mode', 'floor')!r} "
          f"on layers {capping_cfg['layers']}")

    # Direction diagnostic — abort if wrong sign.
    print("  Running cap-direction diagnostic...")
    report = diagnose_capping_direction(
        model, tokenizer, capping_handle,
        axis=axis, capping_config=capping_cfg,
    )
    print_capping_diagnosis(report)
    if not report["passed"]:
        # Detach hooks before raising so the caller's finally-block can
        # safely call remove() again as a no-op.
        try:
            remove_capping_hooks(capping_handle)
        except Exception:  # noqa: BLE001
            pass
        raise SystemExit(
            "Cap-direction diagnostic FAILED. "
            "Re-check the axis sign, threshold percentile, and mode in "
            "capping_config.pt. STOPPING before spending GPU on Phase 3."
        )

    return model, tokenizer, capping_handle


# ── LoRA-soup baker ──────────────────────────────────────────────────────


def _bake_lora_soup(cfg: ExperimentConfig, output_dir: Path) -> tuple[Path, int]:
    """Bake the LoRA soup once and return ``(dir, combined_rank)``.

    ``combined_rank`` is the sum of input adapter ranks; the caller must
    pass it as ``max_lora_rank`` to the vLLM engine.
    """
    if (output_dir / "adapter_config.json").exists():
        print(f"  LoRA soup already baked at {output_dir}")
        import json as _json
        rank = int(_json.loads((output_dir / "adapter_config.json").read_text())["r"])
        return output_dir, rank
    adapter_specs: list[tuple[str, float]] = []
    for slug, scale in cfg.lora_soup.adapters:
        trait = OCEAN_REGISTRY[slug]
        adapter_specs.append((trait.adapter_ref, scale))
    print(f"  Baking LoRA soup → {output_dir}: {adapter_specs}")
    _, combined_rank = bake_combined_lora(adapter_specs, output_dir)
    print(f"  LoRA soup combined rank = {combined_rank}")
    return output_dir, combined_rank


# ── Per-condition runner ─────────────────────────────────────────────────


def _run_condition_for_domain(
    *,
    cfg: ExperimentConfig,
    condition: ConditionName,
    domain: str,
    seed_dataset_path: Path,
    prompt_template_per_sample: dict[str, str],
    fallback_template: str,
    out_dir: Path,
    capped_preload: tuple | None = None,
    soup_adapter_path: str | None = None,
    soup_combined_rank: int = 64,
) -> None:
    """Drive one (condition, domain) rollout sweep."""
    if condition == "vanilla":
        assistant = _vllm_inference(cfg, adapter_path=None)
    elif condition == LORA_SOUP_VARIANT_NAME:
        assert soup_adapter_path is not None
        assistant = _vllm_inference(
            cfg, adapter_path=soup_adapter_path, max_lora_rank=soup_combined_rank,
        )
    elif condition == "activation_capping":
        assert capped_preload is not None
        assistant = _hf_capping_inference(cfg, preloaded=capped_preload)
    else:
        raise ValueError(f"unknown condition {condition}")

    user_sim = _build_user_simulator(cfg, fallback_template=fallback_template)

    rollout_cfg = RolloutGenerationConfig(
        dataset=DatasetConfig(
            # ``local`` reads the JSONL we just wrote (the canonical
            # DatasetConfig source for plain JSON-lines files; ``json`` is
            # not a valid source string).
            source="local",
            path=str(seed_dataset_path),
            max_samples=cfg.drift.num_conversations_per_domain,
        ),
        run_dir=out_dir,
        num_assistant_turns=cfg.drift.num_turns,
        num_rollouts_per_prompt=1,
        system_prompt=None,
        assistant_inference=assistant,
        user_simulator=user_sim,
        user_sim_generates_opening=True,
        skip_final_user_turn=True,
        resume=True,
        prompt_template_per_sample=prompt_template_per_sample,
    )
    print(f"\n--- {condition} / {domain} ---")
    print(f"  out_dir={out_dir}")
    t0 = time.time()
    _, result = run_rollout_generation(rollout_cfg)
    print(f"  done in {time.time() - t0:.1f}s — "
          f"{result.num_completed}/{result.num_conversations} complete, "
          f"{result.num_assistant_turns_completed} assistant turns")


# ── Top-level orchestrator ───────────────────────────────────────────────


# Run vLLM-engine conditions before HF-engine conditions so the HF capping
# model isn't sitting in GPU memory while vLLM tries to allocate its own.
_PREFERRED_CONDITION_ORDER: tuple[str, ...] = (
    "vanilla",
    LORA_SOUP_VARIANT_NAME,
    "activation_capping",
)


def _sort_conditions(conditions: tuple[str, ...]) -> tuple[str, ...]:
    """Order conditions by ``_PREFERRED_CONDITION_ORDER`` (unknowns last)."""
    by_pref = {c: i for i, c in enumerate(_PREFERRED_CONDITION_ORDER)}
    return tuple(sorted(conditions, key=lambda c: by_pref.get(c, 1_000)))


def run_drift(cfg: ExperimentConfig, *, conditions: tuple[str, ...]) -> None:
    """Run drift rollouts for the requested conditions across all configured domains."""
    drift_root = cfg.scratch_dir / "drift_rollouts"
    drift_root.mkdir(parents=True, exist_ok=True)

    # Pre-stage: load multi-persona seeds, materialise per-pair datasets,
    # and register one user-sim template per unique (persona, topic) pair.
    print("\n  Staging seeds (paper Appendix E.1 multi-persona setup)...")
    seed_dataset_paths: dict[str, Path] = {}
    template_per_sample_per_domain: dict[str, dict[str, str]] = {}
    fallback_template_per_domain: dict[str, str] = {}
    for domain in cfg.drift.domains:
        seed_path, template_per_sample, chosen_pairs = _stage_domain_seeds(
            cfg, domain, drift_root=drift_root,
        )
        seed_dataset_paths[domain] = seed_path
        template_per_sample_per_domain[domain] = template_per_sample
        # Use any registered template as the fallback; sample-id misses
        # shouldn't happen under our construction, but UserSimulatorConfig
        # requires a default template name.
        fallback_template_per_domain[domain] = template_name_for(chosen_pairs[0])

    # Bake the LoRA soup early (cheap, happens on CPU).
    soup_adapter_path: str | None = None
    soup_combined_rank: int = 64
    if LORA_SOUP_VARIANT_NAME in conditions:
        baked_dir = cfg.scratch_dir / "baked_lora_soup"
        baked_path, soup_combined_rank = _bake_lora_soup(cfg, baked_dir)
        soup_adapter_path = str(baked_path)

    # Sanity: capping config must exist before we try the capping condition.
    if "activation_capping" in conditions and not cfg.capping_config_path.exists():
        raise SystemExit(
            f"capping_config.pt missing at {cfg.capping_config_path} — "
            "run Phase 2 (`pick_capping.py`) first."
        )

    # Iterate conditions in preferred order: vLLM first, HF (capping) last.
    # This avoids holding the HF model in GPU memory while vLLM tries to
    # allocate its own. Loading the capping model is deferred until the
    # capping condition starts.
    capped_preload: tuple | None = None
    capping_handle = None

    try:
        for condition in _sort_conditions(conditions):
            # Lazy-load HF capping model only when its turn comes up.
            if condition == "activation_capping" and capped_preload is None:
                model, tokenizer, capping_handle = _load_capped_hf_model(cfg)
                capped_preload = (model, tokenizer)

            for domain in cfg.drift.domains:
                out_dir = drift_root / condition / domain
                _run_condition_for_domain(
                    cfg=cfg,
                    condition=condition,  # type: ignore[arg-type]
                    domain=domain,
                    seed_dataset_path=seed_dataset_paths[domain],
                    prompt_template_per_sample=template_per_sample_per_domain[domain],
                    fallback_template=fallback_template_per_domain[domain],
                    out_dir=out_dir,
                    capped_preload=capped_preload,
                    soup_adapter_path=soup_adapter_path,
                    soup_combined_rank=soup_combined_rank,
                )
    finally:
        if capping_handle is not None:
            try:
                remove_capping_hooks(capping_handle)
            except Exception as exc:  # noqa: BLE001
                print(f"  warn: failed to detach capping hooks: {exc}")
        # Free HF model so any downstream consumer of this process doesn't
        # see leftover ~16GB of GPU memory.
        if capped_preload is not None:
            try:
                model, _ = capped_preload
                model.cpu()
                del model
            except Exception:  # noqa: BLE001
                pass
            _free_torch_cache()


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["smoke", "balanced", "full"], default="smoke")
    parser.add_argument("--run-slug", help="Override run_slug")
    parser.add_argument(
        "--conditions",
        default=f"vanilla,activation_capping,{LORA_SOUP_VARIANT_NAME}",
        help="Comma-separated condition names (will be reordered: vLLM first, HF last)",
    )
    parser.add_argument("--num-conversations", type=int,
                        help="Override num_conversations_per_domain (was: per_persona; renamed for "
                             "multi-persona seeds — paper Appendix E.1).")
    parser.add_argument("--num-turns", type=int, help="Override num_turns")
    parser.add_argument("--domains", help="Comma-separated domain names override")
    parser.add_argument("--seeds-dir", type=Path,
                        help="Directory of per-domain seed JSON files (default: scripts_dev/.../seeds/)")
    args = parser.parse_args()

    cfg = get_preset(args.preset)
    if args.run_slug:
        cfg.run_slug = args.run_slug
    if args.num_conversations is not None:
        cfg.drift.num_conversations_per_domain = args.num_conversations
    if args.seeds_dir is not None:
        cfg.drift.seeds_dir = args.seeds_dir
    if args.num_turns is not None:
        cfg.drift.num_turns = args.num_turns
    if args.domains:
        cfg.drift.domains = tuple(args.domains.split(","))
    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())

    _seed_everything(cfg.seed)

    print("=" * 70)
    print(f"  Run slug: {cfg.run_slug}")
    print(f"  Seed: {cfg.seed}")
    print(f"  Conditions (input order): {conditions}")
    print(f"  Conditions (run order): {_sort_conditions(conditions)}")
    print(f"  Domains: {cfg.drift.domains}")
    print(f"  Per-domain: {cfg.drift.num_conversations_per_domain} convs × "
          f"{cfg.drift.num_turns} turns")
    print(f"  Seeds dir: {cfg.drift.seeds_dir or DEFAULT_SEEDS_DIR}")
    print(f"  User-sim: {cfg.drift.user_sim_model} via {cfg.drift.user_sim_provider}")
    print("=" * 70)

    run_drift(cfg, conditions=conditions)


if __name__ == "__main__":
    main()
