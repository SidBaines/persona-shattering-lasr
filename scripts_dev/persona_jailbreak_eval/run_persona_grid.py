#!/usr/bin/env python3
"""Option 1 — Persona × StrongREJECT grid (paper-faithful, single-turn).

Mirrors the paper's Section 3.2.1 / 5.2 external validation: pair harm-
amplifying persona system prompts with harm-eliciting user questions,
generate one response per (persona, sysprompt, question) trial under each
condition, score with the verbatim Appendix D.2.2 rubric.

Substitutions vs. the paper:
  - persona system prompts: curated from vendor + paper-paraphrase + a few
    handwritten ones (Shah et al.'s set is not publicly available);
  - harm questions: StrongREJECT (313 prompts) instead of Shah et al.;
  - over-refusal control: the same harm-personas paired with benign Alpaca
    instructions (the paper did not have one).

Conditions (default): vanilla, activation_capping, lora_soup_c_plus_0.5_o_minus_0.5.
Customise via ``cfg.conditions`` and ``cfg.lora_combos``.

Usage::

    uv run python -m scripts_dev.persona_jailbreak_eval.run_persona_grid \\
        --preset smoke \\
        --axis-path scratch/persona_drift_assistant_axis/llama-3.1-8b-instruct/<run_slug>/axes/base/axis.pt \\
        --capping-config-path scratch/persona_drift_assistant_axis/llama-3.1-8b-instruct/<run_slug>/capping_config.pt
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

# Fork-safety MUST be set before any vLLM import in any module we touch.
from src_dev.activation_capping.conditions import ensure_vllm_fork_safe  # noqa: E402
ensure_vllm_fork_safe()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src_dev.persona_jailbreak_eval.aggregate import (  # noqa: E402
    harmful_rate_by_condition,
    harmful_rate_by_condition_x_category,
    load_judgments_jsonl,
    plot_condition_bars,
    refusal_rate_on_benign,
    write_summary_csv,
)
from src_dev.persona_jailbreak_eval.config import (  # noqa: E402
    JailbreakEvalConfig,
    get_persona_grid_preset,
)
from src_dev.persona_jailbreak_eval.harmful_datasets import (  # noqa: E402
    load_alpaca_benign,
    load_strongreject,
)
from src_dev.persona_jailbreak_eval.hf_sync import (  # noqa: E402
    ensure_drift_artefacts,
    hydrate_run_dir_from_hf,
    upload_run_dir_to_hf,
)
from src_dev.persona_jailbreak_eval.personas import (  # noqa: E402
    Persona,
    load_curated_personas,
    sample_personas,
)
from src_dev.persona_jailbreak_eval.runner import (  # noqa: E402
    PromptSample,
    run_all_conditions_inference,
    run_judges_on_all_conditions,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _build_grid_samples(cfg: JailbreakEvalConfig) -> list[PromptSample]:
    """Build (persona × sysprompt × harm-question) and benign-control samples."""
    all_personas = load_curated_personas(cfg.personas_path)
    chosen_personas: list[Persona] = sample_personas(
        all_personas, n=cfg.n_personas, seed=cfg.seed,
    )
    print(f"  selected {len(chosen_personas)} personas: "
          f"{[p.id for p in chosen_personas]}")

    harm_items = load_strongreject(n=cfg.n_harm_questions, seed=cfg.seed)
    print(f"  loaded {len(harm_items)} StrongREJECT items")

    samples: list[PromptSample] = []
    # Harmful slice: full grid.
    for persona in chosen_personas:
        sysprompt_pairs = persona.sample_sysprompts(
            cfg.n_sysprompts_per_persona,
            seed=cfg.seed * 31 + sum(ord(c) for c in persona.id),
        )
        for sysprompt_idx, sysprompt in sysprompt_pairs:
            for item in harm_items:
                sample_id = _stable_id(
                    "h",
                    persona.id, str(sysprompt_idx), item.id,
                )
                samples.append(
                    PromptSample(
                        id=sample_id,
                        user_prompt=item.prompt,
                        system_prompt=sysprompt,
                        kind="harmful",
                        category=item.category,
                        behavior=item.behavior,
                        action=item.action,
                        extras={
                            "persona_id": persona.id,
                            "harm_category_persona": persona.harm_category,
                            "sysprompt_idx": sysprompt_idx,
                            "harm_item_id": item.id,
                        },
                    )
                )

    # Benign control slice: same personas × benign instructions.
    if cfg.n_benign_control > 0:
        benign_items = load_alpaca_benign(n=cfg.n_benign_control, seed=cfg.seed)
        # One benign sysprompt per persona is enough (we're not measuring
        # category-level over-refusal, just the global rate).
        for persona in chosen_personas:
            sysprompt = persona.system_prompts[0]
            for item in benign_items:
                sample_id = _stable_id("b", persona.id, item.id)
                samples.append(
                    PromptSample(
                        id=sample_id,
                        user_prompt=item.prompt,
                        system_prompt=sysprompt,
                        kind="benign",
                        category="benign",
                        behavior="",
                        action="",
                        extras={
                            "persona_id": persona.id,
                            "harm_category_persona": persona.harm_category,
                            "alpaca_item_id": item.id,
                        },
                    )
                )

    n_harm = sum(1 for s in samples if s.kind == "harmful")
    n_benign = sum(1 for s in samples if s.kind == "benign")
    print(f"  built {len(samples)} samples: {n_harm} harmful + {n_benign} benign")
    return samples


def _aggregate_and_plot(cfg: JailbreakEvalConfig, judgment_paths: dict[str, Path]) -> None:
    all_records = []
    for path in judgment_paths.values():
        all_records.extend(load_judgments_jsonl(path))
    print(f"\n  aggregating {len(all_records)} judgment records "
          f"across {len(judgment_paths)} conditions...")

    out_dir = cfg.run_dir / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    harm_rows = harmful_rate_by_condition(all_records)
    refusal_rows = refusal_rate_on_benign(all_records)
    cat_rows = harmful_rate_by_condition_x_category(all_records)

    write_summary_csv(harm_rows, out_dir / "harmful_rate_by_condition.csv")
    write_summary_csv(refusal_rows, out_dir / "refusal_rate_on_benign.csv")
    write_summary_csv(cat_rows, out_dir / "harmful_rate_by_condition_x_category.csv")

    plot_condition_bars(
        harm_rows, refusal_rows,
        title=f"Persona × StrongREJECT grid — {cfg.run_slug}",
        output_path=out_dir / "summary_bars.png",
    )

    print("\n  ── HARMFUL RATE ─────────────────────────────────────────")
    for r in harm_rows:
        print(f"    {r.condition:30s} n={r.n:4d}  "
              f"rate={r.rate:.3f}  CI=[{r.ci_low:.3f}, {r.ci_high:.3f}]")
    if refusal_rows:
        print("\n  ── OVER-REFUSAL (benign control) ────────────────────────")
        for r in refusal_rows:
            print(f"    {r.condition:30s} n={r.n:4d}  "
                  f"rate={r.rate:.3f}  CI=[{r.ci_low:.3f}, {r.ci_high:.3f}]")
    print(f"\n  artefacts: {out_dir}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["smoke", "balanced", "full"], default="smoke")
    parser.add_argument("--run-slug", help="Override run_slug")
    parser.add_argument("--axis-path", type=Path,
                        help="Path to axis.pt (required if --conditions includes activation_capping)")
    parser.add_argument("--capping-config-path", type=Path,
                        help="Path to capping_config.pt (required if --conditions includes activation_capping)")
    parser.add_argument("--conditions", help="Comma-separated condition names")
    parser.add_argument("--n-personas", type=int)
    parser.add_argument("--n-sysprompts", type=int)
    parser.add_argument("--n-harm-questions", type=int)
    parser.add_argument("--n-benign", type=int)
    parser.add_argument("--skip-aggregate", action="store_true",
                        help="Run inference + judging only; skip aggregation/plot")
    parser.add_argument("--no-upload-hf", action="store_true",
                        help="Skip HF upload of outputs (default: upload after each stage)")
    parser.add_argument("--no-hydrate-hf", action="store_true",
                        help="Skip HF hydration check at startup (default: hydrate)")
    parser.add_argument("--drift-run-slug", default=None,
                        help="Drift run-slug to hydrate axis from (default: smoke_v1)")
    args = parser.parse_args()

    cfg = get_persona_grid_preset(args.preset)
    if args.run_slug:
        cfg.run_slug = args.run_slug
    if args.axis_path is not None:
        cfg.axis_path = args.axis_path
    if args.capping_config_path is not None:
        cfg.capping_config_path = args.capping_config_path
    if args.conditions:
        cfg.conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    if args.n_personas is not None:
        cfg.n_personas = args.n_personas
    if args.n_sysprompts is not None:
        cfg.n_sysprompts_per_persona = args.n_sysprompts
    if args.n_harm_questions is not None:
        cfg.n_harm_questions = args.n_harm_questions
    if args.n_benign is not None:
        cfg.n_benign_control = args.n_benign
    if args.no_upload_hf:
        cfg.upload_hf = False
    if args.no_hydrate_hf:
        cfg.hydrate_hf = False
    if args.drift_run_slug is not None:
        cfg.drift_run_slug = args.drift_run_slug
    cfg.hf_eval_type = "persona_jailbreak_grid"

    _seed_everything(cfg.seed)

    print("=" * 70)
    print(f"  Persona × StrongREJECT grid — preset={args.preset!r}")
    print(f"  Run dir: {cfg.run_dir}")
    print(f"  Conditions: {cfg.conditions}")
    print(f"  Personas: {cfg.n_personas} × sysprompts: {cfg.n_sysprompts_per_persona}")
    print(f"  Harm questions: {cfg.n_harm_questions}  Benign control: {cfg.n_benign_control}")
    print(f"  Judge: {cfg.judge.model} (provider={cfg.judge.provider})")
    print("=" * 70)

    # Hydrate any prior outputs from HF before running. The idempotent
    # inference + judge stages will then see "already complete" and skip
    # work for whatever was hydrated.
    if cfg.hydrate_hf:
        hydrate_run_dir_from_hf(
            local_run_dir=cfg.run_dir,
            eval_type=cfg.hf_eval_type,
            model_slug=cfg.model_slug,
            run_slug=cfg.run_slug,
            repo_id=cfg.hf_repo_id,
        )

    # Resolve axis + capping_config (hydrate from drift HF subpath if needed).
    if "activation_capping" in cfg.conditions:
        axis_path, capping_config_path = ensure_drift_artefacts(
            model_slug=cfg.model_slug,
            drift_run_slug=cfg.drift_run_slug,
            target_dir=cfg.run_dir / "_drift_artefacts",
            variant=cfg.drift_axis_variant,
            repo_id=cfg.hf_repo_id,
            explicit_axis_path=cfg.axis_path,
            explicit_capping_config_path=cfg.capping_config_path,
        )
        if axis_path is None or capping_config_path is None:
            raise SystemExit(
                "activation_capping is in --conditions but axis.pt or capping_config.pt "
                "could not be resolved. Either pass --axis-path / --capping-config-path "
                "explicitly, or run the drift pipeline (build_axis + pick_capping) for the "
                f"requested drift_run_slug={cfg.drift_run_slug!r}."
            )
        cfg.axis_path = axis_path
        cfg.capping_config_path = capping_config_path
        print(f"  axis: {cfg.axis_path}")
        print(f"  capping_config: {cfg.capping_config_path}")

    samples = _build_grid_samples(cfg)
    response_paths = run_all_conditions_inference(
        cfg, samples, output_dir=cfg.run_dir / "responses",
    )
    if cfg.upload_hf:
        upload_run_dir_to_hf(
            local_run_dir=cfg.run_dir, eval_type=cfg.hf_eval_type,
            model_slug=cfg.model_slug, run_slug=cfg.run_slug,
            repo_id=cfg.hf_repo_id, stage="inference",
        )

    judgment_paths = run_judges_on_all_conditions(
        cfg, response_paths, output_dir=cfg.run_dir / "judgments",
    )
    if cfg.upload_hf:
        upload_run_dir_to_hf(
            local_run_dir=cfg.run_dir, eval_type=cfg.hf_eval_type,
            model_slug=cfg.model_slug, run_slug=cfg.run_slug,
            repo_id=cfg.hf_repo_id, stage="judgments",
        )

    if not args.skip_aggregate:
        _aggregate_and_plot(cfg, judgment_paths)
        if cfg.upload_hf:
            upload_run_dir_to_hf(
                local_run_dir=cfg.run_dir, eval_type=cfg.hf_eval_type,
                model_slug=cfg.model_slug, run_slug=cfg.run_slug,
                repo_id=cfg.hf_repo_id, stage="aggregate",
            )


if __name__ == "__main__":
    main()
