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

Conditions (default): vanilla, activation_capping, lora_soup_c_plus_o_minus.
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

    _seed_everything(cfg.seed)

    print("=" * 70)
    print(f"  Persona × StrongREJECT grid — preset={args.preset!r}")
    print(f"  Run dir: {cfg.run_dir}")
    print(f"  Conditions: {cfg.conditions}")
    print(f"  Personas: {cfg.n_personas} × sysprompts: {cfg.n_sysprompts_per_persona}")
    print(f"  Harm questions: {cfg.n_harm_questions}  Benign control: {cfg.n_benign_control}")
    print(f"  Judge: {cfg.judge.model} (provider={cfg.judge.provider})")
    print("=" * 70)

    samples = _build_grid_samples(cfg)
    response_paths = run_all_conditions_inference(
        cfg, samples, output_dir=cfg.run_dir / "responses",
    )
    judgment_paths = run_judges_on_all_conditions(
        cfg, response_paths, output_dir=cfg.run_dir / "judgments",
    )

    if not args.skip_aggregate:
        _aggregate_and_plot(cfg, judgment_paths)


if __name__ == "__main__":
    main()
