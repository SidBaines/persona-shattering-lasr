#!/usr/bin/env python3
"""Option 2 — WildJailbreak single-turn eval (high statistical power, with
built-in over-refusal split).

Sends the WildJailbreak ``adversarial_harmful`` and ``adversarial_benign``
prompts directly to the model with no system prompt — the jailbreak tactic
is baked into the user prompt itself. Harmful rows are scored with the
verbatim Appendix D.2.2 rubric; benign rows are scored with a binary
refusal judge.

This is a more permissive complement to Option 1: it doesn't have the
paper-faithful (persona × harm-question) grid structure, but the eval set
is well-curated and large (2000 + 210), giving tight CIs at lower compute
than Option 1 at full scale.

Usage::

    uv run python -m scripts_dev.persona_jailbreak_eval.run_wildjailbreak \\
        --preset smoke \\
        --axis-path scratch/persona_drift_assistant_axis/.../axis.pt \\
        --capping-config-path scratch/persona_drift_assistant_axis/.../capping_config.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from src_dev.activation_capping.conditions import ensure_vllm_fork_safe  # noqa: E402
ensure_vllm_fork_safe()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src_dev.persona_jailbreak_eval.aggregate import (  # noqa: E402
    harmful_rate_by_condition,
    load_judgments_jsonl,
    plot_condition_bars,
    refusal_rate_on_benign,
    write_summary_csv,
)
from src_dev.persona_jailbreak_eval.config import (  # noqa: E402
    JailbreakEvalConfig,
    get_wildjailbreak_preset,
)
from src_dev.persona_jailbreak_eval.harmful_datasets import load_wildjailbreak  # noqa: E402
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


def _build_wj_samples(cfg: JailbreakEvalConfig) -> list[PromptSample]:
    """Load WildJailbreak adversarial harmful + benign and convert to PromptSample."""
    harmful = load_wildjailbreak(
        split="adversarial_harmful", n=cfg.n_wildjailbreak_harmful, seed=cfg.seed,
    )
    benign = load_wildjailbreak(
        split="adversarial_benign", n=cfg.n_wildjailbreak_benign, seed=cfg.seed,
    )
    print(f"  loaded {len(harmful)} adv-harmful + {len(benign)} adv-benign")

    samples: list[PromptSample] = []
    for item in harmful:
        samples.append(
            PromptSample(
                id=item.id,
                user_prompt=item.prompt,
                system_prompt=None,
                kind="harmful",
                category=item.category,
                behavior=item.behavior,
                action=item.action,
                extras={"wj_split": "adversarial_harmful"},
            )
        )
    for item in benign:
        samples.append(
            PromptSample(
                id=item.id,
                user_prompt=item.prompt,
                system_prompt=None,
                kind="benign",
                category=item.category,
                behavior=item.behavior,
                action=item.action,
                extras={"wj_split": "adversarial_benign"},
            )
        )
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

    write_summary_csv(harm_rows, out_dir / "harmful_rate_by_condition.csv")
    write_summary_csv(refusal_rows, out_dir / "refusal_rate_on_benign.csv")

    plot_condition_bars(
        harm_rows, refusal_rows,
        title=f"WildJailbreak — {cfg.run_slug}",
        output_path=out_dir / "summary_bars.png",
    )

    print("\n  ── HARMFUL RATE ─────────────────────────────────────────")
    for r in harm_rows:
        print(f"    {r.condition:30s} n={r.n:4d}  "
              f"rate={r.rate:.3f}  CI=[{r.ci_low:.3f}, {r.ci_high:.3f}]")
    if refusal_rows:
        print("\n  ── OVER-REFUSAL (adversarial benign) ────────────────────")
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
    parser.add_argument("--n-harmful", type=int)
    parser.add_argument("--n-benign", type=int)
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()

    cfg = get_wildjailbreak_preset(args.preset)
    if args.run_slug:
        cfg.run_slug = args.run_slug
    if args.axis_path is not None:
        cfg.axis_path = args.axis_path
    if args.capping_config_path is not None:
        cfg.capping_config_path = args.capping_config_path
    if args.conditions:
        cfg.conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    if args.n_harmful is not None:
        cfg.n_wildjailbreak_harmful = args.n_harmful
    if args.n_benign is not None:
        cfg.n_wildjailbreak_benign = args.n_benign

    _seed_everything(cfg.seed)

    print("=" * 70)
    print(f"  WildJailbreak eval — preset={args.preset!r}")
    print(f"  Run dir: {cfg.run_dir}")
    print(f"  Conditions: {cfg.conditions}")
    print(f"  adv-harmful: {cfg.n_wildjailbreak_harmful}  "
          f"adv-benign: {cfg.n_wildjailbreak_benign}")
    print(f"  Judge: {cfg.judge.model} (provider={cfg.judge.provider})")
    print("=" * 70)

    samples = _build_wj_samples(cfg)
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
