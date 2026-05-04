#!/usr/bin/env python3
"""WildJailbreak LoRA-ablation driver — single-LoRA conditions at scale 1.0.

Ablations to test which OCEAN trait adapter is responsible for the
LoRA-soup's increased jailbreak vulnerability. Uses the same WildJailbreak
adv-harmful + adv-benign protocol as ``run_wildjailbreak.py`` but with a
fixed set of single-LoRA conditions (no soup, no capping):

    vanilla
    lora_c_plus_1.0    — conscientiousness amplifier alone
    lora_o_minus_1.0   — openness suppressor alone
    lora_o_plus_1.0    — openness amplifier alone
    lora_c_minus_1.0   — conscientiousness suppressor alone
    lora_a_minus_1.0   — agreeableness suppressor alone

Defaults to N=400 harmful + 100 benign per condition (~3.5pp Wilson CI
half-width at p≈0.5). Six conditions × 500 samples ≈ 3k generations +
judge calls; ~25-30 min, ~$3-5.

Capping is intentionally excluded — we already know its WJ effect is
modest (~10pp) and it requires the drift axis + capping_config. If you
want to compare ablations against capping, look at the wj_balanced run
on HF.

Usage::

    uv run python -m scripts_dev.persona_jailbreak_eval.run_wj_ablations \\
        --run-slug wj_ablations_v1
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
    LoraComboCondition,
    get_wildjailbreak_preset,
)
from src_dev.persona_jailbreak_eval.harmful_datasets import load_wildjailbreak  # noqa: E402
from src_dev.persona_jailbreak_eval.hf_sync import (  # noqa: E402
    hydrate_run_dir_from_hf,
    upload_run_dir_to_hf,
)
from src_dev.persona_jailbreak_eval.runner import (  # noqa: E402
    PromptSample,
    run_all_conditions_inference,
    run_judges_on_all_conditions,
)


# Single-LoRA ablation set: each condition uses exactly one OCEAN adapter
# at scale 1.0. Names start with "lora_soup" so the engine-routing helpers
# in src_dev.activation_capping.conditions recognise them as vLLM conditions.
ABLATION_COMBOS: tuple[LoraComboCondition, ...] = (
    LoraComboCondition(name="lora_soup_c_plus_1.0",  adapters=[("c_plus",  1.0)]),
    LoraComboCondition(name="lora_soup_o_minus_1.0", adapters=[("o_minus", 1.0)]),
    LoraComboCondition(name="lora_soup_o_plus_1.0",  adapters=[("o_plus",  1.0)]),
    LoraComboCondition(name="lora_soup_c_minus_1.0", adapters=[("c_minus", 1.0)]),
    LoraComboCondition(name="lora_soup_a_minus_1.0", adapters=[("a_minus", 1.0)]),
)
ABLATION_CONDITION_NAMES: tuple[str, ...] = tuple(c.name for c in ABLATION_COMBOS)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_wj_samples(cfg: JailbreakEvalConfig) -> list[PromptSample]:
    harmful = load_wildjailbreak(
        split="adversarial_harmful", n=cfg.n_wildjailbreak_harmful, seed=cfg.seed,
    )
    benign = load_wildjailbreak(
        split="adversarial_benign", n=cfg.n_wildjailbreak_benign, seed=cfg.seed,
    )
    print(f"  loaded {len(harmful)} adv-harmful + {len(benign)} adv-benign")
    samples: list[PromptSample] = []
    for item in harmful:
        samples.append(PromptSample(
            id=item.id, user_prompt=item.prompt, system_prompt=None,
            kind="harmful", category=item.category,
            behavior=item.behavior, action=item.action,
            extras={"wj_split": "adversarial_harmful"},
        ))
    for item in benign:
        samples.append(PromptSample(
            id=item.id, user_prompt=item.prompt, system_prompt=None,
            kind="benign", category=item.category,
            behavior=item.behavior, action=item.action,
            extras={"wj_split": "adversarial_benign"},
        ))
    return samples


def _aggregate_and_plot(cfg: JailbreakEvalConfig, judgment_paths: dict[str, Path]) -> None:
    all_records = []
    for path in judgment_paths.values():
        all_records.extend(load_judgments_jsonl(path))
    print(f"\n  aggregating {len(all_records)} judgment records...")
    out_dir = cfg.run_dir / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    harm_rows = harmful_rate_by_condition(all_records)
    refusal_rows = refusal_rate_on_benign(all_records)
    write_summary_csv(harm_rows, out_dir / "harmful_rate_by_condition.csv")
    write_summary_csv(refusal_rows, out_dir / "refusal_rate_on_benign.csv")
    plot_condition_bars(
        harm_rows, refusal_rows,
        title=f"WJ LoRA ablations — {cfg.run_slug}",
        output_path=out_dir / "summary_bars.png",
    )
    print("\n  ── HARMFUL RATE ─────────────────────────────────────────")
    for r in harm_rows:
        print(f"    {r.condition:30s} n={r.n:4d}  rate={r.rate:.3f}  "
              f"CI=[{r.ci_low:.3f}, {r.ci_high:.3f}]")
    if refusal_rows:
        print("\n  ── OVER-REFUSAL (adversarial benign) ────────────────────")
        for r in refusal_rows:
            print(f"    {r.condition:30s} n={r.n:4d}  rate={r.rate:.3f}  "
                  f"CI=[{r.ci_low:.3f}, {r.ci_high:.3f}]")
    print(f"\n  artefacts: {out_dir}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-slug", default="wj_ablations_v1",
                        help="HF/scratch path leaf (default: wj_ablations_v1)")
    parser.add_argument("--n-harmful", type=int, default=400)
    parser.add_argument("--n-benign", type=int, default=100)
    parser.add_argument("--no-upload-hf", action="store_true")
    parser.add_argument("--no-hydrate-hf", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()

    # Build config: start from the WJ balanced preset (so generation knobs
    # match the existing balanced run), then override conditions, combos,
    # run_slug, and N.
    cfg = get_wildjailbreak_preset("balanced")
    cfg.run_slug = args.run_slug
    cfg.hf_eval_type = "persona_jailbreak_wildjailbreak"
    cfg.conditions = ("vanilla", *ABLATION_CONDITION_NAMES)
    cfg.lora_combos = ABLATION_COMBOS
    cfg.n_wildjailbreak_harmful = args.n_harmful
    cfg.n_wildjailbreak_benign = args.n_benign
    if args.no_upload_hf:
        cfg.upload_hf = False
    if args.no_hydrate_hf:
        cfg.hydrate_hf = False

    _seed_everything(cfg.seed)

    print("=" * 70)
    print(f"  WildJailbreak LoRA ablations — run_slug={cfg.run_slug!r}")
    print(f"  Run dir: {cfg.run_dir}")
    print(f"  Conditions ({len(cfg.conditions)}): {cfg.conditions}")
    print(f"  adv-harmful: {cfg.n_wildjailbreak_harmful}  adv-benign: {cfg.n_wildjailbreak_benign}")
    print(f"  Judge: {cfg.judge.model} (provider={cfg.judge.provider})")
    print("=" * 70)

    if cfg.hydrate_hf:
        hydrate_run_dir_from_hf(
            local_run_dir=cfg.run_dir, eval_type=cfg.hf_eval_type,
            model_slug=cfg.model_slug, run_slug=cfg.run_slug,
            repo_id=cfg.hf_repo_id,
        )

    samples = _build_wj_samples(cfg)
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
