#!/usr/bin/env python3
"""WildJailbreak runner for one custom LoRA combo at ablation-scale N.

Default combo:
    1.0 * a_plus + 1.0 * c_plus

Usage::

    uv run python -m scripts_dev.persona_jailbreak_eval.run_wj_custom_combo \\
        --run-slug wj_combo_a_plus_c_plus_v1
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
    explicit_refusal_rate_on_benign,
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


def _scale_tag(scale: float) -> str:
    return f"{scale:.1f}"


def _build_combo(a_plus_scale: float, c_plus_scale: float) -> LoraComboCondition:
    return LoraComboCondition(
        name=(
            f"lora_soup_a_plus_{_scale_tag(a_plus_scale)}"
            f"_c_plus_{_scale_tag(c_plus_scale)}"
        ),
        adapters=[("a_plus", a_plus_scale), ("c_plus", c_plus_scale)],
    )


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
    explicit_refusal_rows = explicit_refusal_rate_on_benign(all_records)
    write_summary_csv(harm_rows, out_dir / "harmful_rate_by_condition.csv")
    write_summary_csv(refusal_rows, out_dir / "refusal_rate_on_benign.csv")
    write_summary_csv(explicit_refusal_rows, out_dir / "explicit_refusal_rate_on_benign.csv")
    plot_condition_bars(
        harm_rows, refusal_rows,
        title=f"WJ custom combo — {cfg.run_slug}",
        output_path=out_dir / "summary_bars.png",
    )
    print("\n  ── HARMFUL RATE ─────────────────────────────────────────")
    for r in harm_rows:
        print(f"    {r.condition:30s} n={r.n:4d}  rate={r.rate:.3f}  "
              f"CI=[{r.ci_low:.3f}, {r.ci_high:.3f}]")
    if refusal_rows:
        print("\n  ── BENIGN NONCOMPLIANCE (adversarial benign) ────────────")
        for r in refusal_rows:
            print(f"    {r.condition:30s} n={r.n:4d}  rate={r.rate:.3f}  "
                  f"CI=[{r.ci_low:.3f}, {r.ci_high:.3f}]")
    if explicit_refusal_rows:
        print("\n  ── EXPLICIT REFUSAL (adversarial benign) ────────────────")
        for r in explicit_refusal_rows:
            print(f"    {r.condition:30s} n={r.n:4d}  rate={r.rate:.3f}  "
                  f"CI=[{r.ci_low:.3f}, {r.ci_high:.3f}]")
    print(f"\n  artefacts: {out_dir}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-slug", default="wj_combo_a_plus_c_plus_v1")
    parser.add_argument("--a-plus-scale", type=float, default=1.0)
    parser.add_argument("--c-plus-scale", type=float, default=1.0)
    parser.add_argument("--n-harmful", type=int, default=400)
    parser.add_argument("--n-benign", type=int, default=100)
    parser.add_argument("--no-upload-hf", action="store_true")
    parser.add_argument("--no-hydrate-hf", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()

    combo = _build_combo(args.a_plus_scale, args.c_plus_scale)

    cfg = get_wildjailbreak_preset("balanced")
    cfg.vllm_batch_size = 64
    cfg.vllm_gpu_memory_utilization = 0.80
    cfg.run_slug = args.run_slug
    cfg.hf_eval_type = "persona_jailbreak_wildjailbreak"
    cfg.conditions = (combo.name,)
    cfg.lora_combos = (combo,)
    cfg.n_wildjailbreak_harmful = args.n_harmful
    cfg.n_wildjailbreak_benign = args.n_benign
    if args.no_upload_hf:
        cfg.upload_hf = False
    if args.no_hydrate_hf:
        cfg.hydrate_hf = False

    _seed_everything(cfg.seed)

    print("=" * 70)
    print(f"  WildJailbreak custom LoRA combo — run_slug={cfg.run_slug!r}")
    print(f"  Run dir: {cfg.run_dir}")
    print(f"  Conditions: {cfg.conditions}")
    print(f"  combo scales: a_plus={args.a_plus_scale}  c_plus={args.c_plus_scale}")
    print(f"  adv-harmful: {cfg.n_wildjailbreak_harmful}  adv-benign: {cfg.n_wildjailbreak_benign}")
    print(f"  vLLM batch size: {cfg.vllm_batch_size}")
    print(f"  vLLM gpu_memory_utilization: {cfg.vllm_gpu_memory_utilization}")
    print(f"  Judge: {cfg.judge.model} (provider={cfg.judge.provider})")
    print("=" * 70)

    if cfg.hydrate_hf:
        hydrate_run_dir_from_hf(
            local_run_dir=cfg.run_dir,
            eval_type=cfg.hf_eval_type,
            model_slug=cfg.model_slug,
            run_slug=cfg.run_slug,
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
