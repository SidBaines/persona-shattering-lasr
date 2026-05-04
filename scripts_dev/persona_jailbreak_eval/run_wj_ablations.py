#!/usr/bin/env python3
"""WildJailbreak LoRA ablations for all llama-3.1-8b-it catalogue adapters.

Runs the same WildJailbreak adv-harmful + adv-benign protocol as
``run_wildjailbreak.py`` but over every llama-3.1-8b-it OCEAN adapter in
``OCEAN_REGISTRY`` plus two control adapters:

    control latest  — ocean_def_control / vanton4_paired_dpo_s1vs2
    control legacy  — ocean_def_control / vanton4_seed1

We intentionally do NOT rerun vanilla, activation-capping, or the
balanced c+(0.5)⊕o-(0.5) soup here; those already exist under
``wj_balanced`` and can be compared cross-run.

Defaults: N=400 adv-harmful + 100 adv-benign per condition.

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

from src_dev.common.lora_catalogue import LoraHFCatalogue, OCEAN_REGISTRY  # noqa: E402
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


def _catalogue_condition_name(adapter_key: str) -> str:
    if adapter_key == "control_latest":
        return "lora_soup_control_latest_1.0"
    if adapter_key == "control_legacy":
        return "lora_soup_control_legacy_1.0"
    return f"lora_soup_{adapter_key}_1.0"


def _build_ablation_combos() -> tuple[LoraComboCondition, ...]:
    combos: list[LoraComboCondition] = []
    for slug in OCEAN_REGISTRY:
        combos.append(
            LoraComboCondition(
                name=_catalogue_condition_name(slug),
                adapters=[(slug, 1.0)],
            )
        )

    legacy = LoraHFCatalogue()
    combos.append(
        LoraComboCondition(
            name=_catalogue_condition_name("control_latest"),
            adapters=[(legacy.control_latest, 1.0)],
        )
    )
    combos.append(
        LoraComboCondition(
            name=_catalogue_condition_name("control_legacy"),
            adapters=[(legacy.control_legacy, 1.0)],
        )
    )
    return tuple(combos)


ABLATION_COMBOS: tuple[LoraComboCondition, ...] = _build_ablation_combos()
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
    explicit_refusal_rows = explicit_refusal_rate_on_benign(all_records)
    write_summary_csv(harm_rows, out_dir / "harmful_rate_by_condition.csv")
    write_summary_csv(refusal_rows, out_dir / "refusal_rate_on_benign.csv")
    write_summary_csv(explicit_refusal_rows, out_dir / "explicit_refusal_rate_on_benign.csv")
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
    parser.add_argument("--run-slug", default="wj_ablations_v1",
                        help="HF/scratch path leaf (default: wj_ablations_v1)")
    parser.add_argument("--n-harmful", type=int, default=400)
    parser.add_argument("--n-benign", type=int, default=100)
    parser.add_argument("--no-upload-hf", action="store_true")
    parser.add_argument("--no-hydrate-hf", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()

    cfg = get_wildjailbreak_preset("balanced")
    cfg.vllm_batch_size = 64
    cfg.vllm_gpu_memory_utilization = 0.80
    cfg.run_slug = args.run_slug
    cfg.hf_eval_type = "persona_jailbreak_wildjailbreak"
    cfg.conditions = ABLATION_CONDITION_NAMES
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
    print(f"  vLLM batch size: {cfg.vllm_batch_size}")
    print(f"  vLLM gpu_memory_utilization: {cfg.vllm_gpu_memory_utilization}")
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
