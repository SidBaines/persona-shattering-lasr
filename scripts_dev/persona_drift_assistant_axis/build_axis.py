#!/usr/bin/env python3
"""Phase 1 — Build the Assistant Axis on Llama 3.1 8B.

Wraps the upstream 5-step pipeline from a pinned runtime checkout of
``safety-research/assistant-axis`` and applies our knobs:
roles whitelist, question count, sysprompt-per-role cap, judge model
override, and per-role min-count.

Outputs (under ``scratch/persona_drift_assistant_axis/{model}/{run_slug}/``):

    staged/roles_instructions/   # truncated role files (if knob set)
    staged/extraction_questions.jsonl  # truncated questions
    responses/{role}.jsonl       # Step 1 — vLLM generations
    activations/{role}.pt        # Step 2 — mean response activations per layer
    scores/{role}.json           # Step 3 — judge scores 0/1/2/3
    vectors/{role}.pt            # Step 4 — score=3 mean vector per role
    axis.pt                      # Step 5 — final axis (n_layers, hidden_dim)
    run_info.json                # provenance: knobs, SHA, timing, costs
    logs/{step}.log              # per-step stdout/stderr

Resumability — each step is no-op when its outputs exist (upstream pipeline
behavior). Re-running with the same ``run_slug`` reuses the cache.

Usage::

    # Smoke test — ~10–15 min, ~$1–2
    uv run python -m scripts_dev.persona_drift_assistant_axis.build_axis \\
        --preset smoke

    # Full replication
    uv run python -m scripts_dev.persona_drift_assistant_axis.build_axis \\
        --preset full --upload-hf
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

# Project imports — repo root on sys.path lets us reuse our config + utils.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from scripts_dev.persona_drift_assistant_axis.config import (  # noqa: E402
    HF_REPO,
    LORA_SOUP_VARIANT_NAME,
    AxisBuildConfig,
    ExperimentConfig,
    get_preset,
)
from src_dev.activation_capping.assistant_axis_dependency import (  # noqa: E402
    assistant_axis_source_label,
    ensure_assistant_axis_repo,
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hf_upload_axis_dir(
    out_root: Path, *, hf_subpath: str, variant: str, checkpoint_label: str,
) -> None:
    """Upload the current state of the axis output dir to HF as a checkpoint.

    Used both as an intra-Phase-1 checkpoint after the expensive judge step
    and as the final upload at end-of-Phase-1. Excludes bulky transient
    artefacts (staging dir, per-step subprocess logs, the multi-GB
    merged-model dir for LoRA variants — that's deterministically rebuilt
    from the LoRA adapters by ``_resolve_pipeline_model`` on rehydration).
    """
    from huggingface_hub import HfApi
    api = HfApi()
    upload_subpath = f"{hf_subpath}/axes/{variant}"
    api.upload_folder(
        folder_path=str(out_root),
        repo_id=HF_REPO,
        repo_type="dataset",
        path_in_repo=upload_subpath,
        commit_message=(
            f"persona_drift_assistant_axis: variant={variant} {checkpoint_label}"
        ),
        ignore_patterns=["staged/**", "logs/**", "merged_model/**"],
    )
    print(f"  HF checkpoint ({checkpoint_label}) → "
          f"https://huggingface.co/datasets/{HF_REPO}/tree/main/{upload_subpath}")

# ── Staging: apply knobs to upstream data inputs ──────────────────────────


def stage_roles_dir(
    src_roles_dir: Path,
    dst_dir: Path,
    *,
    num_roles: int | None,
    num_sysprompts_per_role: int | None,
) -> Path:
    """Copy role JSONs to ``dst_dir`` with knobs applied.

    Always includes ``default.json`` (required for axis computation). If
    ``num_roles`` is set, picks the first N (alphabetical, deterministic).
    If ``num_sysprompts_per_role`` is set, truncates each role's
    ``instruction`` list to N entries.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    role_files = sorted(src_roles_dir.glob("*.json"))

    # Default is special — must be present, doesn't count against num_roles.
    default_file = src_roles_dir / "default.json"
    other_role_files = [f for f in role_files if f.name != "default.json"]
    if num_roles is not None:
        other_role_files = other_role_files[:num_roles]
    selected = [default_file, *other_role_files]

    for f in selected:
        with open(f) as fh:
            data = json.load(fh)
        if num_sysprompts_per_role is not None:
            data["instruction"] = data["instruction"][:num_sysprompts_per_role]
        with open(dst_dir / f.name, "w") as fh:
            json.dump(data, fh, indent=2)
    return dst_dir


def stage_questions(src_file: Path, dst_file: Path, *, num_questions: int) -> Path:
    """Copy the questions JSONL with at most ``num_questions`` rows."""
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    with open(src_file) as src, open(dst_file, "w") as dst:
        for i, line in enumerate(src):
            if i >= num_questions:
                break
            dst.write(line)
    return dst_file


# ── Subprocess helpers ────────────────────────────────────────────────────


def _run_step(
    label: str,
    cmd: list[str],
    *,
    log_file: Path,
    cwd: Path,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Run a pipeline step with stdout/stderr teed to ``log_file``."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print(f"\n=== {label} ===", flush=True)
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    print(f"  log: {log_file}", flush=True)
    start = time.time()
    with open(log_file, "w") as logf:
        proc = subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(cwd),
            check=False,
        )
    elapsed = time.time() - start
    print(f"  exit={proc.returncode} elapsed={elapsed:.1f}s", flush=True)
    if proc.returncode != 0:
        # Tail for quick diagnosis without making the user dig in logs.
        with open(log_file) as fh:
            tail = fh.readlines()[-30:]
        sys.stderr.write("".join(tail))
        raise SystemExit(f"{label} failed (see {log_file})")


# ── Pipeline orchestrator ─────────────────────────────────────────────────


def _resolve_pipeline_model(cfg: ExperimentConfig, variant: str) -> str:
    """Return the model name/path that upstream's pipeline should run on.

    For ``base``: just the configured HF model id. For LoRA variants: pre-merge
    the adapters into a standalone HF model dir (idempotent — skips if dir
    already populated) and return that path.
    """
    if variant == "base":
        return cfg.axis.base_model

    if variant == LORA_SOUP_VARIANT_NAME:
        merged = cfg.merged_model_dir(variant)
        if (merged / "config.json").exists():
            print(f"  Using cached merged-LoRA model dir: {merged}")
            return str(merged)

        from src_dev.common.lora_catalogue import OCEAN_REGISTRY
        from src_dev.evals.model_resolution import resolve_model_reference
        from src_dev.utils.lora_composition import (
            WeightedAdapter,
            merge_weighted_adapters,
        )

        adapters = [
            WeightedAdapter(path=OCEAN_REGISTRY[slug].adapter_ref, scale=scale)
            for slug, scale in cfg.lora_soup.adapters
        ]
        print(f"  Merging LoRA soup → {merged}: "
              f"{[(a.path, a.scale) for a in adapters]}")
        merged.mkdir(parents=True, exist_ok=True)
        # The adapter_resolver MUST be ``resolve_model_reference``, NOT
        # ``resolve_adapter_to_local_dir`` itself — passing the latter
        # causes a recursive call that loses the ``::subfolder`` info and
        # makes ``snapshot_download`` pull the entire monorepo (~30 GB)
        # instead of just the adapter's subfolder. Every other caller in
        # the repo uses this same lambda; see e.g. src_dev/evals/lora_merge.py.
        merge_weighted_adapters(
            base_model=cfg.axis.base_model,
            adapters=adapters,
            output_dir=merged,
            adapter_resolver=lambda ref: resolve_model_reference(ref, kind="adapter"),
        )
        return str(merged)

    raise ValueError(f"Unknown variant {variant!r}")


def build_axis(
    cfg: ExperimentConfig,
    *,
    variant: str = "base",
    upload_hf: bool,
) -> dict:
    """Run all 5 upstream pipeline steps for ``variant`` under our config.

    Outputs land in ``cfg.axis_dir(variant)``. ``base`` runs upstream's
    pipeline on the unadapted model; LoRA variants pre-merge the soup into
    a standalone HF model dir before invoking the pipeline on that dir.
    """
    axis_cfg: AxisBuildConfig = cfg.axis
    assistant_axis_dir = ensure_assistant_axis_repo()
    vendor_pipeline = assistant_axis_dir / "pipeline"
    vendor_data = assistant_axis_dir / "data"
    out_root = cfg.axis_dir(variant)
    out_root.mkdir(parents=True, exist_ok=True)

    pipeline_model = _resolve_pipeline_model(cfg, variant)
    print(f"  Pipeline model: {pipeline_model}")

    staged_roles = out_root / "staged" / "roles_instructions"
    staged_questions = out_root / "staged" / "extraction_questions.jsonl"
    responses_dir = out_root / "responses"
    activations_dir = out_root / "activations"
    scores_dir = out_root / "scores"
    vectors_dir = out_root / "vectors"
    axis_path = out_root / "axis.pt"
    logs_dir = out_root / "logs"

    # Knob-aware staging (idempotent — overwrites are cheap).
    stage_roles_dir(
        vendor_data / "roles" / "instructions",
        staged_roles,
        num_roles=axis_cfg.num_roles,
        num_sysprompts_per_role=axis_cfg.num_sysprompts_per_role,
    )
    stage_questions(
        vendor_data / "extraction_questions.jsonl",
        staged_questions,
        num_questions=axis_cfg.num_questions,
    )
    n_role_files = len(list(staged_roles.glob("*.json")))
    print(f"Staged {n_role_files} role files (incl. default), "
          f"{axis_cfg.num_questions} questions, "
          f"{axis_cfg.num_sysprompts_per_role or 'all'} sysprompts/role")

    timings: dict[str, float] = {}

    # Step 1 — generate.
    if not responses_dir.exists() or not any(responses_dir.glob("*.jsonl")):
        t0 = time.time()
        cmd = [
            sys.executable, str(vendor_pipeline / "1_generate.py"),
            "--model", pipeline_model,
            "--roles_dir", str(staged_roles),
            "--questions_file", str(staged_questions),
            "--output_dir", str(responses_dir),
            "--question_count", str(axis_cfg.num_questions),
            "--max_model_len", str(axis_cfg.vllm_max_model_len),
            "--max_tokens", str(axis_cfg.max_new_tokens),
            "--gpu_memory_utilization", str(axis_cfg.vllm_gpu_memory_utilization),
            # Sampling params — explicit so they appear in the log file and
            # in run_info.json (don't rely on upstream's defaults silently).
            "--temperature", str(axis_cfg.temperature),
            "--top_p", str(axis_cfg.top_p),
        ]
        if axis_cfg.tensor_parallel_size is not None:
            cmd += ["--tensor_parallel_size", str(axis_cfg.tensor_parallel_size)]
        _run_step(
            "Step 1 — generate",
            cmd,
            log_file=logs_dir / "1_generate.log",
            cwd=vendor_pipeline,
        )
        timings["step1_generate"] = time.time() - t0
    else:
        print("Step 1 — generate: cached")

    # Step 2 — activations.
    if not activations_dir.exists() or not any(activations_dir.glob("*.pt")):
        t0 = time.time()
        cmd = [
            sys.executable, str(vendor_pipeline / "2_activations.py"),
            "--model", pipeline_model,
            "--responses_dir", str(responses_dir),
            "--output_dir", str(activations_dir),
            "--batch_size", str(axis_cfg.activation_batch_size),
            "--max_length", str(axis_cfg.vllm_max_model_len),
        ]
        if axis_cfg.tensor_parallel_size is not None:
            cmd += ["--tensor_parallel_size", str(axis_cfg.tensor_parallel_size)]
        _run_step(
            "Step 2 — activations",
            cmd,
            log_file=logs_dir / "2_activations.log",
            cwd=vendor_pipeline,
        )
        timings["step2_activations"] = time.time() - t0
    else:
        print("Step 2 — activations: cached")

    # Step 3 — judge. Point upstream OpenAI client at OpenRouter for qwen3-235b.
    if not scores_dir.exists() or not any(scores_dir.glob("*.json")):
        t0 = time.time()
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise SystemExit("OPENROUTER_API_KEY required for judge step")
        cmd = [
            sys.executable, str(vendor_pipeline / "3_judge.py"),
            "--responses_dir", str(responses_dir),
            "--roles_dir", str(staged_roles),
            "--output_dir", str(scores_dir),
            "--judge_model", axis_cfg.judge_model,
            "--batch_size", str(axis_cfg.judge_concurrency),
        ]
        _run_step(
            "Step 3 — judge",
            cmd,
            log_file=logs_dir / "3_judge.log",
            cwd=vendor_pipeline,
            extra_env={
                "OPENAI_API_KEY": openrouter_key,
                "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
            },
        )
        timings["step3_judge"] = time.time() - t0
        # Mid-pipeline HF checkpoint: judge calls cost real money. If the
        # machine dies between here and the end of step 5, we want to be
        # able to rehydrate without re-paying the judge.
        if upload_hf:
            try:
                _hf_upload_axis_dir(
                    out_root, hf_subpath=cfg.hf_subpath,
                    variant=variant, checkpoint_label="post-step3-judge",
                )
            except Exception as exc:  # noqa: BLE001
                # Don't fail the run if the upload trips a transient HF
                # 5xx — the final upload at end-of-pipeline is the
                # primary safety net.
                print(f"  WARN: post-step3 HF checkpoint failed ({exc!r}); "
                      "continuing — end-of-run upload will retry.")
    else:
        print("Step 3 — judge: cached")

    # Step 4 — vectors.
    if not vectors_dir.exists() or not any(vectors_dir.glob("*.pt")):
        t0 = time.time()
        cmd = [
            sys.executable, str(vendor_pipeline / "4_vectors.py"),
            "--activations_dir", str(activations_dir),
            "--scores_dir", str(scores_dir),
            "--output_dir", str(vectors_dir),
            "--min_count", str(axis_cfg.min_count_per_role),
        ]
        _run_step(
            "Step 4 — vectors",
            cmd,
            log_file=logs_dir / "4_vectors.log",
            cwd=vendor_pipeline,
        )
        timings["step4_vectors"] = time.time() - t0
    else:
        print("Step 4 — vectors: cached")

    # Step 5 — axis.
    if not axis_path.exists():
        t0 = time.time()
        cmd = [
            sys.executable, str(vendor_pipeline / "5_axis.py"),
            "--vectors_dir", str(vectors_dir),
            "--output", str(axis_path),
        ]
        _run_step(
            "Step 5 — axis",
            cmd,
            log_file=logs_dir / "5_axis.log",
            cwd=vendor_pipeline,
        )
        timings["step5_axis"] = time.time() - t0
    else:
        print("Step 5 — axis: cached")

    # Provenance.
    axis_tensor = torch.load(axis_path, map_location="cpu", weights_only=False)
    axis_norms = axis_tensor.norm(dim=1)
    # Per-vector count: how many role vectors actually contributed to step 5
    # (i.e., score=3 cleared the min_count_per_role bar). Useful smoke
    # diagnostic — see HANDOVER.md §5.
    n_role_vectors_built = len(list(vectors_dir.glob("*.pt"))) if vectors_dir.exists() else 0
    run_info = {
        "config": cfg.model_dump(mode="json"),
        "variant": variant,
        "pipeline_model": pipeline_model,
        "assistant_axis_source": assistant_axis_source_label(),
        "seed": cfg.seed,
        "axis_convention": "default_minus_role",  # per upstream 5_axis.py
        "sampling": {
            "temperature": axis_cfg.temperature,
            "top_p": axis_cfg.top_p,
            "max_new_tokens": axis_cfg.max_new_tokens,
        },
        "axis_shape": list(axis_tensor.shape),
        "axis_norms_per_layer_first10": axis_norms[:10].tolist(),
        "axis_norm_mean": float(axis_norms.mean()),
        "axis_norm_max_layer": int(axis_norms.argmax()),
        "n_role_files_staged": n_role_files,
        "n_role_vectors_built": n_role_vectors_built,
        "timings_seconds": timings,
    }
    with open(out_root / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"\nWrote run_info.json — axis shape {axis_tensor.shape}, "
          f"max-norm layer {run_info['axis_norm_max_layer']}, "
          f"mean-norm {run_info['axis_norm_mean']:.4f}")

    if upload_hf:
        _hf_upload_axis_dir(
            out_root, hf_subpath=cfg.hf_subpath,
            variant=variant, checkpoint_label="final",
        )

    return run_info


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["smoke", "balanced", "full"], default="smoke")
    parser.add_argument(
        "--variant",
        default="base",
        choices=["base", LORA_SOUP_VARIANT_NAME],
        help="Which model to build the axis on. base = unadapted Llama; "
             f"{LORA_SOUP_VARIANT_NAME} = base + LoRA soup pre-merged into "
             "a standalone model dir (composition controlled by "
             "cfg.lora_soup.adapters).",
    )
    parser.add_argument("--run-slug", help="Override run_slug (defaults to preset's slug)")
    parser.add_argument("--num-roles", type=int, help="Override num_roles")
    parser.add_argument("--num-questions", type=int, help="Override num_questions")
    parser.add_argument("--num-sysprompts", type=int, help="Override num_sysprompts_per_role")
    parser.add_argument("--judge-model", help="Override judge_model")
    parser.add_argument("--upload-hf", action="store_true", help="Upload artefacts to monorepo")
    args = parser.parse_args()

    cfg = get_preset(args.preset)
    if args.run_slug:
        cfg.run_slug = args.run_slug
    if args.num_roles is not None:
        cfg.axis.num_roles = args.num_roles
    if args.num_questions is not None:
        cfg.axis.num_questions = args.num_questions
    if args.num_sysprompts is not None:
        cfg.axis.num_sysprompts_per_role = args.num_sysprompts
    if args.judge_model:
        cfg.axis.judge_model = args.judge_model

    _seed_everything(cfg.seed)

    print("=" * 70)
    print(f"  Run slug: {cfg.run_slug}")
    print(f"  Variant:  {args.variant}")
    print(f"  Model:    {cfg.axis.base_model}")
    print(f"  Out dir:  {cfg.axis_dir(args.variant)}")
    print(f"  Knobs:    {cfg.axis.num_roles or 'all'} roles × "
          f"{cfg.axis.num_questions} questions × "
          f"{cfg.axis.num_sysprompts_per_role or 'all'} sysprompts")
    print(f"  Judge:    {cfg.axis.judge_model}")
    print("=" * 70)

    build_axis(cfg, variant=args.variant, upload_hf=args.upload_hf)


if __name__ == "__main__":
    main()
