"""Compute the persona activation axis + per-layer projection ranges for one
OCEAN direction trained with the vanton4 pipeline, and upload artifacts to the
HuggingFace monorepo next to the LoRA.

This generalizes the per-persona notebooks in
``scripts_dev/activation_capping_notebooks/ocean/`` into a single parameterized
script. Runs one persona per invocation; batch across all 10 via
``run_all_vanton4.sh`` (or the master runner ``run_everything_vanton4.sh``).

Usage
-----
    uv run python scripts_dev/activation_capping/ocean/vanton4/compute_axis.py \\
        --persona {o_plus,o_minus,c_plus,c_minus,e_plus,e_minus,a_plus,a_minus,n_plus,n_minus} \\
        [--max-samples N] [--dry-run] [--skip-upload]

Outputs
-------
Local (``scratch/llama_8b_instruct/activation_capping/{persona}_vanton4/``):
    {persona}_axis.pt               # axis + metadata incl. recommended_capping_layers
    {persona}_per_layer_range.pt    # (min, max) projection range per layer
    {persona}_activations.pt        # raw base/lora activations + responses
    axis_norms_per_layer.png
    relative_axis_norms.png
    projection_boxplots_per_layer.png
    run_info.json                   # provenance

Monorepo (``<lora_parent>/activation_capping/``, sibling of ``lora/`` and ``evals/``):
    All of the above, minus intermediate local-only caches, via ``upload_folder``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset  # noqa: F401  (kept for parity with notebook imports)
from dotenv import load_dotenv
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src_dev.activation_capping.axis import (
    best_contiguous_window,
    cohens_d_per_layer,
    compute_axis,
    compute_per_layer_range,
    extract_response_activations_batched,
    flatten_rollouts,
    generate_responses_batched,
)
from src_dev.common.lora_catalogue import HF_REPO, LoraHFCatalogue
from src_dev.utils.hf_hub import download_from_dataset_repo, login_from_env

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_VERSION = "vanton4"
MONOREPO_ID = HF_REPO

MAX_NEW_TOKENS = 256
BATCH_SIZE = 16
NUM_ROLLOUTS = 3
TEMPERATURE = 1.0
TOP_P: float | None = None
WINDOW_SIZE = 15
SEED = 42

REPO_ROOT = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
)
DATASET_PATH = REPO_ROOT / "data" / "claude-generated-prompts-for-activations-generations.jsonl"

PERSONA_CHOICES = [
    "o_plus", "o_minus", "c_plus", "c_minus",
    "e_plus", "e_minus", "a_plus", "a_minus",
    "n_plus", "n_minus",
]


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=str(REPO_ROOT), text=True).strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    out = _git("status", "--porcelain")
    if out is None:
        return None
    return bool(out)


def _resolve_paths(persona: str) -> tuple[str, str, Path, Path]:
    """Resolve (lora_path_in_repo, monorepo_axis_upload_path, local_output_dir, local_lora_cache)."""
    lora_path_in_repo: str = getattr(LoraHFCatalogue(), persona)
    # catalogue value ends in ".../{version}/lora/<adapter-name>" — strip the last two segments
    lora_parent = str(Path(lora_path_in_repo).parent.parent)
    monorepo_upload_path = f"{lora_parent}/activation_capping"
    output_dir = REPO_ROOT / "scratch" / "llama_8b_instruct" / "activation_capping" / f"{persona}_{LORA_VERSION}"
    local_lora_cache = REPO_ROOT / "scratch" / "lora_cache" / f"{persona}_{LORA_VERSION}"
    return lora_path_in_repo, monorepo_upload_path, output_dir, local_lora_cache


def _load_prompts(max_samples: int | None) -> list[str]:
    with open(DATASET_PATH) as f:
        rows = [json.loads(line) for line in f]
    if max_samples is not None:
        rows = rows[:max_samples]
    return [r["question"] for r in rows]


def _make_plots(
    axis: torch.Tensor,
    base_stack: torch.Tensor,
    lora_stack: torch.Tensor,
    output_dir: Path,
) -> None:
    axis_norms = axis.norm(dim=1).numpy()
    base_mean_norms = base_stack.float().norm(dim=2).mean(dim=0).numpy()
    lora_mean_norms = lora_stack.float().norm(dim=2).mean(dim=0).numpy()
    avg_mean_norms = (base_mean_norms + lora_mean_norms) / 2
    x = np.arange(len(axis_norms))

    # 1. Raw norms
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.25
    ax.bar(x - width, base_mean_norms, width, label="Base mean activation norm", color="steelblue", alpha=0.7)
    ax.bar(x, lora_mean_norms, width, label="LoRA mean activation norm", color="coral", alpha=0.7)
    ax.bar(x + width, axis_norms, width, label="Axis norm", color="green", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 norm")
    ax.set_title("Per-layer norms: activations vs axis")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "axis_norms_per_layer.png", dpi=150)
    plt.close(fig)

    # 2. Norm ratios
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, axis_norms / (base_mean_norms + 1e-8), marker="o", markersize=3, label="Axis / base norm", color="steelblue")
    ax.plot(x, axis_norms / (lora_mean_norms + 1e-8), marker="s", markersize=3, label="Axis / LoRA norm", color="coral")
    ax.plot(x, axis_norms / (avg_mean_norms + 1e-8), marker="^", markersize=3, label="Axis / avg norm", color="green", linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Axis norm / activation norm")
    ax.set_title("Relative persona signal strength per layer")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "relative_axis_norms.png", dpi=150)
    plt.close(fig)

    # 3. Normalized projection boxplots per layer
    n_layers = axis.shape[0]
    base_projs = []
    lora_projs = []
    for layer_idx in range(n_layers):
        ax_vec = axis[layer_idx].float()
        ax_normed = ax_vec / (ax_vec.norm() + 1e-8)
        proj_base = (base_stack[:, layer_idx, :].float() @ ax_normed).numpy()
        proj_lora = (lora_stack[:, layer_idx, :].float() @ ax_normed).numpy()
        base_norm_mean = base_stack[:, layer_idx, :].float().norm(dim=-1).mean().item()
        lora_norm_mean = lora_stack[:, layer_idx, :].float().norm(dim=-1).mean().item()
        b = proj_base / base_norm_mean
        l = proj_lora / lora_norm_mean
        center = (b.mean() + l.mean()) / 2
        base_projs.append(b - center)
        lora_projs.append(l - center)

    positions_base = np.arange(n_layers) * 3
    positions_lora = positions_base + 1
    fig, ax = plt.subplots(figsize=(20, 6))
    bp_base = ax.boxplot(base_projs, positions=positions_base, widths=0.8, patch_artist=True, showfliers=False)
    bp_lora = ax.boxplot(lora_projs, positions=positions_lora, widths=0.8, patch_artist=True, showfliers=False)
    for patch in bp_base["boxes"]:
        patch.set_facecolor("cornflowerblue")
        patch.set_alpha(0.7)
    for patch in bp_lora["boxes"]:
        patch.set_facecolor("salmon")
        patch.set_alpha(0.7)
    ax.set_xticks(positions_base + 0.5)
    ax.set_xticklabels([str(i) for i in range(n_layers)], fontsize=8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized projection (proj / mean activation norm)")
    ax.set_title("Normalized projection onto axis — Base vs LoRA per layer")
    ax.legend([bp_base["boxes"][0], bp_lora["boxes"][0]], ["Base", "LoRA"], loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "projection_boxplots_per_layer.png", dpi=150)
    plt.close(fig)


def run(persona: str, *, max_samples: int | None, dry_run: bool, skip_upload: bool) -> None:
    _set_seeds(SEED)
    login_from_env()

    lora_path_in_repo, monorepo_upload_path, output_dir, local_lora_cache = _resolve_paths(persona)

    print("=" * 70)
    print(f"  Persona:               {persona}")
    print(f"  Base model:            {BASE_MODEL}")
    print(f"  LoRA repo:             {MONOREPO_ID}")
    print(f"  LoRA path_in_repo:     {lora_path_in_repo}")
    print(f"  Monorepo upload path:  {monorepo_upload_path}")
    print(f"  Local output dir:      {output_dir}")
    print(f"  Local LoRA cache:      {local_lora_cache}")
    print(f"  Dataset:               {DATASET_PATH.relative_to(REPO_ROOT)}")
    print(f"  max_samples:           {max_samples}")
    print("=" * 70)

    if dry_run:
        print("[--dry-run] exiting before generation / upload.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    questions = _load_prompts(max_samples)
    print(f"Loaded {len(questions)} prompts")

    # LoRA adapter download
    download_from_dataset_repo(
        repo_id=MONOREPO_ID,
        path_in_repo=lora_path_in_repo,
        local_dir=local_lora_cache,
    )
    lora_local_path = local_lora_cache / lora_path_in_repo
    print(f"LoRA downloaded to {lora_local_path}")

    # Model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model = PeftModel.from_pretrained(model, str(lora_local_path))
    model.eval()

    # Phase 1: base rollouts
    print(f"\nPhase 1/4: base rollouts ({NUM_ROLLOUTS} x {len(questions)} questions)")
    with model.disable_adapter():
        base_rollouts = generate_responses_batched(
            model, tokenizer, questions,
            max_new_tokens=MAX_NEW_TOKENS, batch_size=BATCH_SIZE,
            num_rollouts=NUM_ROLLOUTS, temperature=TEMPERATURE, top_p=TOP_P,
        )

    # Phase 2: LoRA rollouts
    print(f"Phase 2/4: LoRA rollouts ({NUM_ROLLOUTS} x {len(questions)} questions)")
    lora_rollouts = generate_responses_batched(
        model, tokenizer, questions,
        max_new_tokens=MAX_NEW_TOKENS, batch_size=BATCH_SIZE,
        num_rollouts=NUM_ROLLOUTS, temperature=TEMPERATURE, top_p=TOP_P,
    )

    base_qs_flat, base_resps_flat = flatten_rollouts(questions, base_rollouts)
    lora_qs_flat, lora_resps_flat = flatten_rollouts(questions, lora_rollouts)

    base_convs = [
        [{"role": "user", "content": q}, {"role": "assistant", "content": r}]
        for q, r in zip(base_qs_flat, base_resps_flat)
    ]
    lora_convs = [
        [{"role": "user", "content": q}, {"role": "assistant", "content": r}]
        for q, r in zip(lora_qs_flat, lora_resps_flat)
    ]

    # Phase 3: base activations
    print("Phase 3/4: base activations")
    with model.disable_adapter():
        base_stack = extract_response_activations_batched(
            model, tokenizer, base_convs, batch_size=BATCH_SIZE,
        )
    print(f"  base activations: {base_stack.shape}")

    # Phase 4: LoRA activations
    print("Phase 4/4: LoRA activations")
    lora_stack = extract_response_activations_batched(
        model, tokenizer, lora_convs, batch_size=BATCH_SIZE,
    )
    print(f"  LoRA activations: {lora_stack.shape}")

    # Axis, Cohen's d, best window
    axis = compute_axis(base_stack, lora_stack)
    cohens_d = cohens_d_per_layer(base_stack, lora_stack, axis)
    capping_layers = best_contiguous_window(cohens_d, window_size=WINDOW_SIZE)
    best_sep_layer = int(np.argmax(cohens_d))
    print(f"\nBest layer by Cohen's d: {best_sep_layer} (d={cohens_d[best_sep_layer]:.3f})")
    print(f"Recommended capping layers (window={WINDOW_SIZE}): {capping_layers[0]}-{capping_layers[-1]}")

    # Per-layer projection ranges (all layers)
    all_layers = list(range(axis.shape[0]))
    per_layer_range = compute_per_layer_range(base_stack, lora_stack, axis, all_layers)

    # ----------------------------- Save artifacts -----------------------------
    torch.save(
        {
            "base": base_stack,
            "lora": lora_stack,
            "base_responses": base_resps_flat,
            "lora_responses": lora_resps_flat,
            "base_rollouts": base_rollouts,
            "lora_rollouts": lora_rollouts,
        },
        output_dir / f"{persona}_activations.pt",
    )
    torch.save(
        {
            "axis": axis,
            "metadata": {
                "model": BASE_MODEL,
                "lora_hf_dataset_repo": MONOREPO_ID,
                "lora_path_in_repo": lora_path_in_repo,
                "persona": persona,
                "n_samples": int(base_stack.shape[0]),
                "best_layer_by_separation": best_sep_layer,
                "recommended_capping_layers": capping_layers,
                "dataset": str(DATASET_PATH),
            },
        },
        output_dir / f"{persona}_axis.pt",
    )
    torch.save(
        {
            "per_layer_range": per_layer_range,
            "metadata": {
                "model": BASE_MODEL,
                "lora_hf_dataset_repo": MONOREPO_ID,
                "lora_path_in_repo": lora_path_in_repo,
                "persona": persona,
                "n_samples": int(base_stack.shape[0]),
                "layers": all_layers,
                "dataset": str(DATASET_PATH),
            },
        },
        output_dir / f"{persona}_per_layer_range.pt",
    )

    _make_plots(axis, base_stack, lora_stack, output_dir)

    # Provenance
    run_info = {
        "persona": persona,
        "base_model": BASE_MODEL,
        "lora": {"hf_dataset_repo": MONOREPO_ID, "path_in_repo": lora_path_in_repo},
        "lora_version": LORA_VERSION,
        "dataset": str(DATASET_PATH.relative_to(REPO_ROOT)),
        "n_samples": int(base_stack.shape[0]),
        "num_rollouts": NUM_ROLLOUTS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "batch_size": BATCH_SIZE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "window_size": WINDOW_SIZE,
        "best_layer_by_separation": best_sep_layer,
        "capping_layers_recommended": list(map(int, capping_layers)),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": _git_dirty(),
        },
        "script": "scripts_dev/activation_capping/ocean/vanton4/compute_axis.py",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(output_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2, default=str)

    print(f"\nSaved all artifacts under {output_dir}")

    if skip_upload:
        print("[--skip-upload] not uploading to monorepo.")
        return

    # ----------------------------- Upload --------------------------------------
    upload_staging = output_dir / "monorepo_upload"
    if upload_staging.exists():
        shutil.rmtree(upload_staging)
    upload_staging.mkdir(parents=True)

    artifacts = [
        output_dir / f"{persona}_axis.pt",
        output_dir / f"{persona}_per_layer_range.pt",
        output_dir / f"{persona}_activations.pt",
        output_dir / "axis_norms_per_layer.png",
        output_dir / "relative_axis_norms.png",
        output_dir / "projection_boxplots_per_layer.png",
        output_dir / "run_info.json",
    ]
    for p in artifacts:
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  staging {p.name} ({size_mb:.1f} MB)")
            shutil.copy2(p, upload_staging / p.name)

    api = HfApi()
    api.upload_folder(
        folder_path=str(upload_staging),
        repo_id=MONOREPO_ID,
        repo_type="dataset",
        path_in_repo=monorepo_upload_path,
        commit_message=f"activation_capping: add {persona} {LORA_VERSION} axis + per-layer ranges",
    )
    print(f"\nUploaded to https://huggingface.co/datasets/{MONOREPO_ID}/tree/main/{monorepo_upload_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona", required=True, choices=PERSONA_CHOICES)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size for smoke tests.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve config and exit before generation/upload.")
    parser.add_argument("--skip-upload", action="store_true", help="Run everything locally but skip HF upload.")
    args = parser.parse_args()

    run(args.persona, max_samples=args.max_samples, dry_run=args.dry_run, skip_upload=args.skip_upload)


if __name__ == "__main__":
    main()
