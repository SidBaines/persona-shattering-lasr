#!/usr/bin/env python3
"""Phase 4 — Project per-turn assistant activations onto every Assistant Axis.

For every rollout produced in Phase 3, walk each assistant turn, extract
mean response-token activations using the correct model for that condition
(see ``_CONDITION_EXTRACTION_VARIANT``), and project onto every axis built
in Phase 1 (base, lora_soup_c_plus_o_minus, …).

The output JSONL has one row per (condition, domain, sample, turn,
axis_variant) combination — Phase 5 facets by ``axis_variant`` to produce
one trajectory plot per axis.

Why per-condition extraction model? The activations a model produces
depend on its weights. For the LoRA condition, projecting BASE-model
activations of LoRA outputs onto any axis would be measuring "how the
base model represents these utterances", not "where the LoRA model puts
itself in activation space". So:

  vanilla              → base model, no hooks
  activation_capping   → base model + capping hooks (mirrors generation)
  lora_soup_*          → LoRA-merged model, no hooks

We process conditions in batches grouped by extraction model so each
model is loaded only once.

Usage::

    uv run python -m scripts_dev.persona_drift_assistant_axis.project_drift \\
        --preset smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm.auto import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src_dev.activation_capping.assistant_axis_loader import (  # noqa: E402
    apply_assistant_axis_capping,
    load_axis,
    load_capping_config,
)
from src_dev.activation_capping.axis import (  # noqa: E402
    extract_response_activations_batched,
)
from src_dev.datasets import load_samples, materialize_canonical_samples  # noqa: E402
from scripts_dev.persona_drift_assistant_axis.config import (  # noqa: E402
    ExperimentConfig,
    get_preset,
)


# Condition → which axis-variant's model to use for activation extraction.
# Capping condition shares the base extraction model but ALSO needs the
# capping hooks active during extraction (otherwise the bounded-projection
# property of capping won't be visible in the trajectory plot).
_CONDITION_EXTRACTION_VARIANT: dict[str, str] = {
    "vanilla": "base",
    "activation_capping": "base",  # + hooks
    "lora_soup_c_plus_o_minus": "lora_soup_c_plus_o_minus",
}
_CAPPING_CONDITIONS = {"activation_capping"}


# ── Axis discovery + sanity-check ────────────────────────────────────────


def _cosine_per_layer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-layer cos similarity for two ``(n_layers, hidden_dim)`` axes."""
    a = a.float()
    b = b.float()
    a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return (a_n * b_n).sum(dim=-1)


def discover_axes(cfg: ExperimentConfig) -> dict[str, torch.Tensor]:
    """Load every built axis from ``{scratch_dir}/axes/*/axis.pt``."""
    variants = cfg.discover_axis_variants()
    if not variants:
        raise SystemExit(
            f"No axes found under {cfg.scratch_dir / 'axes'}; "
            f"run `build_axis.py --variant base` first."
        )
    axes = {v: load_axis(cfg.axis_path(v)) for v in variants}
    print(f"Discovered {len(axes)} axis variant(s): {list(axes)}")
    for name, ax in axes.items():
        norms = ax.float().norm(dim=-1)
        print(f"  {name}: shape={tuple(ax.shape)}, "
              f"norm-mean={norms.mean():.3f}, max-norm-layer={int(norms.argmax())}")
    return axes


def report_axis_similarity(axes: dict[str, torch.Tensor], output_path: Path) -> None:
    """Compute per-layer cosine similarity between every axis pair, save report."""
    if len(axes) < 2:
        return
    names = list(axes)
    lines: list[str] = ["# Pairwise per-layer cosine similarity between axes\n"]
    n_layers = next(iter(axes.values())).shape[0]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            cos = _cosine_per_layer(axes[a], axes[b])
            mean = float(cos.mean())
            mn, mx = float(cos.min()), float(cos.max())
            lines.append(
                f"\n{a}  vs  {b}\n"
                f"  mean={mean:+.4f}  min={mn:+.4f}  max={mx:+.4f}\n"
                f"  per-layer ({n_layers}): "
                + " ".join(f"{c:+.2f}" for c in cos.tolist())
                + "\n"
            )
            print(f"  cos({a}, {b}) mean={mean:+.4f} (min={mn:+.4f} max={mx:+.4f})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines))
    print(f"  similarity report → {output_path}")


# ── Slice extraction ─────────────────────────────────────────────────────


def _conversation_assistant_slices(messages: list[dict]) -> list[tuple[int, list[dict]]]:
    """Return ``(turn_position, prefix_slice)`` per assistant turn in a conversation."""
    out: list[tuple[int, list[dict]]] = []
    asst_count = 0
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            slice_msgs = [
                {"role": mm["role"], "content": mm["content"]}
                for mm in messages[: i + 1]
                if mm["role"] in {"user", "assistant", "system"}
            ]
            out.append((asst_count, slice_msgs))
            asst_count += 1
    return out


def _gather_slices(run_dir: Path) -> list[dict]:
    """Read all rollouts under ``run_dir`` and return per-turn slice records."""
    if not run_dir.exists():
        return []
    materialize_canonical_samples(run_dir)
    samples = load_samples(run_dir)
    records: list[dict] = []
    for sample in samples:
        msgs = [{"role": m.role, "content": m.content} for m in sample.messages]
        for turn_pos, slice_msgs in _conversation_assistant_slices(msgs):
            records.append(
                {"sample_id": sample.sample_id,
                 "turn_position": turn_pos,
                 "messages": slice_msgs}
            )
    return records


# ── Model loading ────────────────────────────────────────────────────────


def _load_extraction_model(model_path_or_name: str):
    """Load HF model + tokenizer for activation extraction."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {model_path_or_name} for activation extraction...")
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def _free_model(model) -> None:
    """Release a HF model from GPU memory."""
    try:
        model.cpu()
    except Exception:  # noqa: BLE001
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Projection ───────────────────────────────────────────────────────────


def _project_records(
    model,
    tokenizer,
    records: list[dict],
    *,
    axes: dict[str, torch.Tensor],
    batch_size: int,
) -> list[dict]:
    """Forward-pass each record, project onto every axis, return enriched rows.

    One forward pass per record, then N cheap dot-product projections
    (one per axis variant). Output is one row per (record, axis_variant).
    """
    if not records:
        return []
    # Pre-normalise axes once (they're tiny).
    axes_normed = {
        name: (ax.float() / (ax.float().norm(dim=-1, keepdim=True) + 1e-8))
        for name, ax in axes.items()
    }

    convs = [r["messages"] for r in records]
    chunk = 64
    out_rows: list[dict] = []

    for start in tqdm(range(0, len(convs), chunk), desc="Projection batches"):
        chunk_convs = convs[start : start + chunk]
        chunk_records = records[start : start + chunk]
        acts = extract_response_activations_batched(
            model, tokenizer, chunk_convs, batch_size=batch_size,
        )  # (N, n_layers, hidden_dim)
        acts_f = acts.float()
        for axis_variant, ax_n in axes_normed.items():
            proj = torch.einsum("nld,ld->nl", acts_f, ax_n).cpu().numpy()
            for r, p in zip(chunk_records, proj):
                out_rows.append({
                    **r,
                    "axis_variant": axis_variant,
                    "projection_per_layer": [float(x) for x in p],
                })
        del acts, acts_f
    return out_rows


# ── Top-level ────────────────────────────────────────────────────────────


def project_drift(cfg: ExperimentConfig) -> Path:
    """Run projection extraction over all condition × domain rollouts."""
    drift_root = cfg.scratch_dir / "drift_rollouts"
    if not drift_root.exists():
        raise SystemExit(f"No drift rollouts at {drift_root}; run Phase 3 first.")

    axes = discover_axes(cfg)
    report_axis_similarity(axes, cfg.scratch_dir / "axis_cosine_similarity.txt")

    output_path = cfg.scratch_dir / "drift_projections.jsonl"

    # Discover (condition, domain) directories.
    cells: list[tuple[str, str, Path]] = []
    for cond_dir in sorted(p for p in drift_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for domain_dir in sorted(p for p in cond_dir.iterdir() if p.is_dir()):
            cells.append((cond_dir.name, domain_dir.name, domain_dir))
    if not cells:
        raise SystemExit(f"No (condition, domain) rollout dirs found under {drift_root}")
    print(f"\nDiscovered {len(cells)} (condition, domain) cells:")
    for c, d, _ in cells:
        print(f"  {c} / {d}")

    # Group conditions by extraction-model variant. Within each group, we
    # load the model once.  Capping conditions also need hooks applied
    # during extraction; we apply them per-condition (they're cheap to
    # register/remove) since the model itself is shared.
    by_extract_variant: dict[str, list[tuple[str, str, Path]]] = {}
    for cond, dom, run_dir in cells:
        ext_variant = _CONDITION_EXTRACTION_VARIANT.get(cond)
        if ext_variant is None:
            print(f"  WARN: condition {cond!r} has no extraction-variant mapping; skipping.")
            continue
        by_extract_variant.setdefault(ext_variant, []).append((cond, dom, run_dir))

    rows_written = 0
    with open(output_path, "w") as fh:
        for ext_variant, cell_list in by_extract_variant.items():
            model_path = cfg.variant_to_model(ext_variant)
            print(f"\n=== Extraction model: {ext_variant} ({model_path}) ===")
            model, tokenizer = _load_extraction_model(model_path)

            try:
                for cond, dom, run_dir in cell_list:
                    records = _gather_slices(run_dir)
                    print(f"\n{cond} / {dom}: {len(records)} (sample, turn) slices")
                    if not records:
                        continue

                    # For capping, register hooks for the duration of extraction.
                    steering = None
                    if cond in _CAPPING_CONDITIONS:
                        capping_cfg = load_capping_config(cfg.capping_config_path)
                        steering = apply_assistant_axis_capping(
                            model, axes["base"], capping_cfg, debug=False,
                        )
                        print(f"  capping hooks active on layers {capping_cfg['layers']}")

                    try:
                        enriched = _project_records(
                            model, tokenizer, records,
                            axes=axes,
                            batch_size=cfg.axis.activation_batch_size,
                        )
                    finally:
                        if steering is not None:
                            steering.remove()

                    for r in enriched:
                        fh.write(json.dumps({
                            "condition": cond,
                            "domain": dom,
                            "extraction_variant": ext_variant,
                            "axis_variant": r["axis_variant"],
                            "sample_id": r["sample_id"],
                            "turn_position": r["turn_position"],
                            "projection_per_layer": r["projection_per_layer"],
                        }) + "\n")
                        rows_written += 1
            finally:
                _free_model(model)

    print(f"\nWrote {rows_written} projection rows to {output_path}")
    return output_path


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["smoke", "balanced", "full"], default="smoke")
    parser.add_argument("--run-slug", help="Override run_slug")
    args = parser.parse_args()

    cfg = get_preset(args.preset)
    if args.run_slug:
        cfg.run_slug = args.run_slug
    project_drift(cfg)


if __name__ == "__main__":
    main()
