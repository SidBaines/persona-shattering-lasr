#!/usr/bin/env python3
"""Phase 4 — Project per-turn assistant activations onto every Assistant Axis.

For every rollout produced in Phase 3, walk each assistant turn, extract
mean response-token activations using the correct model for that condition
(see ``_CONDITION_EXTRACTION_VARIANT``), and project onto every axis built
in Phase 1 (base, lora-soup, …).

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
import random
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm.auto import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src_dev.activation_capping.assistant_axis_loader import (  # noqa: E402
    apply_assistant_axis_capping,
    cohens_d,
    diagnose_capping_direction,
    load_axis,
    load_capping_config,
    load_role_activations,
    print_capping_diagnosis,
    project_onto_axis,
    remove_capping_hooks,
)
from src_dev.activation_capping.axis import (  # noqa: E402
    extract_response_activations_batched,
)
from src_dev.datasets import load_samples, materialize_canonical_samples  # noqa: E402
from scripts_dev.persona_drift_assistant_axis.config import (  # noqa: E402
    LORA_SOUP_VARIANT_NAME,
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
    LORA_SOUP_VARIANT_NAME: LORA_SOUP_VARIANT_NAME,
}
_CAPPING_CONDITIONS = {"activation_capping"}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Axis discovery + sanity-check ────────────────────────────────────────


def _cosine_per_layer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-layer cos similarity for two ``(n_layers, hidden_dim)`` axes."""
    a = a.float()
    b = b.float()
    a_n = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_n = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return (a_n * b_n).sum(dim=-1)


def discover_axes(
    cfg: ExperimentConfig, *, expected_variants: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Load every built axis from ``{scratch_dir}/axes/*/axis.pt``.

    If ``expected_variants`` is provided and any are missing on disk, prints
    a loud warning. (We don't hard-fail because the user might have chosen
    to run with only one axis intentionally.)
    """
    variants = cfg.discover_axis_variants()
    if not variants:
        raise SystemExit(
            f"No axes found under {cfg.scratch_dir / 'axes'}; "
            f"run `build_axis.py --variant base` first."
        )
    if expected_variants is not None:
        missing = [v for v in expected_variants if v not in variants]
        if missing:
            print(f"\n  WARNING: expected axis variants {expected_variants}, "
                  f"but missing on disk: {missing}.")
            print(f"           Discovered only: {variants}.")
            print(f"           Continuing with what's available — but "
                  "if you want the multi-axis comparison, run "
                  f"`build_axis.py --variant <missing>` for each missing variant.\n")
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


def report_axis_quality(
    cfg: ExperimentConfig,
    axes: dict[str, torch.Tensor],
    output_path: Path,
) -> None:
    """For each axis variant, compute mean Cohen's d across the capping window
    using its own ``activations/`` cache (default vs. roles).

    Captures whether each per-variant axis represents a meaningful
    Assistant ↔ role-play contrast. Low Cohen's d ⇒ the axis is noisy,
    and trajectory plots in that variant should be interpreted skeptically.
    """
    capping_cfg = (
        load_capping_config(cfg.capping_config_path)
        if cfg.capping_config_path.exists() else None
    )

    lines: list[str] = ["# Per-variant axis quality (Cohen's d, default vs. role)\n"]
    quality: dict[str, dict] = {}
    for variant, axis in axes.items():
        acts_dir = cfg.axis_dir(variant) / "activations"
        if not acts_dir.exists():
            print(f"  {variant}: no activations/ dir; skipping quality check")
            continue
        try:
            default_acts = load_role_activations(acts_dir, "default")
        except Exception as exc:  # noqa: BLE001
            print(f"  {variant}: cannot load default activations ({exc}); skipping")
            continue

        # Sample up to 10 role files for a quick joint Cohen's d.
        role_files = sorted(p for p in acts_dir.glob("*.pt") if p.stem != "default")[:10]
        role_acts_list: list[torch.Tensor] = []
        for f in role_files:
            try:
                role_acts_list.append(load_role_activations(acts_dir, f.stem))
            except Exception:  # noqa: BLE001
                continue
        if not role_acts_list:
            print(f"  {variant}: no usable role activations; skipping")
            continue
        default_proj = project_onto_axis(default_acts, axis)
        role_proj = torch.cat(
            [project_onto_axis(a, axis) for a in role_acts_list], dim=0,
        )

        d_per_layer = np.array([
            cohens_d(default_proj[:, l].numpy(), role_proj[:, l].numpy())
            for l in range(axis.shape[0])
        ])
        # Window mean: prefer the capping window if available, else top-quarter.
        if capping_cfg is not None and capping_cfg.get("layers"):
            window = capping_cfg["layers"]
            window_label = f"capping_window={min(window)}:{max(window)}"
        else:
            n = axis.shape[0]
            window = list(range(int(n * 0.75), n))
            window_label = f"top_quarter={window[0]}:{window[-1]}"
        window_mean = float(d_per_layer[window].mean())
        peak_layer = int(np.argmax(np.abs(d_per_layer)))
        quality[variant] = {
            "window": window_label,
            "cohens_d_window_mean": window_mean,
            "cohens_d_peak_layer": peak_layer,
            "cohens_d_peak_value": float(d_per_layer[peak_layer]),
        }
        lines.append(
            f"\n{variant}\n"
            f"  {window_label}: mean Cohen's d = {window_mean:+.3f}\n"
            f"  peak layer {peak_layer}: Cohen's d = {d_per_layer[peak_layer]:+.3f}\n"
            f"  per-layer ({len(d_per_layer)}): "
            + " ".join(f"{d:+.2f}" for d in d_per_layer.tolist())
            + "\n"
        )
        flag = "" if window_mean > 0.3 else "  ← LOW SEPARATION (interpret with care)"
        print(f"  axis quality  {variant}: window mean Cohen's d = {window_mean:+.3f}{flag}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines))
    print(f"  axis-quality report → {output_path}")
    return quality


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


def _load_extraction_model(model_path_or_name: str, *, tokenizer_source: str):
    """Load HF model from ``model_path_or_name`` + tokenizer from
    ``tokenizer_source``.

    We always pull the tokenizer from the base model id rather than from
    the merged-LoRA model dir. ``merge_weighted_adapters`` saves whichever
    tokenizer it found first (the LoRA's, if it bundled one) — using that
    for activation extraction would risk subtle response-token-span
    shifts vs. the base-axis side, since the chat template / special-token
    handling could differ. The model weights and the tokenizer are
    independent for OCEAN-style LoRA fine-tunes (vocabulary unchanged).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading model: {model_path_or_name}")
    print(f"  Loading tokenizer: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
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

    # Expected variants come from the conditions configured for this run —
    # one for each axis-extraction-variant we'd want to project against.
    expected_variants = sorted({
        _CONDITION_EXTRACTION_VARIANT[c]
        for c in cfg.conditions
        if c in _CONDITION_EXTRACTION_VARIANT
    })
    axes = discover_axes(cfg, expected_variants=expected_variants)
    report_axis_similarity(axes, cfg.scratch_dir / "axis_cosine_similarity.txt")
    report_axis_quality(cfg, axes, cfg.scratch_dir / "axis_quality.txt")

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
    # load the model once. Capping conditions also need hooks applied
    # during extraction; we apply them per-condition (cheap to register/
    # remove) since the model itself is shared.
    by_extract_variant: dict[str, list[tuple[str, str, Path]]] = {}
    for cond, dom, run_dir in cells:
        ext_variant = _CONDITION_EXTRACTION_VARIANT.get(cond)
        if ext_variant is None:
            print(f"  WARN: condition {cond!r} has no extraction-variant mapping; skipping.")
            continue
        if ext_variant not in axes:
            print(f"  WARN: condition {cond!r} needs axis variant {ext_variant!r} which "
                  "wasn't discovered; skipping.")
            continue
        by_extract_variant.setdefault(ext_variant, []).append((cond, dom, run_dir))

    rows_written = 0
    with open(output_path, "w") as fh:
        for ext_variant, cell_list in by_extract_variant.items():
            model_path = cfg.variant_to_model(ext_variant)
            print(f"\n=== Extraction model: {ext_variant} ({model_path}) ===")
            # Always source the tokenizer from the base model so chat-template
            # behaviour is identical across base and LoRA-merged extraction
            # variants (B5 in HANDOVER review).
            model, tokenizer = _load_extraction_model(
                model_path, tokenizer_source=cfg.axis.base_model,
            )

            try:
                for cond, dom, run_dir in cell_list:
                    records = _gather_slices(run_dir)
                    print(f"\n{cond} / {dom}: {len(records)} (sample, turn) slices")
                    if not records:
                        continue

                    # For capping, register hooks for the duration of extraction.
                    capping_handle = None
                    if cond in _CAPPING_CONDITIONS:
                        capping_cfg = load_capping_config(cfg.capping_config_path)
                        capping_handle = apply_assistant_axis_capping(
                            model, axes["base"], capping_cfg, debug=False,
                        )
                        print(f"  capping hooks active "
                              f"(mode={capping_cfg.get('mode', 'floor')}) "
                              f"on layers {capping_cfg['layers']}")
                        # Quick direction check on the first cell only —
                        # don't pay the cost on every (condition, domain).
                        if cell_list[0] is (cond, dom, run_dir):
                            report = diagnose_capping_direction(
                                model, tokenizer, capping_handle,
                                axis=axes["base"], capping_config=capping_cfg,
                            )
                            print_capping_diagnosis(report)
                            if not report["passed"]:
                                raise SystemExit(
                                    "Phase-4 cap-direction diagnostic FAILED "
                                    "during extraction. STOP — projections "
                                    "would not reflect what we think they do."
                                )

                    try:
                        enriched = _project_records(
                            model, tokenizer, records,
                            axes=axes,
                            batch_size=cfg.axis.activation_batch_size,
                        )
                    finally:
                        if capping_handle is not None:
                            remove_capping_hooks(capping_handle)

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
    _seed_everything(cfg.seed)
    project_drift(cfg)


if __name__ == "__main__":
    main()
