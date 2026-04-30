#!/usr/bin/env python3
"""Phase 4 — Project per-turn assistant activations onto the Assistant Axis.

For every rollout produced in Phase 3, walk each assistant turn, extract
mean response-token activations at every layer, and project onto the
axis built in Phase 1. The output is a per-turn-position table that
Phase 5 plots.

We use HuggingFace transformers (not vLLM) for the extraction pass since
we need forward hooks on the residual stream — same setup as Phase 1's
upstream ``2_activations.py``. The model is loaded ONCE and reused
across all conditions × domains.

Output: ``{scratch_dir}/drift_projections.jsonl`` — one row per
(condition, domain, conv_id, turn_position) with projections at every
layer.

Usage::

    uv run python -m scripts_dev.persona_drift_assistant_axis.project_drift \\
        --preset smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm.auto import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src_dev.activation_capping.assistant_axis_loader import load_axis  # noqa: E402
from src_dev.activation_capping.axis import (  # noqa: E402
    extract_response_activations_batched,
)
from src_dev.datasets import load_samples, materialize_canonical_samples  # noqa: E402
from scripts_dev.persona_drift_assistant_axis.config import (  # noqa: E402
    ExperimentConfig,
    get_preset,
)


# ── Load HF model (no LoRA, no capping) for clean activation extraction ──


def _load_extraction_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {model_name} for activation extraction...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


# ── Iterate rollouts and build (sample_id, turn_position, slice) records ──


def _conversation_assistant_slices(messages: list[dict]) -> list[tuple[int, list[dict]]]:
    """For a conversation, return ``(turn_position, prefix_slice)`` per assistant turn.

    A "prefix_slice" is the conversation truncated at that assistant turn
    (inclusive), suitable for feeding to
    :func:`extract_response_activations_batched`.
    """
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
        msgs = [
            {"role": m.role, "content": m.content}
            for m in sample.messages
        ]
        for turn_pos, slice_msgs in _conversation_assistant_slices(msgs):
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "turn_position": turn_pos,
                    "messages": slice_msgs,
                }
            )
    return records


# ── Project records onto axis ────────────────────────────────────────────


def _project_records(
    model,
    tokenizer,
    records: list[dict],
    *,
    axis: torch.Tensor,
    batch_size: int = 4,
) -> list[dict]:
    """Run forward passes, extract activations, project onto axis.

    Adds ``projection_per_layer`` (list[float], length n_layers) to each
    record and returns the records.
    """
    if not records:
        return []
    n_layers = axis.shape[0]
    ax_norm = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)

    convs = [r["messages"] for r in records]

    # Process in chunks to avoid materialising a giant activation tensor.
    chunk = 64
    enriched: list[dict] = []
    for start in tqdm(range(0, len(convs), chunk), desc="Projection batches"):
        chunk_convs = convs[start : start + chunk]
        chunk_records = records[start : start + chunk]
        acts = extract_response_activations_batched(
            model, tokenizer, chunk_convs, batch_size=batch_size,
        )  # (N, n_layers, hidden_dim)
        # Project: einsum('Nld,ld->Nl')
        proj = torch.einsum(
            "nld,ld->nl", acts.float(), ax_norm.float()
        ).cpu().numpy()
        for r, p in zip(chunk_records, proj):
            enriched.append(
                {
                    **r,
                    "projection_per_layer": [float(x) for x in p],
                }
            )
        del acts
    return enriched


# ── Top-level orchestrator ───────────────────────────────────────────────


def project_drift(cfg: ExperimentConfig) -> Path:
    """Run projection extraction over all condition × domain rollouts."""
    drift_root = cfg.scratch_dir / "drift_rollouts"
    if not drift_root.exists():
        raise SystemExit(f"No drift rollouts at {drift_root}; run Phase 3 first.")
    axis_path = cfg.scratch_dir / "axis.pt"
    if not axis_path.exists():
        raise SystemExit(f"axis.pt missing at {axis_path}; run Phase 1 first.")
    axis = load_axis(axis_path)

    output_path = cfg.scratch_dir / "drift_projections.jsonl"

    # Discover (condition, domain) directories.
    cells: list[tuple[str, str, Path]] = []
    for cond_dir in sorted(p for p in drift_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for domain_dir in sorted(p for p in cond_dir.iterdir() if p.is_dir()):
            cells.append((cond_dir.name, domain_dir.name, domain_dir))
    if not cells:
        raise SystemExit(f"No (condition, domain) rollout dirs found under {drift_root}")
    print(f"Discovered {len(cells)} (condition, domain) cells:")
    for c, d, _ in cells:
        print(f"  {c} / {d}")

    model, tokenizer = _load_extraction_model(cfg.axis.base_model)

    rows_written = 0
    with open(output_path, "w") as fh:
        for condition, domain, run_dir in cells:
            records = _gather_slices(run_dir)
            print(f"\n{condition} / {domain}: {len(records)} (sample, turn) slices")
            if not records:
                continue
            enriched = _project_records(
                model, tokenizer, records,
                axis=axis, batch_size=cfg.axis.activation_batch_size,
            )
            for r in enriched:
                fh.write(json.dumps({
                    "condition": condition,
                    "domain": domain,
                    "sample_id": r["sample_id"],
                    "turn_position": r["turn_position"],
                    "projection_per_layer": r["projection_per_layer"],
                }) + "\n")
                rows_written += 1
    print(f"\nWrote {rows_written} projection rows to {output_path}")
    return output_path


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--run-slug", help="Override run_slug")
    args = parser.parse_args()

    cfg = get_preset(args.preset)
    if args.run_slug:
        cfg.run_slug = args.run_slug
    project_drift(cfg)


if __name__ == "__main__":
    main()
