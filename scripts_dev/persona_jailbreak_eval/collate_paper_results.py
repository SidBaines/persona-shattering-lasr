#!/usr/bin/env python3
"""Collate WildJailbreak runs into a single ``wj_paper_v1`` HF folder.

The paper figure pulls per-condition rates from one consolidated source
folder so:

  - every condition is reported on the same 500 prompts (400 harmful +
    100 benign — the ablation sample-id set), and
  - newer runs can supersede older results just by bumping the folder
    version (``wj_paper_v1`` → ``wj_paper_v2``) rather than mutating in
    place.

The 500-prompt sample-id set is the strict subset of ``wj_balanced_v2``
that matches ``wj_ablations_v1_v2``, ``wj_combo_a_plus_0p5_c_plus_0p5_v1``,
``wj_combo_a_plus_c_plus_v1``, and ``wj_combo_a_plus_1p0_c_plus_0p5_v1``
exactly. Verified by set-membership at collation time.

Output layout:

    {DEST_DIR}/judgments/judgments_<condition>.jsonl   — filtered rows
    {DEST_DIR}/aggregate/<metric>.csv                  — re-aggregated CSVs
    {DEST_DIR}/aggregate/summary_bars.{png,pdf}        — diagnostic plot
    {DEST_DIR}/manifest.json                           — provenance per condition

Run with::

    uv run python -m scripts_dev.persona_jailbreak_eval.collate_paper_results
    uv run python -m scripts_dev.persona_jailbreak_eval.collate_paper_results --no-upload
    uv run python -m scripts_dev.persona_jailbreak_eval.collate_paper_results --version v2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from huggingface_hub import HfApi, hf_hub_download

from src_dev.persona_jailbreak_eval.aggregate import (
    explicit_refusal_rate_on_benign,
    harmful_rate_by_condition,
    harmful_rate_by_condition_x_category,
    load_judgments_jsonl,
    plot_condition_bars,
    refusal_rate_on_benign,
    write_summary_csv,
)

HF_REPO_ID = "persona-shattering-lasr/monorepo"
HF_BASE = "evals/persona_jailbreak_wildjailbreak/llama-3.1-8b-instruct"
SCRATCH_BASE = project_root / "scratch" / "persona_jailbreak_eval" / "llama-3.1-8b-instruct"

# Source map: target condition name → (source run name, source filename).
# Filenames follow the existing ``judgments_<condition>.jsonl`` convention.
# When ``source_run == "wj_balanced_v2"`` the rows are subset to the canonical
# 500-id set; everywhere else the file is already 500 rows.
_ABL_RUN = "wj_ablations_v1_v2"


@dataclass(frozen=True)
class Source:
    target_condition: str
    source_run: str
    source_file: str  # relative to {source_run}/judgments/


SOURCES: list[Source] = [
    # Baselines — from balanced (subset to 500).
    Source("vanilla",            "wj_balanced_v2", "judgments_vanilla.jsonl"),
    Source("activation_capping", "wj_balanced_v2", "judgments_activation_capping.jsonl"),
    # 10 OCEAN LoRAs at scale 1.0 — from ablations (already 500).
    Source("lora_soup_o_plus_1.0",  _ABL_RUN, "judgments_lora_soup_o_plus_1.0.jsonl"),
    Source("lora_soup_o_minus_1.0", _ABL_RUN, "judgments_lora_soup_o_minus_1.0.jsonl"),
    Source("lora_soup_c_plus_1.0",  _ABL_RUN, "judgments_lora_soup_c_plus_1.0.jsonl"),
    Source("lora_soup_c_minus_1.0", _ABL_RUN, "judgments_lora_soup_c_minus_1.0.jsonl"),
    Source("lora_soup_e_plus_1.0",  _ABL_RUN, "judgments_lora_soup_e_plus_1.0.jsonl"),
    Source("lora_soup_e_minus_1.0", _ABL_RUN, "judgments_lora_soup_e_minus_1.0.jsonl"),
    Source("lora_soup_a_plus_1.0",  _ABL_RUN, "judgments_lora_soup_a_plus_1.0.jsonl"),
    Source("lora_soup_a_minus_1.0", _ABL_RUN, "judgments_lora_soup_a_minus_1.0.jsonl"),
    Source("lora_soup_n_plus_1.0",  _ABL_RUN, "judgments_lora_soup_n_plus_1.0.jsonl"),
    Source("lora_soup_n_minus_1.0", _ABL_RUN, "judgments_lora_soup_n_minus_1.0.jsonl"),
    # Controls — from ablations.
    Source("lora_soup_control_latest_1.0", _ABL_RUN, "judgments_lora_soup_control_latest_1.0.jsonl"),
    Source("lora_soup_control_legacy_1.0", _ABL_RUN, "judgments_lora_soup_control_legacy_1.0.jsonl"),
    # Combos — from individual combo runs.
    Source(
        "lora_soup_a_plus_0.5_c_plus_0.5",
        "wj_combo_a_plus_0p5_c_plus_0p5_v1",
        "judgments_lora_soup_a_plus_0.5_c_plus_0.5.jsonl",
    ),
    Source(
        "lora_soup_a_plus_1.0_c_plus_1.0",
        "wj_combo_a_plus_c_plus_v1",
        "judgments_lora_soup_a_plus_1.0_c_plus_1.0.jsonl",
    ),
    Source(
        "lora_soup_a_plus_1.0_c_plus_0.5",
        "wj_combo_a_plus_1p0_c_plus_0p5_v1",
        "judgments_lora_soup_a_plus_1.0_c_plus_0.5.jsonl",
    ),
    # Spare: c+0.5 o-0.5 soup from balanced (subset).
    Source(
        "lora_soup_c_plus_0.5_o_minus_0.5",
        "wj_balanced_v2",
        "judgments_lora_soup_c_plus_0.5_o_minus_0.5.jsonl",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hf_path(run: str, fname: str) -> str:
    return f"{HF_BASE}/{run}/judgments/{fname}"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _canonical_id_set() -> set[tuple[str, str]]:
    """The 500-id (sample_id, kind) tuples used by the ablation sweep."""
    local = hf_hub_download(
        HF_REPO_ID,
        repo_type="dataset",
        filename=_hf_path(_ABL_RUN, "judgments_lora_soup_a_plus_1.0.jsonl"),
    )
    rows = _read_jsonl(Path(local))
    return {(r["sample_id"], r["kind"]) for r in rows}


def _hydrate_source(src: Source, canonical: set[tuple[str, str]], judgments_out_dir: Path) -> Path:
    """Download src, set the row condition to ``src.target_condition``, filter
    to the canonical id set if necessary, write to ``judgments_out_dir``.

    Returns the local path of the written file.
    """
    local_in = Path(hf_hub_download(
        HF_REPO_ID, repo_type="dataset", filename=_hf_path(src.source_run, src.source_file),
    ))
    rows_in = _read_jsonl(local_in)

    rows_kept: list[dict] = []
    for r in rows_in:
        if (r["sample_id"], r["kind"]) not in canonical:
            continue
        # Force the condition tag to the target name. Some source files use a
        # condition string that drops the "_1.0" / "_0.5" suffix (e.g. older
        # combo runs); we want a single canonical set of condition strings in
        # the consolidated folder.
        r = dict(r)
        r["condition"] = src.target_condition
        rows_kept.append(r)

    if not rows_kept:
        raise RuntimeError(
            f"no rows kept for {src.target_condition!r} from "
            f"{src.source_run}/{src.source_file} after filtering to canonical set"
        )

    out_path = judgments_out_dir / f"judgments_{src.target_condition}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows_kept:
            f.write(json.dumps(r) + "\n")
    print(f"  ✓ {src.target_condition:42s} ← {src.source_run}/{src.source_file}  ({len(rows_kept)} rows)")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def collate(version: str, *, sources: list[Source] = SOURCES) -> Path:
    dest_run = f"wj_paper_{version}"
    dest_local = SCRATCH_BASE / dest_run
    judgments_dir = dest_local / "judgments"
    aggregate_dir = dest_local / "aggregate"
    if dest_local.exists():
        shutil.rmtree(dest_local)
    judgments_dir.mkdir(parents=True)
    aggregate_dir.mkdir(parents=True)

    print(f"[collate] establishing canonical id set from {_ABL_RUN}...")
    canonical = _canonical_id_set()
    print(f"[collate] canonical set has {len(canonical)} (sample_id, kind) tuples")

    print(f"[collate] hydrating {len(sources)} sources...")
    manifest: list[dict] = []
    for src in sources:
        out_path = _hydrate_source(src, canonical, judgments_dir)
        manifest.append({
            "target_condition": src.target_condition,
            "source_run": src.source_run,
            "source_file": src.source_file,
            "n_rows": sum(1 for _ in out_path.open()),
        })

    (dest_local / "manifest.json").write_text(json.dumps({
        "canonical_id_set_size": len(canonical),
        "canonical_source_run": _ABL_RUN,
        "canonical_source_file": "judgments_lora_soup_a_plus_1.0.jsonl",
        "sources": manifest,
    }, indent=2))

    print(f"\n[collate] aggregating into {aggregate_dir}...")
    records = []
    for path in sorted(judgments_dir.glob("judgments_*.jsonl")):
        records.extend(load_judgments_jsonl(path))
    print(f"[collate] loaded {len(records)} judgment rows total")

    write_summary_csv(harmful_rate_by_condition(records),         aggregate_dir / "harmful_rate_by_condition.csv")
    write_summary_csv(refusal_rate_on_benign(records),            aggregate_dir / "refusal_rate_on_benign.csv")
    write_summary_csv(explicit_refusal_rate_on_benign(records),   aggregate_dir / "explicit_refusal_rate_on_benign.csv")
    write_summary_csv(harmful_rate_by_condition_x_category(records),
                      aggregate_dir / "harmful_rate_by_condition_x_category.csv")

    plot_condition_bars(
        harmful_rate_by_condition(records),
        refusal_rate_on_benign(records),
        title=f"Persona-Jailbreak ({dest_run}) — diagnostic",
        output_path=aggregate_dir / "summary_bars.png",
    )

    print(f"\n[collate] wrote local: {dest_local}")
    return dest_local


def upload(local_dir: Path) -> None:
    import os
    # Some HF flows pick up HF_TOKEN from the env regardless of the
    # ``token=...`` passed to HfApi(). To make sure a write token is used,
    # also override the env var when HF_WRITE_TOKEN is set.
    token = os.getenv("HF_WRITE_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("No HF token found in env (HF_WRITE_TOKEN or HF_TOKEN)")
    if os.getenv("HF_WRITE_TOKEN"):
        os.environ["HF_TOKEN"] = token
    api = HfApi(token=token)
    dest_path = f"{HF_BASE}/{local_dir.name}"
    print(f"[upload] {local_dir} → {HF_REPO_ID}:{dest_path}")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        path_in_repo=dest_path,
        commit_message=f"Add consolidated WJ paper results: {local_dir.name}",
    )
    print(f"[upload] done")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default="v1", help="Folder version suffix (default: v1 → wj_paper_v1).")
    p.add_argument("--no-upload", action="store_true", help="Skip the HF upload step.")
    args = p.parse_args()

    local_dir = collate(args.version)
    if args.no_upload:
        print("\n[upload] skipped (--no-upload)")
        return
    upload(local_dir)


if __name__ == "__main__":
    main()
