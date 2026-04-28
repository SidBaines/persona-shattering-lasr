"""How much factor-score variance is explained by archetype / scenario seed?

For each model in the paper analysis, this loads:
  - factor scores (n_personas × k) from the FA .npz
  - per-persona archetype + scenario_id (from the rollout's
    canonical_samples.jsonl, using the existing regex helper)

and reports:
  1. One-way η² per factor for ``archetype`` and for ``scenario_id`` —
     the share of factor-score variance that group membership accounts for.
  2. Two-way ANOVA decomposition (archetype, scenario, interaction,
     residual) so we can see whether the two grouping factors are
     additive or whether their interaction matters.
  3. Per-archetype mean factor scores (sorted) so we can see which
     archetypes pull each factor in which direction.

Outputs land in ``scratch/psychometric_fa_paper/<slug>/variance_decomp/``:

  - ``eta2_oneway.json``           — η² per factor for archetype, scenario
  - ``eta2_twoway.json``           — partition of total variance per factor
  - ``per_archetype_factor_means.csv``
  - ``factor_means_by_archetype.png`` (heatmap)

Run: ``python scripts_dev/unsupervised_embeddings/factor_variance_decomp.py``
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

SEED = 436
random.seed(SEED)
np.random.seed(SEED)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src_dev.factor_analysis.interpretation import prompt_effects

ROOT = Path("scratch/psychometric_fa_paper")
ROLLOUT_DIR = Path(
    "scratch/psychometric_fa/"
    "rollouts-llama318binstruct-t1.0-15t-2500p-seed436-scenarios_v2-uprompt_v6"
)
SCENARIOS_FILE = Path("datasets/scenarios/v2.json")


def build_sample_lookup() -> dict[str, dict]:
    """Build sample_id -> {archetype, scenario_id} for the 2500p rollout.

    Uses canonical_samples.jsonl for sample_id -> row_index, then
    archetype_assignments.json (keyed by row_index 0..2499) for archetype,
    and matches the system prompt against scenarios v2.json for scenario_id.
    """
    arch = json.loads(
        (ROLLOUT_DIR / "archetype_assignments.json").read_text()
    )
    scen_data = json.loads(SCENARIOS_FILE.read_text())
    prompt_to_scenario = {sc["target_system_prompt"]: sc["id"] for sc in scen_data["scenarios"]}
    out: dict[str, dict] = {}
    with (ROLLOUT_DIR / "datasets/canonical_samples.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec["sample_id"]
            row_idx = rec["source_info"]["row_index"]
            sys_prompt = rec["input"]["messages"][0]["content"]
            scen_id = prompt_to_scenario.get(sys_prompt)
            archetype = arch.get(str(row_idx))
            if scen_id is None or archetype is None:
                continue
            out[sid] = {
                "archetype": archetype,
                "scenario_id": scen_id,
                "row_index": row_idx,
            }
    return out

MODELS = [
    ("llama-3.1-8b", "Llama-3.1-8B-Instruct"),
    ("qwen2.5-7b", "Qwen2.5-7B-Instruct"),
]
FA_FILE = "factor_analysis/raw/fa_3_principal_oblimin.npz"


def two_way_eta2(
    scores: np.ndarray, a: np.ndarray, b: np.ndarray
) -> dict:
    """Type-II-style two-way variance decomposition without statsmodels.

    Returns η² for factor A (archetype), B (scenario), and AB interaction,
    plus residual share. Uses the unbalanced-design ANOVA partition based
    on cell, marginal, and grand means; this is the standard method when
    cell counts differ. With ~2500 personas split across ~9 archetypes ×
    many scenarios, cells will be small but the partition is still
    informative as a variance-share decomposition.
    """
    out: dict[str, list[float]] = {
        "eta2_archetype": [],
        "eta2_scenario": [],
        "eta2_interaction": [],
        "eta2_residual": [],
        "n_cells": [],
        "n_arch": [],
        "n_scen": [],
        "n_obs": [],
    }
    arch_levels = np.unique(a)
    scen_levels = np.unique(b)
    for f in range(scores.shape[1]):
        y = scores[:, f]
        grand = y.mean()
        ss_total = ((y - grand) ** 2).sum()
        # Marginal means
        ss_a = 0.0
        for la in arch_levels:
            m = a == la
            if m.sum() == 0:
                continue
            ss_a += m.sum() * (y[m].mean() - grand) ** 2
        ss_b = 0.0
        for lb in scen_levels:
            m = b == lb
            if m.sum() == 0:
                continue
            ss_b += m.sum() * (y[m].mean() - grand) ** 2
        # Cell means → SS for combined (a, b)
        ss_cells = 0.0
        n_cells = 0
        for la in arch_levels:
            for lb in scen_levels:
                m = (a == la) & (b == lb)
                if m.sum() == 0:
                    continue
                n_cells += 1
                ss_cells += m.sum() * (y[m].mean() - grand) ** 2
        ss_inter = max(0.0, ss_cells - ss_a - ss_b)
        ss_resid = max(0.0, ss_total - ss_cells)
        out["eta2_archetype"].append(float(ss_a / ss_total) if ss_total > 0 else 0.0)
        out["eta2_scenario"].append(float(ss_b / ss_total) if ss_total > 0 else 0.0)
        out["eta2_interaction"].append(float(ss_inter / ss_total) if ss_total > 0 else 0.0)
        out["eta2_residual"].append(float(ss_resid / ss_total) if ss_total > 0 else 0.0)
        out["n_cells"].append(int(n_cells))
        out["n_arch"].append(int(len(arch_levels)))
        out["n_scen"].append(int(len(scen_levels)))
        out["n_obs"].append(int(len(y)))
    return out


def per_archetype_means(
    scores: np.ndarray, archetypes: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray]:
    levels = sorted(set(archetypes))
    means = np.zeros((len(levels), scores.shape[1]))
    counts = np.zeros(len(levels), dtype=int)
    for i, la in enumerate(levels):
        m = archetypes == la
        means[i] = scores[m].mean(axis=0)
        counts[i] = int(m.sum())
    return levels, means, counts


def plot_heatmap(
    levels: list[str],
    means: np.ndarray,
    counts: np.ndarray,
    out_png: Path,
    title: str,
) -> None:
    order = np.argsort(means[:, 0])  # sort by F0 mean for readability
    M = means[order]
    L = [f"{levels[i]}  (n={counts[i]})" for i in order]
    fig, ax = plt.subplots(figsize=(0.8 * M.shape[1] + 3, 0.3 * M.shape[0] + 1))
    vmax = float(np.abs(M).max())
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(L)))
    ax.set_yticklabels(L)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels([f"F{i}" for i in range(M.shape[1])])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(M[i, j]) < 0.5 * vmax else "white")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    lookup = build_sample_lookup()
    if not lookup:
        raise SystemExit(
            f"No archetype/scenario lookup found in {ROLLOUT_DIR}. "
            "Did the canonical_samples.jsonl download succeed?"
        )
    print(f"loaded archetype lookup: {len(lookup)} sample_ids")

    overall_summary = {}

    for slug, label in MODELS:
        print(f"\n=== {label} ===")
        # Factor scores aligned with metadata.jsonl row order.
        npz = np.load(ROOT / slug / FA_FILE, allow_pickle=True)
        scores = npz["scores"]
        meta = [
            json.loads(l)
            for l in (ROOT / slug / "hydrated/questionnaire/metadata.jsonl").read_text().splitlines()
            if l.strip()
        ]
        assert len(meta) == scores.shape[0], f"{len(meta)} vs {scores.shape[0]}"
        for row in meta:
            sid = row.get("sample_id")
            hit = lookup.get(sid)
            if hit is not None:
                row["archetype"] = hit["archetype"]
                row["scenario_id"] = hit["scenario_id"]

        # Drop rows that didn't resolve.
        keep = np.array([("archetype" in r and "scenario_id" in r) for r in meta])
        if keep.sum() < len(meta):
            print(f"  dropping {len(meta) - keep.sum()} unresolved rows")
        scores_k = scores[keep]
        meta_k = [r for r, k in zip(meta, keep) if k]

        # One-way eta-squared.
        eta_arch = prompt_effects(scores_k, meta_k, group_field="archetype").tolist()
        eta_scen = prompt_effects(scores_k, meta_k, group_field="scenario_id").tolist()

        a = np.array([r["archetype"] for r in meta_k])
        b = np.array([r["scenario_id"] for r in meta_k])
        twoway = two_way_eta2(scores_k, a, b)

        # Per-archetype means.
        levels, means, counts = per_archetype_means(scores_k, a)

        # Save outputs.
        out_dir = ROOT / slug / "variance_decomp"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eta2_oneway.json").write_text(
            json.dumps(
                {
                    "model": label,
                    "n_factors": scores.shape[1],
                    "n_personas_resolved": int(keep.sum()),
                    "n_archetypes": len(set(a)),
                    "n_scenarios": len(set(b)),
                    "eta2_archetype_per_factor": eta_arch,
                    "eta2_scenario_per_factor": eta_scen,
                },
                indent=2,
            )
        )
        (out_dir / "eta2_twoway.json").write_text(
            json.dumps({"model": label, **twoway}, indent=2)
        )
        # CSV of per-archetype means.
        with (out_dir / "per_archetype_factor_means.csv").open("w") as fh:
            fh.write("archetype,n," + ",".join(f"F{i}_mean" for i in range(scores.shape[1])) + "\n")
            for la, n_, row in zip(levels, counts, means):
                fh.write(f"{la},{n_}," + ",".join(f"{v:.4f}" for v in row) + "\n")
        # Heatmap.
        plot_heatmap(
            levels, means, counts,
            out_dir / "factor_means_by_archetype.png",
            f"{label}: mean factor score per archetype (sorted by F0)",
        )

        # Console summary.
        print(f"  one-way η² per factor:")
        for i, (ea, es) in enumerate(zip(eta_arch, eta_scen)):
            print(f"    F{i}:  archetype={ea:.3f}   scenario={es:.3f}")
        print(f"  two-way decomp per factor:")
        for i in range(len(eta_arch)):
            print(
                f"    F{i}:  arch={twoway['eta2_archetype'][i]:.3f}  "
                f"scen={twoway['eta2_scenario'][i]:.3f}  "
                f"interact={twoway['eta2_interaction'][i]:.3f}  "
                f"resid={twoway['eta2_residual'][i]:.3f}  "
                f"(n_cells={twoway['n_cells'][i]})"
            )
        overall_summary[slug] = {
            "label": label,
            "n_archetypes": len(set(a)),
            "n_scenarios": len(set(b)),
            "eta2_archetype_oneway": eta_arch,
            "eta2_scenario_oneway": eta_scen,
            "two_way": twoway,
        }

    (ROOT / "variance_decomp_summary.json").write_text(
        json.dumps(overall_summary, indent=2)
    )
    print(f"\nwrote {ROOT / 'variance_decomp_summary.json'}")


if __name__ == "__main__":
    main()
