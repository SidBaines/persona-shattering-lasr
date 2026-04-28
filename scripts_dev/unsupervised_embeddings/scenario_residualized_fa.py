"""Re-fit FA on the scenario-residualized response matrix.

The non-residualized FA in ``analysis_for_paper.py`` ends up with factor
scores whose variance is dominated (52–78%) by ``scenario_id``, with
``archetype`` contributing < 5%. This script subtracts per-scenario item
means from the response matrix before FA so the factor structure reflects
*persona × scenario interaction + persona main effects*, not raw
scenario-genre differences.

Outputs (per model under
``scratch/psychometric_fa_paper/<slug>/factor_analysis_resid/``):

  - ``fa_3_principal_oblimin.npz`` — fit npz (loadings, scores, etc.)
  - ``fa_3_principal_oblimin_top30.txt`` — top-loading items per factor
  - ``n_factors_suggest.json`` — n-factors criteria on the residualized
    correlation matrix
  - ``variance_decomp.json`` — η² for archetype + scenario on the new
    factor scores (sanity check that residualization worked)

Plus a cross-model summary at
``scratch/psychometric_fa_paper/scenario_residualized_summary.json``.

Run: ``python scripts_dev/unsupervised_embeddings/scenario_residualized_fa.py``
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

SEED = 436
random.seed(SEED)
np.random.seed(SEED)

from src_dev.factor_analysis import (
    run_factor_analysis,
    save_factor_analysis,
    suggest_n_factors,
)
from src_dev.factor_analysis.interpretation import prompt_effects
from src_dev.factor_analysis.preprocessing import residualize as resid_primitive

ROOT = Path("scratch/psychometric_fa_paper")
ROLLOUT_DIR = Path(
    "scratch/psychometric_fa/"
    "rollouts-llama318binstruct-t1.0-15t-2500p-seed436-scenarios_v2-uprompt_v6"
)
SCENARIOS_FILE = Path("datasets/scenarios/v2.json")

MODELS = [
    ("llama-3.1-8b", "Llama-3.1-8B-Instruct"),
    ("qwen2.5-7b", "Qwen2.5-7B-Instruct"),
]
K = 3
ROTATION = "oblimin"
METHOD = "principal"
TOP_N = 30
N_FACTORS_METHODS = (
    "parallel", "map", "ekc", "acceleration", "kaiser",
    "optimal_coordinates", "scree_elbow",
)
N_FACTORS_K_MAX = 25
N_FACTORS_PARALLEL_ITERS = 200


def build_sample_lookup() -> dict[str, dict]:
    arch = json.loads((ROLLOUT_DIR / "archetype_assignments.json").read_text())
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
            out[sid] = {"archetype": archetype, "scenario_id": scen_id}
    return out


def load_model_matrix(slug: str, lookup: dict[str, dict]):
    base = ROOT / slug / "hydrated/questionnaire"
    M_full = np.load(base / "response_matrix.npy").astype(float)
    items_full = json.loads((base / "items.json").read_text())
    meta = [json.loads(l) for l in (base / "metadata.jsonl").read_text().splitlines() if l.strip()]
    assert len(meta) == M_full.shape[0]
    # Annotate metadata.
    for row in meta:
        sid = row.get("sample_id")
        hit = lookup.get(sid)
        if hit:
            row["archetype"] = hit["archetype"]
            row["scenario_id"] = hit["scenario_id"]
    # Drop rows missing archetype/scenario or with NaNs.
    keep_meta = np.array([
        ("archetype" in r and "scenario_id" in r)
        for r in meta
    ])
    nan_rows = np.isnan(M_full).any(axis=1)
    keep_meta = keep_meta & ~nan_rows
    M = M_full[keep_meta]
    meta = [r for r, k in zip(meta, keep_meta) if k]
    # Apply the same per-block relative variance floor used in the main run
    # (MIN_ITEM_VARIANCE = 0.2 there; here we replicate by computing block
    # median and dropping low-variance items). Since the questionnaire is
    # all one block (likert), this collapses to: keep cols whose var ≥
    # 0.2 × median var.
    var = np.nanvar(M, axis=0)
    med = float(np.median(var[var > 0]))
    keep_col = var >= 0.2 * med
    M = M[:, keep_col]
    items = [it for it, k in zip(items_full, keep_col) if k]
    return M, items, meta


def write_top_items(
    fa: dict, items: list[dict], out_path: Path, *, model_label: str, k: int, top_n: int
) -> None:
    loadings = fa["loadings"]
    prop = fa["proportion_variance"]
    cum = fa["cumulative_variance"]
    lines = [
        f"FA fit (scenario-residualized) — {model_label}",
        f"rotation={ROTATION}  n_factors={k}  n_items={loadings.shape[0]}",
        "",
        "Variance explained per factor (ss_loadings fraction):",
    ]
    for fi in range(k):
        lines.append(f"  F{fi+1}:  prop={prop[fi]:.3f}   cum={cum[fi]:.3f}")
    lines.append("")
    for fi in range(k):
        lines.append("=" * 78)
        lines.append(f"F{fi+1}  (top-{top_n} by signed loading)")
        lines.append("=" * 78)
        order = np.argsort(loadings[:, fi])
        neg = order[:top_n]
        pos = order[::-1][:top_n]
        lines.append(f"  Positive loadings ({top_n} items):")
        for idx in pos:
            it = items[idx]
            rev = " (REVERSED)" if it.get("reverse_keyed") else ""
            lines.append(f"    {loadings[idx, fi]:+.3f}  <{it['block']} enc={it.get('encoding','')}>{rev}  {it['text']}")
        lines.append("")
        lines.append(f"  Negative loadings ({top_n} items):")
        for idx in neg:
            it = items[idx]
            rev = " (REVERSED)" if it.get("reverse_keyed") else ""
            lines.append(f"    {loadings[idx, fi]:+.3f}  <{it['block']} enc={it.get('encoding','')}>{rev}  {it['text']}")
        lines.append("")
    fcm = fa.get("factor_correlation_matrix")
    if fcm is not None:
        lines.append("=" * 78)
        lines.append("Factor correlation matrix (oblique rotation)")
        lines.append("=" * 78)
        for i in range(k):
            row = "  ".join(f"{fcm[i, j]:+.3f}" for j in range(k))
            lines.append(f"  F{i+1}   {row}")
    out_path.write_text("\n".join(lines))


def main() -> None:
    lookup = build_sample_lookup()
    print(f"sample lookup: {len(lookup)} ids")

    summary: dict[str, dict] = {}

    for slug, label in MODELS:
        print(f"\n=== {label} ===")
        M, items, meta = load_model_matrix(slug, lookup)
        print(f"  matrix: {M.shape}, items: {len(items)}")

        # Residualize by scenario_id.
        M_resid, group_means, group_inv = resid_primitive(
            M, meta, group_field="scenario_id"
        )
        # Drop columns that became near-zero variance after residualization.
        var_post = np.nanvar(M_resid, axis=0)
        keep_col = var_post > 1e-9
        if not keep_col.all():
            print(f"  dropping {(~keep_col).sum()} columns with ~0 var post-residualization")
        M_resid = M_resid[:, keep_col]
        items_post = [it for it, k in zip(items, keep_col) if k]
        print(f"  residualized matrix: {M_resid.shape}")

        out_dir = ROOT / slug / "factor_analysis_resid"
        out_dir.mkdir(parents=True, exist_ok=True)

        # n-factors suggest on residualized matrix.
        nf = suggest_n_factors(
            M_resid,
            methods=N_FACTORS_METHODS,
            k_max=N_FACTORS_K_MAX,
            parallel_n_iterations=N_FACTORS_PARALLEL_ITERS,
            seed=SEED,
            verbose=False,
        )
        # Some helpers return a dict, some a result object — normalise.
        nf_dict = nf if isinstance(nf, dict) else getattr(nf, "summary", None) or vars(nf)
        # Strip non-JSON-serialisable internals.
        nf_clean = {k: (v if isinstance(v, (int, float, str, list, dict)) else None) for k, v in nf_dict.items()}
        (out_dir / "n_factors_suggest.json").write_text(json.dumps(nf_clean, indent=2, default=str))
        print(f"  n-factors suggestions:")
        for k_, v_ in nf_clean.items():
            if isinstance(v_, (int, float)) and k_ != "n_personas" and k_ != "n_items":
                print(f"    {k_}: {v_}")

        # Fit FA at k=3 and save.
        print(f"  fitting FA k={K} rotation={ROTATION}…")
        fa = run_factor_analysis(M_resid, n_factors=K, method=METHOD, rotation=ROTATION)
        base = out_dir / f"fa_{K}_{METHOD}_{ROTATION}"
        save_factor_analysis(
            fa,
            base,
            config={
                "n_factors": K, "method": METHOD, "rotation": ROTATION,
                "residualized_by": "scenario_id", "n_samples": int(M_resid.shape[0]),
                "n_cols": int(M_resid.shape[1]), "model_slug": slug, "model_label": label,
            },
        )
        write_top_items(fa, items_post, Path(str(base) + f"_top{TOP_N}.txt"),
                        model_label=label, k=K, top_n=TOP_N)
        print(f"  variance explained (cumulative): {fa['cumulative_variance'].tolist()}")

        # Variance decomposition on new factor scores.
        scores = fa["scores"]
        eta_arch = prompt_effects(scores, meta, group_field="archetype").tolist()
        eta_scen = prompt_effects(scores, meta, group_field="scenario_id").tolist()
        decomp = {
            "model": label,
            "n_personas": int(M_resid.shape[0]),
            "n_items": int(M_resid.shape[1]),
            "proportion_variance": fa["proportion_variance"].tolist(),
            "cumulative_variance": fa["cumulative_variance"].tolist(),
            "eta2_archetype_per_factor": eta_arch,
            "eta2_scenario_per_factor": eta_scen,
        }
        (out_dir / "variance_decomp.json").write_text(json.dumps(decomp, indent=2))
        print(f"  η² archetype/scenario per factor:")
        for i, (ea, es) in enumerate(zip(eta_arch, eta_scen)):
            print(f"    F{i}:  archetype={ea:.3f}   scenario={es:.3f}")
        summary[slug] = decomp

    (ROOT / "scenario_residualized_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {ROOT / 'scenario_residualized_summary.json'}")


if __name__ == "__main__":
    main()
