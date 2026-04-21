"""End-to-end analysis driver for external-rollout psychometric studies.

One script, one command, everything:

    1. For each (external-rollout preset, questionnaire preset) pair in the
       config: cache-aware Stage-1 ingest + Stage-2 admin. Hits local/HF
       cache when available, generates when not.
    2. Combines per-pair outputs into one multi-preset response matrix.
    3. Fits three flavours of factor analysis:
        - ``baseline``        — raw combined matrix (model identity often
                                dominates the top factor).
        - ``residualised``    — subtract per-``rollout_preset_key`` means
                                before FA (strips between-model variance).
        - ``per_preset``      — run FA separately on each preset's row
                                subset (same k, same rotation).
    4. Computes:
        - Preset-variance η²  — how much of each factor's variance comes
                                from the preset/model split. Run on both
                                ``baseline`` and ``residualised`` solutions.
        - Tucker's φ          — factor-structure similarity across every
                                pair of per-preset FA solutions, with
                                greedy one-to-one factor alignment.
    5. Writes CSVs + JSON summaries + diagnostic plots under
       ``scratch/psychometric_fa/external_analysis/<OUTPUT_TAG>/``.

The orchestrator (``psychometric_rollout_fa.py``) is untouched — this
script imports its ``EXTERNAL_ROLLOUT_PRESETS`` / ``QUESTIONNAIRE_PRESETS``
dicts as the single source of truth for preset definitions.

Run with::

    uv run python -m scripts_dev.psychometric_assessment.external_rollout_analysis
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from src_dev.factor_analysis.factor_analysis import run_factor_analysis
from src_dev.factor_analysis.interpretation import prompt_effects
from src_dev.psychometric.combine import combine_per_pair_outputs, load_pair_outputs
from src_dev.psychometric.config import (
    ExternalRolloutsStageConfig,
    QuestionnaireStageConfig,
    RunContext,
)
from src_dev.psychometric.preprocessing import preprocess_response_matrix
from src_dev.psychometric.questionnaire_io import load_questionnaire
from src_dev.psychometric.rollout_stats import (
    compute_rollout_stats,
    summarise_stats,
)
from src_dev.psychometric.stages import (
    run_stage_ingest_external_rollouts,
    run_stage_questionnaire,
)
from src_dev.psychometric.tucker_congruence import (
    align_factors,
    classify_phi,
    summarise_alignment,
    tucker_phi_matrix,
)

# Single source of truth for preset definitions — imported from the
# orchestrator so adding/editing a preset there automatically flows here.
from scripts_dev.unsupervised_embeddings.psychometric_rollout_fa import (
    EXTERNAL_ROLLOUT_PRESETS,
    QUESTIONNAIRE_PRESETS,
    _rollout_run_id,
    _questionnaire_run_id,
)

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG — edit in place to change the run
# ═════════════════════════════════════════════════════════════════════════════

SEED = 436
random.seed(SEED)
np.random.seed(SEED)

SCRATCH_ROOT = Path("scratch/psychometric_fa")
HF_REPO_ID = "persona-shattering-lasr/psychometric-fa-runs"

# ── Which presets to run ────────────────────────────────────────────────────
# Every rollout must have entries for every questionnaire (orchestrator's
# combine step enforces this), so we use a Cartesian product here.
PRESETS: list[str] = [
    # PRISM (short, values-dialogue; ~3-turn p50)
    "prism_zephyr_7b_beta",
    "prism_mistral_7b_v01",
    "prism_llama2_7b_chat",
    "prism_llama2_13b_chat",
    "prism_falcon_7b_instruct",
    "prism_oasst_pythia_12b",
    # LMSYS (real-user chat, turn≥5 English subset)
    "lmsys_koala_13b_t5",
    # "lmsys_mpt_7b_chat_t5",   # mosaicml/mpt-7b-chat HF repo unavailable; find a mirror before enabling
    # Kwai-Klear (long coding-agent rollouts; ~24-turn p50, ~13k tok p50)
    "kwai_swe",
]
QUESTIONNAIRES: list[str] = ["v5", "trait_ocean_v1_nolead"]

# ── Pipeline control ────────────────────────────────────────────────────────
# Set True to stop after the rollout-stats stage (ingest + heuristics + plots
# for every preset). Useful for sanity-checking rollout properties before
# committing to questionnaire admin + FA compute. Stage 1 ingest is
# cache-aware so this stage costs ~nothing on re-runs.
STOP_AFTER_ROLLOUT_STATS: bool = True

# ── FA knobs ────────────────────────────────────────────────────────────────
FA_METHOD = "principal"
FA_N_FACTORS = 7
FA_ROTATIONS: list[str] = ["oblimin", "varimax"]
MIN_ITEM_VARIANCE = 0.1
HIGH_VARIANCE_PERSONA_DROP_PCT = 0.0

# ── Stage-2 defaults (mirror orchestrator values; most inherit preset-level
# ── overrides via the preset registry) ──────────────────────────────────────
QUESTIONNAIRE_PROVIDER = "vllm"
QUESTIONNAIRE_PHRASING = "aside"
LIKERT_SCALE = 5
QUESTIONNAIRE_MAX_CONCURRENT = 32
QUESTIONNAIRE_MAX_NEW_TOKENS = 32
QUESTIONNAIRE_TIMEOUT = 60
MAX_PARSE_RETRIES = 3
QUESTIONNAIRE_VLLM_PERSONAS_PER_BATCH = 8
QUESTIONNAIRE_VLLM_GPU_MEMORY_UTILIZATION = 0.95
QUESTIONNAIRE_VLLM_TENSOR_PARALLEL_SIZE = 1
QUESTIONNAIRE_TOP_LOGPROBS = 20
QUESTIONNAIRE_LOGPROB_TEMPERATURE = 1.0
QUESTIONNAIRE_DYNAMIC_MASS_FILTER = True
QUESTIONNAIRE_MIN_CHOICE_MASS = 0.0
QUESTIONNAIRE_MIN_TRAIT_COVERAGE = 0.25
QUESTIONNAIRE_CONTEXT_BUFFER_TOKENS = 1024
QUESTIONNAIRE_RESET_MODE = "none"
FC_PAIR_SIGN_ALIGNMENT = True

# ── Output ──────────────────────────────────────────────────────────────────
OUTPUT_TAG = "multisource_n500_9models"  # change when config changes


# ═════════════════════════════════════════════════════════════════════════════
# Helpers — Stage 1 + Stage 2 for a single (preset, questionnaire) pair
# ═════════════════════════════════════════════════════════════════════════════


def _rollout_dir(rollout_key: str) -> Path:
    return SCRATCH_ROOT / _rollout_run_id(rollout_key)


def _questionnaire_dir(rollout_key: str, q_key: str) -> Path:
    return SCRATCH_ROOT / _questionnaire_run_id(rollout_key, q_key)


def _build_ctx(rollout_key: str, q_key: str) -> RunContext:
    return RunContext(
        scratch_root=SCRATCH_ROOT,
        hf_repo_id=HF_REPO_ID,
        rollout_run_id=_rollout_run_id(rollout_key),
        questionnaire_run_id=_questionnaire_run_id(rollout_key, q_key),
        rollout_dir=_rollout_dir(rollout_key),
        questionnaire_dir=_questionnaire_dir(rollout_key, q_key),
        effective_questionnaire_dir=_questionnaire_dir(rollout_key, q_key),
        is_multi_preset=True,
        provenance={
            "rollout_preset_key": rollout_key,
            "questionnaire_preset_key": q_key,
        },
    )


def _ingest_preset(ctx: RunContext, rollout_key: str) -> None:
    p = EXTERNAL_ROLLOUT_PRESETS[rollout_key]
    cfg = ExternalRolloutsStageConfig(
        ctx=ctx,
        source=p.source,
        assistant_model=p.assistant_model,
        assistant_provider=p.assistant_provider,
        max_samples=p.max_samples,
        seed=p.seed,
        max_scan=p.max_scan,
        filter_config=dict(p.filter_config),
        min_assistant_turns=p.min_assistant_turns,
    )
    run_stage_ingest_external_rollouts(cfg)


def _administer_questionnaire(
    ctx: RunContext, rollout_key: str, q_key: str
) -> None:
    r = EXTERNAL_ROLLOUT_PRESETS[rollout_key]
    q = QUESTIONNAIRE_PRESETS[q_key]
    cfg = QuestionnaireStageConfig(
        ctx=ctx,
        questionnaire_path=Path(q.path),
        questionnaire_version=q.version,
        fa_blocks=tuple(q.fa_blocks),
        use_logprobs=q.use_logprobs,
        phrasing=QUESTIONNAIRE_PHRASING,
        likert_scale=LIKERT_SCALE,
        provider=QUESTIONNAIRE_PROVIDER,
        model=r.assistant_model,
        max_new_tokens=QUESTIONNAIRE_MAX_NEW_TOKENS,
        max_concurrent=QUESTIONNAIRE_MAX_CONCURRENT,
        timeout=QUESTIONNAIRE_TIMEOUT,
        max_parse_retries=MAX_PARSE_RETRIES,
        vllm_personas_per_batch=QUESTIONNAIRE_VLLM_PERSONAS_PER_BATCH,
        vllm_gpu_memory_utilization=QUESTIONNAIRE_VLLM_GPU_MEMORY_UTILIZATION,
        vllm_tensor_parallel_size=QUESTIONNAIRE_VLLM_TENSOR_PARALLEL_SIZE,
        top_logprobs=QUESTIONNAIRE_TOP_LOGPROBS,
        logprob_temperature=QUESTIONNAIRE_LOGPROB_TEMPERATURE,
        dynamic_mass_filter=QUESTIONNAIRE_DYNAMIC_MASS_FILTER,
        min_choice_mass=QUESTIONNAIRE_MIN_CHOICE_MASS,
        min_trait_coverage=QUESTIONNAIRE_MIN_TRAIT_COVERAGE,
        reset_mode=QUESTIONNAIRE_RESET_MODE,
        max_context_tokens=r.max_context_tokens,
        context_buffer_tokens=QUESTIONNAIRE_CONTEXT_BUFFER_TOKENS,
        write_inspection_file=False,
    )
    run_stage_questionnaire(
        cfg,
        num_conversation_turns=r.min_assistant_turns,
        openrouter_provider_routing=None,
        fc_pair_sign_alignment=FC_PAIR_SIGN_ALIGNMENT,
    )


def _ingest_all_presets() -> dict[str, Path]:
    """Run Stage 1 ingest for each preset; return the canonical rollout dir."""
    rollout_dirs: dict[str, Path] = {}
    for r_key in PRESETS:
        # Use the first questionnaire just to construct a RunContext —
        # Stage 1 only reads the rollout fields of the context.
        ctx = _build_ctx(r_key, QUESTIONNAIRES[0])
        print(f"\n{'#' * 60}")
        print(f"# Stage 1 (ingest): preset={r_key!r}")
        print(f"#   {ctx.rollout_run_id}")
        print(f"{'#' * 60}")
        _ingest_preset(ctx, r_key)
        rollout_dirs[r_key] = ctx.rollout_dir
    return rollout_dirs


def _administer_all_pairs(
    rollout_dirs: dict[str, Path],
) -> dict[tuple[str, str], tuple[np.ndarray, list[dict], list[dict]]]:
    """Run (or hydrate) Stage 2 for every (preset, questionnaire) pair."""
    pair_data: dict[tuple[str, str], tuple[np.ndarray, list[dict], list[dict]]] = {}
    for r_key in PRESETS:
        for q_key in QUESTIONNAIRES:
            ctx = _build_ctx(r_key, q_key)
            print(f"\n{'#' * 60}")
            print(f"# Stage 2: rollout={r_key!r} × questionnaire={q_key!r}")
            print(f"#   {ctx.questionnaire_run_id}")
            print(f"{'#' * 60}")
            _administer_questionnaire(ctx, r_key, q_key)
            pair_data[(r_key, q_key)] = load_pair_outputs(ctx.questionnaire_dir)
    return pair_data


# ═════════════════════════════════════════════════════════════════════════════
# Rollout-stats stage
# ═════════════════════════════════════════════════════════════════════════════


def _run_rollout_stats(
    rollout_dirs: dict[str, Path],
    out_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """Compute per-sample stats for each preset; write CSVs + plots.

    Returns (per-sample rows, per-preset summary rows).
    """
    print(f"\n{'=' * 60}")
    print("[Rollout stats] Computing per-preset heuristics")
    print(f"{'=' * 60}")

    all_rows: list[dict] = []
    for r_key, rollout_dir in rollout_dirs.items():
        preset = EXTERNAL_ROLLOUT_PRESETS[r_key]
        print(
            f"\n[Rollout stats] preset={r_key!r}  model={preset.assistant_model!r}"
        )
        rows = compute_rollout_stats(
            rollout_dir,
            preset_key=r_key,
            assistant_model=preset.assistant_model,
        )
        print(
            f"  n={len(rows)}  "
            f"median n_assistant_turns={int(np.median([r['n_assistant_turns'] for r in rows]))}"
            f"  median n_tokens="
            + (
                f"{int(np.nanmedian([r['n_tokens'] for r in rows]))}"
                if any(not np.isnan(r['n_tokens']) for r in rows)
                else "n/a"
            )
        )
        all_rows.extend(rows)

    summary_rows = [
        summarise_stats(all_rows, preset_key=r) for r in rollout_dirs.keys()
    ]

    _write_csv(out_dir / "rollout_stats_per_sample.csv", all_rows)
    _write_csv(out_dir / "rollout_stats_per_preset.csv", summary_rows)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_rollout_stats(all_rows, plots_dir / "rollout_stats.png")

    # Compact summary printout.
    print("\n  preset                              n   turns_p50  turns_p90  tokens_p50  tokens_p90")
    for s in summary_rows:
        print(
            f"  {s['preset_key']:<35s}  {s['n_rollouts']:>3d}  "
            f"{s['n_assistant_turns_median']:>9.0f}  "
            f"{s['n_assistant_turns_p90']:>9.0f}  "
            f"{s['n_tokens_median']:>10.0f}  "
            f"{s['n_tokens_p90']:>10.0f}"
        )
    return all_rows, summary_rows


def _plot_rollout_stats(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    presets = sorted({r["preset_key"] for r in rows})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(presets), 3)))
    turns_by_preset = {
        p: np.array([r["n_assistant_turns"] for r in rows if r["preset_key"] == p])
        for p in presets
    }
    tokens_by_preset = {
        p: np.array(
            [r["n_tokens"] for r in rows if r["preset_key"] == p and not np.isnan(r["n_tokens"])]
        )
        for p in presets
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (A) turn-count histogram (log-x)
    ax = axes[0, 0]
    bins = np.logspace(0, 3, 30)
    for (p, vals), c in zip(turns_by_preset.items(), colors):
        if len(vals):
            ax.hist(vals.clip(1, None), bins=bins, alpha=0.5,
                    label=f"{p} (n={len(vals)})", color=c, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Assistant turns per conversation")
    ax.set_ylabel("Conversations")
    ax.set_title("(A) Assistant turn-count distribution")
    ax.legend(fontsize=7, loc="upper right")

    # (B) token-count histogram (log-x) with context budget lines
    ax = axes[0, 1]
    nonzero = [v for v in tokens_by_preset.values() if len(v)]
    if nonzero:
        all_tokens = np.concatenate(nonzero)
        lo = max(all_tokens.min(), 1)
        hi = max(all_tokens.max() * 2, 1000)
        bins = np.logspace(np.log10(lo), np.log10(hi), 30)
        for (p, vals), c in zip(tokens_by_preset.items(), colors):
            if len(vals):
                ax.hist(vals, bins=bins, alpha=0.5, label=p, color=c, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Total tokens per conversation")
    ax.set_ylabel("Conversations")
    ax.set_title("(B) Token-count distribution")
    for budget, label in [(4096, "4k"), (32768, "32k"), (131072, "128k")]:
        ax.axvline(budget, color="black", lw=0.6, ls=":")
    ax.legend(fontsize=7, loc="upper right")

    # (C) token CDF with budget markers
    ax = axes[1, 0]
    for (p, vals), c in zip(tokens_by_preset.items(), colors):
        if len(vals):
            s = np.sort(vals)
            y = np.arange(1, len(s) + 1) / len(s)
            ax.plot(s, y, label=p, color=c, lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Total tokens per conversation")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("(C) Token-count CDF")
    for budget in [4096, 8192, 32768, 65536, 131072]:
        ax.axvline(budget, color="gray", lw=0.5, ls=":")
        ax.text(budget, 0.02, f"{budget // 1024}k",
                rotation=90, fontsize=8, color="gray", va="bottom")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)

    # (D) turns vs tokens scatter
    ax = axes[1, 1]
    for (p, turns), c in zip(turns_by_preset.items(), colors):
        tokens = tokens_by_preset[p]
        # Align: turns array is all samples of preset p; tokens was filtered for
        # NaNs. Recompute over the same sample set so lengths match.
        paired_turns = [
            r["n_assistant_turns"] for r in rows
            if r["preset_key"] == p and not np.isnan(r["n_tokens"])
        ]
        if len(paired_turns) and len(tokens):
            ax.scatter(paired_turns, tokens, s=10, alpha=0.5, color=c, label=p)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Assistant turns")
    ax.set_ylabel("Total tokens")
    ax.set_title("(D) Turns vs tokens")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)

    fig.suptitle("Rollout-set heuristics — per preset")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[Plot] Wrote {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Helpers — FA flavours
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class FaFitResult:
    label: str                          # "baseline", "residualised", or a preset key
    rotation: str
    loadings: np.ndarray                # (n_items_kept, k)
    scores: np.ndarray                  # (n_samples_kept, k)
    proportion_variance: np.ndarray     # (k,)
    metadata: list[dict]                # aligned with scores
    items: list[dict]                   # aligned with loadings


def _fit_fa(
    matrix: np.ndarray,
    metadata: list[dict],
    items: list[dict],
    *,
    label: str,
    rotation: str,
    do_residualize: bool = False,
    residualize_group_field: str | None = None,
) -> FaFitResult:
    data, meta_filtered, items_filtered, _group_ids = preprocess_response_matrix(
        matrix, metadata, items,
        min_item_variance=MIN_ITEM_VARIANCE,
        high_variance_persona_drop_pct=HIGH_VARIANCE_PERSONA_DROP_PCT,
        do_residualize=do_residualize,
        residualize_group_field=residualize_group_field,
    )
    fa = run_factor_analysis(
        data, n_factors=FA_N_FACTORS, method=FA_METHOD, rotation=rotation,
    )
    return FaFitResult(
        label=label,
        rotation=rotation,
        loadings=fa["loadings"],
        scores=fa["scores"],
        proportion_variance=fa["proportion_variance"],
        metadata=meta_filtered,
        items=items_filtered,
    )


def _fit_per_preset_fas(
    matrix: np.ndarray,
    metadata: list[dict],
    items: list[dict],
    *,
    rotation: str,
) -> dict[str, FaFitResult]:
    """One FA per preset's row subset, sharing the combined item order."""
    preset_keys = sorted({m.get("rollout_preset_key") for m in metadata if m.get("rollout_preset_key")})
    results: dict[str, FaFitResult] = {}
    for p_key in preset_keys:
        keep_mask = np.array(
            [m.get("rollout_preset_key") == p_key for m in metadata], dtype=bool
        )
        sub_matrix = matrix[keep_mask]
        sub_meta = [m for m, k in zip(metadata, keep_mask) if k]
        print(
            f"\n[Per-preset FA] preset={p_key!r}  rows={sub_matrix.shape[0]}  "
            f"rotation={rotation}"
        )
        results[p_key] = _fit_fa(
            sub_matrix, sub_meta, items,
            label=p_key, rotation=rotation,
            do_residualize=False,
        )
    return results


# ═════════════════════════════════════════════════════════════════════════════
# Stats
# ═════════════════════════════════════════════════════════════════════════════


def _preset_eta2(fa: FaFitResult) -> list[dict]:
    eta2 = prompt_effects(fa.scores, fa.metadata, group_field="rollout_preset_key")
    return [
        {
            "factor": i + 1,
            "proportion_variance": float(fa.proportion_variance[i]),
            "eta2_preset": float(eta2[i]),
            "rotation": fa.rotation,
            "label": fa.label,
        }
        for i in range(fa.scores.shape[1])
    ]


def _tucker_comparison(
    per_preset: dict[str, FaFitResult],
    *,
    anchor: str,
) -> tuple[dict[str, np.ndarray], dict[str, list]]:
    """Full φ matrices + greedy alignments for every target vs ``anchor``."""
    anchor_fa = per_preset[anchor]
    full_phi: dict[str, np.ndarray] = {}
    alignments = {}
    for target, target_fa in per_preset.items():
        if target == anchor:
            continue
        # Align items — in this flow every per-preset FA shares the same
        # ``items`` list (we filter rows, not columns, before preprocessing).
        # The preprocessing may drop different low-variance columns per
        # preset; we restrict to their item-id intersection to keep rows
        # comparable.
        shared_ids = [it["item_id"] for it in anchor_fa.items] \
            if [it["item_id"] for it in anchor_fa.items] == [it["item_id"] for it in target_fa.items] \
            else sorted(
                set(it["item_id"] for it in anchor_fa.items)
                & set(it["item_id"] for it in target_fa.items)
            )
        anchor_idx = [
            i for i, it in enumerate(anchor_fa.items)
            if it["item_id"] in shared_ids
        ]
        target_idx = [
            i for i, it in enumerate(target_fa.items)
            if it["item_id"] in shared_ids
        ]
        # Re-sort target_idx to match anchor's item order.
        anchor_id_order = [anchor_fa.items[i]["item_id"] for i in anchor_idx]
        target_id_to_local = {target_fa.items[i]["item_id"]: i for i in target_idx}
        target_idx_reordered = [target_id_to_local[iid] for iid in anchor_id_order]

        L_a = anchor_fa.loadings[anchor_idx]
        L_b = target_fa.loadings[target_idx_reordered]
        phi = tucker_phi_matrix(L_a, L_b)
        full_phi[target] = phi
        alignments[target] = align_factors(L_a, L_b)
    return full_phi, alignments


# ═════════════════════════════════════════════════════════════════════════════
# Plots
# ═════════════════════════════════════════════════════════════════════════════


def _plot_preset_variance_bars(
    baseline_rows: list[dict],
    residualised_rows: list[dict],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    rotations = sorted({r["rotation"] for r in baseline_rows + residualised_rows})
    fig, axes = plt.subplots(1, len(rotations), figsize=(6 * len(rotations), 5),
                             squeeze=False)
    for ax, rot in zip(axes[0], rotations):
        b = [r for r in baseline_rows if r["rotation"] == rot]
        d = [r for r in residualised_rows if r["rotation"] == rot]
        # Order by baseline proportion_variance desc so rank is intuitive.
        b_sorted = sorted(b, key=lambda r: -r["proportion_variance"])
        rank = [r["factor"] for r in b_sorted]
        b_eta = [r["eta2_preset"] for r in b_sorted]
        d_by_factor = {r["factor"]: r for r in d}
        d_eta = [d_by_factor[f]["eta2_preset"] if f in d_by_factor else np.nan
                 for f in rank]

        x = np.arange(len(rank))
        w = 0.4
        ax.bar(x - w / 2, b_eta, width=w, label="baseline", color="#B71C1C")
        ax.bar(x + w / 2, d_eta, width=w, label="residualised", color="#2E7D32")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{f}" for f in rank])
        ax.set_ylabel("η² (preset split)")
        ax.set_xlabel("factor (baseline-rank order)")
        ax.set_title(f"rotation = {rot}")
        ax.set_ylim(0, 1.0)
        ax.legend()
    fig.suptitle("Preset-variance decomposition — baseline vs preset-residualised FA")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[Plot] Wrote {out_path}")


def _plot_tucker_heatmaps(
    full_phi: dict[str, np.ndarray],
    anchor: str,
    rotation: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    targets = list(full_phi.keys())
    if not targets:
        return
    k = next(iter(full_phi.values())).shape[0]
    fig, axes = plt.subplots(
        1, len(targets), figsize=(4 * len(targets), 4 + 0.4 * k), squeeze=False,
    )
    for ax, tgt in zip(axes[0], targets):
        phi = full_phi[tgt]
        im = ax.imshow(phi, cmap="viridis", vmin=0, vmax=1, origin="lower")
        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                val = phi[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if val < 0.6 else "black")
        ax.set_xlabel(f"{tgt} factor")
        ax.set_ylabel(f"{anchor} factor")
        ax.set_xticks(range(phi.shape[1]), [f"F{i+1}" for i in range(phi.shape[1])])
        ax.set_yticks(range(phi.shape[0]), [f"F{i+1}" for i in range(phi.shape[0])])
        ax.set_title(f"{anchor} → {tgt}")
    fig.suptitle(f"Tucker's |φ| — rotation={rotation}")
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="|φ|")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[Plot] Wrote {out_path}")


def _plot_aligned_phi_strip(
    alignments_by_rot: dict[str, dict[str, list]],
    out_path: Path,
) -> None:
    """Show the aligned top-|φ| per factor, one column per rotation."""
    import matplotlib.pyplot as plt

    rotations = list(alignments_by_rot.keys())
    fig, axes = plt.subplots(1, len(rotations), figsize=(6 * len(rotations), 5),
                             squeeze=False)
    for ax, rot in zip(axes[0], rotations):
        aligns = alignments_by_rot[rot]
        targets = list(aligns.keys())
        if not targets:
            ax.axis("off")
            continue
        k = len(aligns[targets[0]])
        x = np.arange(k)
        for t_idx, tgt in enumerate(targets):
            phis = [a.phi for a in aligns[tgt]]
            ax.plot(x, phis, marker="o", label=tgt)
        for thr, lbl, color in [
            (0.95, "good (≥0.95)", "#2E7D32"),
            (0.85, "fair (≥0.85)", "#F9A825"),
            (0.70, "poor (≥0.70)", "#E53935"),
        ]:
            ax.axhline(thr, color=color, lw=0.6, ls="--", label=lbl)
        ax.set_xticks(x, [f"F{i+1}" for i in range(k)])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("anchor factor")
        ax.set_ylabel("|φ| (best match)")
        ax.set_title(f"rotation = {rot}")
        ax.legend(fontsize=8)
    fig.suptitle("Tucker's φ after greedy factor alignment")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[Plot] Wrote {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# IO helpers
# ═════════════════════════════════════════════════════════════════════════════


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Write] {path}")


def _write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[Write] {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    out_dir = SCRATCH_ROOT / "external_analysis" / OUTPUT_TAG
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir}")

    # ── Validate config ────────────────────────────────────────────────
    for p in PRESETS:
        if p not in EXTERNAL_ROLLOUT_PRESETS:
            raise KeyError(f"Unknown external preset {p!r}")
    for q in QUESTIONNAIRES:
        if q not in QUESTIONNAIRE_PRESETS:
            raise KeyError(f"Unknown questionnaire preset {q!r}")

    # ── Stage 1 (cache-aware): ingest every preset once ────────────────
    rollout_dirs = _ingest_all_presets()

    # ── Rollout-stats stage ────────────────────────────────────────────
    _run_rollout_stats(rollout_dirs, out_dir)

    if STOP_AFTER_ROLLOUT_STATS:
        print(
            "\n[Main] STOP_AFTER_ROLLOUT_STATS=True — stopping before Stage 2. "
            "Flip the flag to False and rerun to continue."
        )
        return

    # ── Stage 2 (cache-aware) for every pair ───────────────────────────
    pair_data = _administer_all_pairs(rollout_dirs)
    pair_version = {
        (r, q): QUESTIONNAIRE_PRESETS[q].version
        for (r, q) in pair_data.keys()
    }

    # ── Combine ────────────────────────────────────────────────────────
    combined_dir = out_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    matrix, metadata, items = combine_per_pair_outputs(
        pair_data, pair_version,
        out_dir=combined_dir / "questionnaire",
        provenance_extra={"output_tag": OUTPUT_TAG, "script": __file__},
    )
    print(
        f"\nCombined matrix: rows={matrix.shape[0]}, cols={matrix.shape[1]}, "
        f"nan_frac={np.isnan(matrix).mean():.4f}"
    )

    # ── FA flavours × rotations ────────────────────────────────────────
    baseline_eta2_rows: list[dict] = []
    residualised_eta2_rows: list[dict] = []
    per_preset_alignments_by_rot: dict[str, dict[str, list]] = {}
    per_preset_phi_by_rot: dict[str, dict[str, np.ndarray]] = {}
    per_preset_classification_rows: list[dict] = []

    for rotation in FA_ROTATIONS:
        print(f"\n{'=' * 60}\n[FA] rotation={rotation}\n{'=' * 60}")

        print(f"\n-- baseline (raw) --")
        baseline = _fit_fa(matrix, metadata, items,
                           label="baseline", rotation=rotation)
        baseline_eta2_rows.extend(_preset_eta2(baseline))

        print(f"\n-- preset-residualised --")
        residualised = _fit_fa(matrix, metadata, items,
                               label="residualised", rotation=rotation,
                               do_residualize=True,
                               residualize_group_field="rollout_preset_key")
        residualised_eta2_rows.extend(_preset_eta2(residualised))

        print(f"\n-- per-preset FAs --")
        per_preset = _fit_per_preset_fas(matrix, metadata, items,
                                         rotation=rotation)
        preset_keys = list(per_preset.keys())
        if len(preset_keys) < 2:
            print(f"[Tucker] Only {len(preset_keys)} preset(s) — skipping congruence.")
            continue
        anchor = preset_keys[0]
        full_phi, alignments = _tucker_comparison(per_preset, anchor=anchor)
        per_preset_phi_by_rot[rotation] = full_phi
        per_preset_alignments_by_rot[rotation] = alignments

        for tgt, aligns in alignments.items():
            for a in aligns:
                per_preset_classification_rows.append({
                    "rotation": rotation,
                    "anchor_preset": anchor,
                    "target_preset": tgt,
                    "anchor_factor": a.anchor_factor + 1,
                    "target_factor": a.target_factor + 1 if a.target_factor >= 0 else None,
                    "phi": float(a.phi),
                    "interpretation": classify_phi(a.phi),
                })

    # ── Save numeric outputs ───────────────────────────────────────────
    _write_csv(out_dir / "preset_variance_baseline.csv", baseline_eta2_rows)
    _write_csv(out_dir / "preset_variance_residualised.csv", residualised_eta2_rows)
    _write_csv(out_dir / "tucker_pairwise.csv", per_preset_classification_rows)

    alignment_summary: dict[str, Any] = {}
    for rot, aligns in per_preset_alignments_by_rot.items():
        anchor = next(iter({r["anchor_preset"] for r in per_preset_classification_rows
                            if r["rotation"] == rot}), None)
        alignment_summary[rot] = summarise_alignment(aligns, anchor_label=anchor or "")
    _write_json(out_dir / "tucker_alignment_summary.json", alignment_summary)

    # ── Plots ──────────────────────────────────────────────────────────
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_preset_variance_bars(
        baseline_eta2_rows, residualised_eta2_rows,
        plots_dir / "preset_variance_bars.png",
    )
    for rotation, full_phi in per_preset_phi_by_rot.items():
        anchor = next(iter(
            {r["anchor_preset"] for r in per_preset_classification_rows
             if r["rotation"] == rotation}
        ))
        _plot_tucker_heatmaps(
            full_phi, anchor=anchor, rotation=rotation,
            out_path=plots_dir / f"tucker_heatmaps_{rotation}.png",
        )
    if per_preset_alignments_by_rot:
        _plot_aligned_phi_strip(
            per_preset_alignments_by_rot,
            plots_dir / "tucker_aligned_phi_strip.png",
        )

    # ── Summary printout ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for rot in FA_ROTATIONS:
        print(f"\nrotation = {rot}")
        b = [r for r in baseline_eta2_rows if r["rotation"] == rot]
        d = [r for r in residualised_eta2_rows if r["rotation"] == rot]
        print("  baseline    : top factor η²(preset) =",
              f"{max(x['eta2_preset'] for x in b):.3f}" if b else "n/a")
        print("  residualised: top factor η²(preset) =",
              f"{max(x['eta2_preset'] for x in d):.3f}" if d else "n/a")
    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
