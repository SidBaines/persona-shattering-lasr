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
        - Parallel analysis   — Horn's null-reference test for the number
                                of factors to retain. Reported alongside
                                eigenvalue scree so ``FA_N_FACTORS`` has
                                a principled justification.
        - Preset-variance η²  — how much of each factor's variance comes
                                from the preset/model split. Baseline
                                solution only; residualised factor scores
                                have η²(preset) ≡ 0 by construction
                                (subtracting per-preset means forces all
                                per-preset means of any linear combination
                                of residualised columns to zero), so we
                                instead measure how much the factor
                                *structure* survives preset-mean removal
                                via Tucker's φ between baseline and
                                residualised loadings.
        - Tucker's φ          — factor-structure similarity (i) across
                                every pair of per-preset FA solutions and
                                (ii) between baseline and preset-
                                residualised solutions, with greedy one-
                                to-one factor alignment.
        - Retention table     — per-preset row counts at each filtering
                                stage (Stage-1 sampled → Stage-2 context
                                filter → parse success → combine
                                intersection → FA input), so unequal data
                                loss across presets is visible at a glance.
        - n-factors sweep     — the key top-line summaries (max baseline
                                η², baseline→residualised |φ|) computed
                                at a user-specified range of n_factors
                                so conclusions can be checked for
                                robustness to that choice.
        - Oblique factor correlations — when the rotation is oblique the
                                inter-factor correlation matrix is dumped
                                for inspection (r > ~0.3 changes the
                                interpretation of any "pure" factor).
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
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from src_dev.factor_analysis.factor_analysis import run_factor_analysis
from src_dev.factor_analysis.interpretation import prompt_effects
from src_dev.factor_analysis.parallel_analysis import parallel_analysis
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
    # "prism_oasst_pythia_12b",  # dropped 2026-04-22: v5 Likert is 99.7% NaN
    # (first token never lands in {1..5} top-20 logprobs — legacy oasst
    # instruction-following fails on structured response formats). Trait is
    # ~84% valid but combine's sample_id intersection with the empty v5 rows
    # leaves the preset contributing ~nothing. Revisit with a rescue strategy.
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
STOP_AFTER_ROLLOUT_STATS: bool = False

# Set True to stop after Stage 2 finishes (ingest + admin + HF upload of
# every pair). Skips combine + FA + Tucker's + plots. Useful for
# parallelising Stage 2 across multiple GPUs with disjoint preset
# subsets: run one tmux session per GPU, each with CUDA_VISIBLE_DEVICES
# + EXT_PRESETS set, then kick off a final run with
# STOP_AFTER_STAGE2=False (and EXT_PRESETS covering the full set) to
# hit the merged HF cache and produce the analysis artefacts.
STOP_AFTER_STAGE2: bool = False

# Env-var override for the PRESETS list — comma-separated keys.
# Example (bash):
#   EXT_PRESETS=kwai_swe,prism_llama2_13b_chat uv run python -m ...
# The env var takes precedence over the module-level PRESETS list.
def _resolve_presets(default: list[str]) -> list[str]:
    raw = os.environ.get("EXT_PRESETS")
    if raw is None:
        return default
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        return default
    print(f"[Config] EXT_PRESETS override active: {keys}")
    return keys

# ── FA knobs ────────────────────────────────────────────────────────────────
FA_METHOD = "principal"
FA_N_FACTORS = 7
FA_ROTATIONS: list[str] = ["oblimin", "varimax"]
MIN_ITEM_VARIANCE = 0.1
HIGH_VARIANCE_PERSONA_DROP_PCT = 0.0

# n_factors robustness sweep: fit lightweight baseline + residualised FAs at
# each of these n and report whether the top-line claims (max baseline
# η²(preset), min/median baseline→residualised |φ|) survive perturbations of
# FA_N_FACTORS. Primary n reuses the full analysis (this list can include
# FA_N_FACTORS but the sweep CSV will just duplicate the primary result for
# cross-check). Set to [] to skip the sweep entirely.
FA_N_FACTORS_SWEEP: list[int] = [5, 6, 7, 8]

# Horn's parallel analysis: permutation-method reference distribution is
# more appropriate for ordinal Likert data than the default Gaussian. 100
# iterations is the standard compromise between stability and compute.
PARALLEL_ANALYSIS_ITERATIONS = 100
PARALLEL_ANALYSIS_METHOD = "permutation"

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

# ── Disk hygiene ────────────────────────────────────────────────────────────
# With 9 external models (many 13B+), the HF hub cache can easily exceed
# 200 GB and blow up mid-run (vLLM snapshot_download hit "No space left on
# device" on 2026-04-21). After both questionnaires for a preset complete,
# delete that preset's model cache so the next preset's download has room.
# Cache-hit presets are skipped (nothing was downloaded for them).
CLEANUP_HF_MODEL_CACHE_PER_PRESET: bool = True

# ── Output ──────────────────────────────────────────────────────────────────
OUTPUT_TAG = "multisource_n500_7models"  # change when config changes


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
):
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
    return run_stage_questionnaire(
        cfg,
        num_conversation_turns=r.min_assistant_turns,
        openrouter_provider_routing=None,
        fc_pair_sign_alignment=FC_PAIR_SIGN_ALIGNMENT,
    )


def _ingest_all_presets(presets: list[str]) -> dict[str, Path]:
    """Run Stage 1 ingest for each preset; return the canonical rollout dir."""
    rollout_dirs: dict[str, Path] = {}
    for r_key in presets:
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


def _delete_hf_model_cache(model_repo_id: str) -> None:
    """Delete the HF hub cache for a specific model repo.

    Frees disk so the next preset's snapshot download has room. Idempotent
    and safe: missing caches are a no-op.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return
    # HF stores models under "models--<org>--<name>" (slashes → double dash).
    cache_root = Path(HF_HUB_CACHE)
    cache_dir = cache_root / f"models--{model_repo_id.replace('/', '--')}"
    if not cache_dir.exists():
        print(f"[Cleanup] HF cache for {model_repo_id!r} not present — skipping")
        return
    # Size for logging — best-effort, skip on error.
    try:
        total_bytes = sum(
            p.stat().st_size for p in cache_dir.rglob("*") if p.is_file()
        )
        size_gb = total_bytes / (1024**3)
    except OSError:
        size_gb = float("nan")
    try:
        shutil.rmtree(cache_dir)
        print(
            f"[Cleanup] Deleted HF cache for {model_repo_id!r} "
            f"(~{size_gb:.1f} GiB freed)"
        )
    except OSError as exc:
        print(f"[Cleanup] Failed to delete {cache_dir}: {exc}")


def _administer_all_pairs(
    rollout_dirs: dict[str, Path],
    presets: list[str],
) -> dict[tuple[str, str], tuple[np.ndarray, list[dict], list[dict]]]:
    """Run (or hydrate) Stage 2 for every (preset, questionnaire) pair."""
    pair_data: dict[tuple[str, str], tuple[np.ndarray, list[dict], list[dict]]] = {}
    for r_key in presets:
        preset_generated = False
        for q_key in QUESTIONNAIRES:
            ctx = _build_ctx(r_key, q_key)
            print(f"\n{'#' * 60}")
            print(f"# Stage 2: rollout={r_key!r} × questionnaire={q_key!r}")
            print(f"#   {ctx.questionnaire_run_id}")
            print(f"{'#' * 60}")
            result = _administer_questionnaire(ctx, r_key, q_key)
            if result is not None and getattr(result, "generated", False):
                preset_generated = True
            pair_data[(r_key, q_key)] = load_pair_outputs(ctx.questionnaire_dir)
        # Release this preset's HF model cache before the next preset's
        # snapshot download kicks off. Only bother if we actually generated
        # (cache-hit runs didn't download anything new).
        if CLEANUP_HF_MODEL_CACHE_PER_PRESET and preset_generated:
            model_id = EXTERNAL_ROLLOUT_PRESETS[r_key].assistant_model
            _delete_hf_model_cache(model_id)
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
    n_factors: int
    loadings: np.ndarray                # (n_items_kept, k)
    scores: np.ndarray                  # (n_samples_kept, k)
    proportion_variance: np.ndarray     # (k,)
    metadata: list[dict]                # aligned with scores
    items: list[dict]                   # aligned with loadings
    factor_correlation_matrix: np.ndarray | None = None  # (k, k) for oblique rotations, None for orthogonal


def _fit_fa(
    matrix: np.ndarray,
    metadata: list[dict],
    items: list[dict],
    *,
    label: str,
    rotation: str,
    do_residualize: bool = False,
    residualize_group_field: str | None = None,
    n_factors: int | None = None,
) -> FaFitResult:
    data, meta_filtered, items_filtered, _group_ids = preprocess_response_matrix(
        matrix, metadata, items,
        min_item_variance=MIN_ITEM_VARIANCE,
        high_variance_persona_drop_pct=HIGH_VARIANCE_PERSONA_DROP_PCT,
        do_residualize=do_residualize,
        residualize_group_field=residualize_group_field,
    )
    n = n_factors if n_factors is not None else FA_N_FACTORS
    fa = run_factor_analysis(
        data, n_factors=n, method=FA_METHOD, rotation=rotation,
    )
    return FaFitResult(
        label=label,
        rotation=rotation,
        n_factors=n,
        loadings=fa["loadings"],
        scores=fa["scores"],
        proportion_variance=fa["proportion_variance"],
        metadata=meta_filtered,
        items=items_filtered,
        factor_correlation_matrix=fa.get("factor_correlation_matrix"),
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


def _shared_item_reorder(
    fa_a: FaFitResult,
    fa_b: FaFitResult,
) -> tuple[list[int], list[int]]:
    """Return (a_idx, b_idx) such that ``fa_a.items[a_idx]`` and
    ``fa_b.items[b_idx]`` enumerate the same ``item_id`` sequence.

    Preprocessing can drop different low-variance columns per FA, so the
    two solutions may not share the same item order. We restrict to the
    item-id intersection and sort ``b`` to match ``a``'s order.
    """
    a_ids = [it["item_id"] for it in fa_a.items]
    b_ids = [it["item_id"] for it in fa_b.items]
    if a_ids == b_ids:
        return list(range(len(a_ids))), list(range(len(b_ids)))
    shared = set(a_ids) & set(b_ids)
    a_idx = [i for i, iid in enumerate(a_ids) if iid in shared]
    b_local = {b_ids[i]: i for i in range(len(b_ids)) if b_ids[i] in shared}
    a_order = [a_ids[i] for i in a_idx]
    b_idx = [b_local[iid] for iid in a_order]
    # Post-condition: both index lists point to the same item_id sequence.
    assert [a_ids[i] for i in a_idx] == [b_ids[i] for i in b_idx]
    return a_idx, b_idx


def _baseline_vs_residualised_alignment(
    baseline: FaFitResult,
    residualised: FaFitResult,
) -> list:
    """Tucker's |φ| alignment between baseline and residualised loadings.

    Measures factor-structure robustness to preset-mean removal: a
    baseline factor with |φ| near 1.0 in the residualised solution is
    *not* carried by preset identity; a low |φ| factor IS primarily
    preset-level variance.
    """
    a_idx, b_idx = _shared_item_reorder(baseline, residualised)
    L_base = baseline.loadings[a_idx]
    L_res = residualised.loadings[b_idx]
    return align_factors(L_base, L_res)


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
        # Preprocessing may drop different low-variance columns per preset,
        # so we restrict to the item-id intersection and sort the target
        # to match the anchor's item order before computing |φ|.
        a_idx, b_idx = _shared_item_reorder(anchor_fa, target_fa)
        L_a = anchor_fa.loadings[a_idx]
        L_b = target_fa.loadings[b_idx]
        phi = tucker_phi_matrix(L_a, L_b)
        full_phi[target] = phi
        alignments[target] = align_factors(L_a, L_b)
    return full_phi, alignments


# ═════════════════════════════════════════════════════════════════════════════
# Diagnostics (parallel analysis, retention, n-factors sweep)
# ═════════════════════════════════════════════════════════════════════════════


def _run_parallel_analysis(
    matrix: np.ndarray,
    metadata: list[dict],
    items: list[dict],
    *,
    out_dir: Path,
    plots_dir: Path,
) -> dict[str, Any]:
    """Horn's parallel analysis + Kaiser rule on the baseline-preprocessed matrix.

    Writes a scree plot and a JSON summary with recommended n_factors under
    both criteria, so ``FA_N_FACTORS`` can be cross-checked against a
    principled null-reference test rather than taken on faith.
    """
    data, _, _, _ = preprocess_response_matrix(
        matrix, metadata, items,
        min_item_variance=MIN_ITEM_VARIANCE,
        high_variance_persona_drop_pct=HIGH_VARIANCE_PERSONA_DROP_PCT,
        do_residualize=False,
    )
    print(
        f"\n[ParallelAnalysis] Running on matrix shape={data.shape} "
        f"method={PARALLEL_ANALYSIS_METHOD!r} iters={PARALLEL_ANALYSIS_ITERATIONS}"
    )
    result = parallel_analysis(
        data,
        n_iterations=PARALLEL_ANALYSIS_ITERATIONS,
        method=PARALLEL_ANALYSIS_METHOD,
        random_state=SEED,
    )
    real_ev = result["real_eigenvalues"]
    thresh = result["random_threshold"]
    n_parallel = int(result["n_recommended"])
    n_kaiser = int((real_ev > 1.0).sum())

    summary = {
        "n_parallel_analysis": n_parallel,
        "n_kaiser": n_kaiser,
        "n_configured": FA_N_FACTORS,
        "eigenvalues_top20": [float(x) for x in real_ev[:20]],
        "random_threshold_top20": [float(x) for x in thresh[:20]],
        "n_samples": int(data.shape[0]),
        "n_vars": int(data.shape[1]),
        "parallel_method": PARALLEL_ANALYSIS_METHOD,
        "parallel_iterations": PARALLEL_ANALYSIS_ITERATIONS,
    }
    _write_json(out_dir / "n_factors_diagnostics.json", summary)
    _plot_scree(real_ev, thresh, n_parallel=n_parallel, n_kaiser=n_kaiser,
                n_configured=FA_N_FACTORS, out_path=plots_dir / "scree.png")
    print(
        f"[ParallelAnalysis] recommended n: parallel={n_parallel}, "
        f"Kaiser>1={n_kaiser}, configured FA_N_FACTORS={FA_N_FACTORS}"
    )
    if FA_N_FACTORS > n_parallel + 2 or FA_N_FACTORS < max(1, n_parallel - 2):
        print(
            f"[ParallelAnalysis] WARNING: FA_N_FACTORS={FA_N_FACTORS} is far "
            f"from the parallel-analysis recommendation ({n_parallel}). "
            "Consider re-running the sweep and picking a more principled value."
        )
    return summary


def _per_preset_retention_table(
    pair_data: dict[tuple[str, str], tuple[np.ndarray, list[dict], list[dict]]],
    rollout_dirs: dict[str, Path],
    combined_metadata: list[dict],
    fa_baseline_metadata: list[dict] | None,
) -> list[dict]:
    """Per-preset row counts at each filtering stage in the pipeline.

    Makes unequal data loss across presets visible at a glance. If one
    preset (e.g. a short-context model under a long-rollout source) loses
    a large fraction of its data at the Stage-2 context filter while
    another doesn't, per-preset FA comparisons are not apples-to-apples.
    """
    q_keys_order: list[str] = []
    seen: set[str] = set()
    for (_r, q) in pair_data.keys():
        if q not in seen:
            seen.add(q)
            q_keys_order.append(q)

    rows: list[dict] = []
    for p_key, rollout_dir in rollout_dirs.items():
        row: dict[str, Any] = {"preset_key": p_key}

        canonical = rollout_dir / "datasets" / "canonical_samples.jsonl"
        if canonical.exists():
            with canonical.open("r", encoding="utf-8") as f:
                row["stage1_sampled"] = sum(1 for ln in f if ln.strip())
        else:
            row["stage1_sampled"] = None

        for q_key in q_keys_order:
            key = (p_key, q_key)
            if key not in pair_data:
                row[f"q[{q_key}]_ctx_kept"] = None
                row[f"q[{q_key}]_fully_parsed"] = None
                continue
            mat, _meta, _items = pair_data[key]
            K, _N = mat.shape
            # Rows with no NaN cells — i.e. every item parsed successfully.
            # These are the only rows that survive preprocess_response_matrix's
            # complete-case deletion.
            n_fully_parsed = int((~np.isnan(mat).any(axis=1)).sum())
            row[f"q[{q_key}]_ctx_kept"] = K
            row[f"q[{q_key}]_fully_parsed"] = n_fully_parsed

        row["combine_intersected"] = sum(
            1 for m in combined_metadata
            if m.get("rollout_preset_key") == p_key
        )
        if fa_baseline_metadata is not None:
            row["fa_baseline_rows"] = sum(
                1 for m in fa_baseline_metadata
                if m.get("rollout_preset_key") == p_key
            )
        else:
            row["fa_baseline_rows"] = None
        rows.append(row)
    return rows


def _n_factors_sweep(
    matrix: np.ndarray,
    metadata: list[dict],
    items: list[dict],
    *,
    rotations: list[str],
    n_factors_list: list[int],
) -> list[dict]:
    """Lightweight robustness check: top-line summaries at each n_factors.

    For every (rotation, n) pair, refits baseline and residualised FAs and
    reports max/median baseline η²(preset) plus min/median
    baseline→residualised |φ|. Skips per-preset Tucker entirely (it is the
    expensive part of the main analysis and unnecessary for a "does the
    top-line finding survive n perturbations?" check).

    Fits that fail (e.g. n_factors exceeds n_columns after preprocessing)
    are logged and skipped rather than aborting the sweep.
    """
    sweep_rows: list[dict] = []
    for rotation in rotations:
        for n in n_factors_list:
            print(f"\n[n_factors sweep] rotation={rotation} n={n}")
            try:
                baseline = _fit_fa(
                    matrix, metadata, items,
                    label=f"baseline_n{n}", rotation=rotation,
                    n_factors=n,
                )
                residualised = _fit_fa(
                    matrix, metadata, items,
                    label=f"residualised_n{n}", rotation=rotation,
                    do_residualize=True,
                    residualize_group_field="rollout_preset_key",
                    n_factors=n,
                )
            except Exception as exc:
                print(f"  skipped: {type(exc).__name__}: {exc}")
                sweep_rows.append({
                    "rotation": rotation,
                    "n_factors": n,
                    "status": f"skipped: {type(exc).__name__}",
                })
                continue
            eta2_rows = _preset_eta2(baseline)
            eta2_vals = [r["eta2_preset"] for r in eta2_rows
                         if not np.isnan(r["eta2_preset"])]
            robustness = _baseline_vs_residualised_alignment(baseline, residualised)
            phis = [a.phi for a in robustness if not np.isnan(a.phi)]
            sweep_rows.append({
                "rotation": rotation,
                "n_factors": n,
                "status": "ok",
                "max_baseline_eta2_preset":
                    float(max(eta2_vals)) if eta2_vals else float("nan"),
                "median_baseline_eta2_preset":
                    float(np.median(eta2_vals)) if eta2_vals else float("nan"),
                "min_baseline_res_phi":
                    float(min(phis)) if phis else float("nan"),
                "median_baseline_res_phi":
                    float(np.median(phis)) if phis else float("nan"),
                "n_factors_aligned": len(phis),
            })
    return sweep_rows


# ═════════════════════════════════════════════════════════════════════════════
# Plots
# ═════════════════════════════════════════════════════════════════════════════


def _plot_scree(
    real_eigenvalues: np.ndarray,
    random_threshold: np.ndarray,
    *,
    n_parallel: int,
    n_kaiser: int,
    n_configured: int,
    out_path: Path,
    top_k: int = 20,
) -> None:
    """Scree plot with parallel-analysis threshold and Kaiser reference line."""
    import matplotlib.pyplot as plt

    k = min(top_k, len(real_eigenvalues))
    x = np.arange(1, k + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, real_eigenvalues[:k], marker="o", lw=1.5, color="#1565C0",
            label="real eigenvalues")
    ax.plot(x, random_threshold[:k], marker="s", lw=1.0, color="#B71C1C",
            label=f"{PARALLEL_ANALYSIS_METHOD} 95%-tile reference")
    ax.axhline(1.0, color="gray", lw=0.5, ls=":", label="Kaiser (λ=1)")
    ax.axvline(n_parallel + 0.5, color="#2E7D32", lw=1.0, ls="--",
               label=f"parallel: n={n_parallel}")
    ax.axvline(n_kaiser + 0.5, color="gray", lw=0.8, ls="--",
               label=f"Kaiser: n={n_kaiser}")
    ax.axvline(n_configured + 0.5, color="#F9A825", lw=1.2, ls="-",
               label=f"configured: n={n_configured}")
    ax.set_xlabel("factor index (descending eigenvalue)")
    ax.set_ylabel("eigenvalue")
    ax.set_title("Eigenvalue scree and parallel-analysis threshold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[Plot] Wrote {out_path}")


def _plot_preset_variance_bars(
    baseline_rows: list[dict],
    robustness_rows: list[dict],
    out_path: Path,
) -> None:
    """Per-factor preset dominance (η²) and robustness to residualisation (|φ|).

    For each baseline factor we show:
      - η²(preset): how much of the factor's variance is attributable to
        the preset split. Values near 1.0 mean the factor is essentially
        model identity; values near 0 mean it is within-preset variance.
      - |φ|(baseline→residualised): Tucker's congruence between the
        baseline factor and its best-match factor in the
        preset-residualised solution. Values near 1.0 mean the factor
        survives preset-mean removal (not carried by preset identity);
        low values mean the factor IS primarily preset-level variance.

    The two diagnostics are complementary: a factor with high η² AND low
    |φ|(→residualised) is unambiguously a preset-identity factor; a
    factor with low η² AND high |φ|(→residualised) is a robust
    within-preset structural factor.
    """
    import matplotlib.pyplot as plt

    rotations = sorted({r["rotation"] for r in baseline_rows})
    fig, axes = plt.subplots(1, len(rotations), figsize=(6 * len(rotations), 5),
                             squeeze=False)
    for ax, rot in zip(axes[0], rotations):
        b = [r for r in baseline_rows if r["rotation"] == rot]
        r_rows = [r for r in robustness_rows if r["rotation"] == rot]
        # Order by baseline proportion_variance desc so rank is intuitive.
        b_sorted = sorted(b, key=lambda r: -r["proportion_variance"])
        rank = [r["factor"] for r in b_sorted]
        b_eta = [r["eta2_preset"] for r in b_sorted]
        r_by_factor = {r["anchor_factor"]: r for r in r_rows}
        phis = [
            r_by_factor[f]["phi"] if (f in r_by_factor and not np.isnan(r_by_factor[f]["phi"])) else np.nan
            for f in rank
        ]

        x = np.arange(len(rank))
        ax.bar(x, b_eta, width=0.6, color="#B71C1C",
               label="η²(preset) — baseline")
        ax.plot(x, phis, marker="o", ms=8, lw=0, color="#1565C0",
                label="|φ|(baseline→residualised)")
        for thr, col in [(0.95, "#2E7D32"), (0.85, "#F9A825"), (0.70, "#E53935")]:
            ax.axhline(thr, color=col, lw=0.5, ls=":")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{f}" for f in rank])
        ax.set_ylabel("η² / |φ|")
        ax.set_xlabel("factor (baseline-rank order)")
        ax.set_title(f"rotation = {rot}")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Baseline factor diagnostics: preset-variance η² and "
        "robustness to preset-mean removal (|φ| vs residualised)"
    )
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

    # Resolve active preset list (env override > module default).
    active_presets = _resolve_presets(PRESETS)

    # ── Validate config ────────────────────────────────────────────────
    for p in active_presets:
        if p not in EXTERNAL_ROLLOUT_PRESETS:
            raise KeyError(f"Unknown external preset {p!r}")
    for q in QUESTIONNAIRES:
        if q not in QUESTIONNAIRE_PRESETS:
            raise KeyError(f"Unknown questionnaire preset {q!r}")

    # ── Stage 1 (cache-aware): ingest every preset once ────────────────
    rollout_dirs = _ingest_all_presets(active_presets)

    # ── Rollout-stats stage ────────────────────────────────────────────
    _run_rollout_stats(rollout_dirs, out_dir)

    if STOP_AFTER_ROLLOUT_STATS:
        print(
            "\n[Main] STOP_AFTER_ROLLOUT_STATS=True — stopping before Stage 2. "
            "Flip the flag to False and rerun to continue."
        )
        return

    # ── Stage 2 (cache-aware) for every pair ───────────────────────────
    pair_data = _administer_all_pairs(rollout_dirs, active_presets)

    if STOP_AFTER_STAGE2 or os.environ.get("STOP_AFTER_STAGE2") in ("1", "true", "True"):
        print(
            f"\n[Main] STOP_AFTER_STAGE2 active — Stage 2 complete for "
            f"{len(active_presets)} presets × {len(QUESTIONNAIRES)} questionnaires. "
            "Outputs uploaded to HF; skipping combine + FA. "
            "Run again without STOP_AFTER_STAGE2 (and with the full "
            "PRESETS list) to do the analysis."
        )
        return


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

    # ── Parallel analysis / scree diagnostic ───────────────────────────
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _run_parallel_analysis(
        matrix, metadata, items,
        out_dir=out_dir, plots_dir=plots_dir,
    )

    # ── FA flavours × rotations ────────────────────────────────────────
    baseline_eta2_rows: list[dict] = []
    # One row per (rotation × baseline anchor factor) recording the best
    # Tucker's |φ| match in the preset-residualised solution. Replaces
    # the old "residualised η²" rows, which were ≡0 by construction
    # (per-group mean of residualised scores is identically 0, so
    # SS_between is 0 and η²(preset) is 0 regardless of factor structure).
    residualised_robustness_rows: list[dict] = []
    per_preset_alignments_by_rot: dict[str, dict[str, list]] = {}
    per_preset_phi_by_rot: dict[str, dict[str, np.ndarray]] = {}
    per_preset_classification_rows: list[dict] = []
    # Captured from the first rotation's baseline FA so retention-table
    # computation (post-preprocessing row counts) doesn't repeat the fit.
    fa_baseline_metadata_for_retention: list[dict] | None = None
    # Per-rotation oblique factor correlation matrices (None for orthogonal).
    oblique_factor_corr_by_label: dict[str, np.ndarray] = {}

    for rotation in FA_ROTATIONS:
        print(f"\n{'=' * 60}\n[FA] rotation={rotation}\n{'=' * 60}")

        print(f"\n-- baseline (raw) --")
        baseline = _fit_fa(matrix, metadata, items,
                           label="baseline", rotation=rotation)
        baseline_eta2_rows.extend(_preset_eta2(baseline))
        if fa_baseline_metadata_for_retention is None:
            fa_baseline_metadata_for_retention = baseline.metadata
        if baseline.factor_correlation_matrix is not None:
            oblique_factor_corr_by_label[f"baseline_{rotation}"] = (
                baseline.factor_correlation_matrix
            )

        print(f"\n-- preset-residualised --")
        residualised = _fit_fa(matrix, metadata, items,
                               label="residualised", rotation=rotation,
                               do_residualize=True,
                               residualize_group_field="rollout_preset_key")
        if residualised.factor_correlation_matrix is not None:
            oblique_factor_corr_by_label[f"residualised_{rotation}"] = (
                residualised.factor_correlation_matrix
            )
        robustness = _baseline_vs_residualised_alignment(baseline, residualised)
        for a in robustness:
            residualised_robustness_rows.append({
                "rotation": rotation,
                "anchor_factor": a.anchor_factor + 1,
                "residualised_factor": a.target_factor + 1 if a.target_factor >= 0 else None,
                "phi": float(a.phi),
                "interpretation": classify_phi(a.phi),
            })

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

    # ── Retention table (per-preset row counts at each filtering stage) ─
    retention_rows = _per_preset_retention_table(
        pair_data, rollout_dirs, metadata, fa_baseline_metadata_for_retention,
    )
    _write_csv(out_dir / "per_preset_retention.csv", retention_rows)
    print("\nPer-preset retention:")
    for row in retention_rows:
        parsed_cols = ", ".join(
            f"{k}={row[k]}" for k in row.keys()
            if k.startswith("q[") and k.endswith("_fully_parsed")
        )
        print(
            f"  {row['preset_key']:<35s}  "
            f"sampled={row['stage1_sampled']}  "
            f"{parsed_cols}  "
            f"combined={row['combine_intersected']}  "
            f"fa={row['fa_baseline_rows']}"
        )

    # ── n-factors robustness sweep ─────────────────────────────────────
    if FA_N_FACTORS_SWEEP:
        sweep_rows = _n_factors_sweep(
            matrix, metadata, items,
            rotations=FA_ROTATIONS, n_factors_list=FA_N_FACTORS_SWEEP,
        )
        _write_csv(out_dir / "n_factors_sweep.csv", sweep_rows)

    # ── Save numeric outputs ───────────────────────────────────────────
    _write_csv(out_dir / "preset_variance_baseline.csv", baseline_eta2_rows)
    _write_csv(
        out_dir / "baseline_vs_residualised_robustness.csv",
        residualised_robustness_rows,
    )
    _write_csv(out_dir / "tucker_pairwise.csv", per_preset_classification_rows)

    # Oblique rotations (oblimin/promax) produce a factor correlation matrix;
    # for orthogonal rotations (varimax) it's None. Dump one CSV per
    # (flavour × rotation) so the factor-correlation structure is visible —
    # correlations > ~0.3 change the interpretation of any "pure" factor.
    for label, corr in oblique_factor_corr_by_label.items():
        out_path = out_dir / f"factor_correlation_{label}.csv"
        header = [""] + [f"F{i+1}" for i in range(corr.shape[1])]
        import csv
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i, row in enumerate(corr):
                writer.writerow([f"F{i+1}"] + [f"{v:.4f}" for v in row])
        print(f"[Write] {out_path}")

    alignment_summary: dict[str, Any] = {}
    for rot, aligns in per_preset_alignments_by_rot.items():
        anchor = next(iter({r["anchor_preset"] for r in per_preset_classification_rows
                            if r["rotation"] == rot}), None)
        alignment_summary[rot] = summarise_alignment(aligns, anchor_label=anchor or "")
    _write_json(out_dir / "tucker_alignment_summary.json", alignment_summary)

    # ── Plots ──────────────────────────────────────────────────────────
    # plots_dir was already created above for the parallel-analysis scree plot.
    _plot_preset_variance_bars(
        baseline_eta2_rows, residualised_robustness_rows,
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
        r_rows = [r for r in residualised_robustness_rows if r["rotation"] == rot]
        phis = [r["phi"] for r in r_rows if not np.isnan(r["phi"])]
        print(
            "  baseline                  : top factor η²(preset) = "
            + (f"{max(x['eta2_preset'] for x in b):.3f}" if b else "n/a")
        )
        print(
            "  baseline→residualised |φ|: min = "
            + (f"{min(phis):.3f}" if phis else "n/a")
            + ", median = "
            + (f"{float(np.median(phis)):.3f}" if phis else "n/a")
        )
    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
