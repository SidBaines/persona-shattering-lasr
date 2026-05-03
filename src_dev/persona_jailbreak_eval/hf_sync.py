"""HF-monorepo hydrate + upload helpers for the persona-jailbreak eval.

Run-dir convention (per repo CLAUDE.md "cross-model evals" path scheme):

    {repo_id}/evals/{eval_type}/{model_slug}/{run_slug}/
        responses/responses_<condition>.jsonl
        judgments/judgments_<condition>.jsonl
        aggregate/...

We deliberately do NOT upload ``responses/baked_lora_soups/`` — those are
gigabytes and trivially re-derivable from the LoRA registry.

Workflow at the top of each driver run:

    1. ``hydrate_run_dir_from_hf(...)`` — if the remote run-dir exists, pull
       it into the local run-dir. Idempotent inference + judge stages then
       see "already complete" and skip work.
    2. Run inference + judging normally.
    3. ``upload_run_dir_to_hf(...)`` — push the local run-dir up. Two
       natural call sites: once after inference (preserves expensive
       generations even if judging fails) and once after aggregation.
"""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub.utils import HfHubHTTPError

from src_dev.common.lora_catalogue import HF_REPO as DEFAULT_HF_REPO_ID
from src_dev.utils.hf_hub import (
    check_exists_in_dataset_repo,
    download_path_to_dir,
    login_from_env,
    upload_folder_to_dataset_repo,
)

logger = logging.getLogger(__name__)


# ── Path conventions ────────────────────────────────────────────────────


def hf_run_path(*, eval_type: str, model_slug: str, run_slug: str) -> str:
    """Return the canonical ``evals/...`` path-in-repo for a run."""
    return f"evals/{eval_type}/{model_slug}/{run_slug}"


# ── Patterns ────────────────────────────────────────────────────────────


# Files we PUSH to HF: everything that's expensive to recompute. Baked LoRA
# soups are excluded (derivable from src_dev.common.lora_catalogue).
DEFAULT_UPLOAD_PATTERNS: tuple[str, ...] = (
    "responses/responses_*.jsonl",
    "judgments/judgments_*.jsonl",
    "aggregate/**",
)

# What we PULL on hydrate: same as upload patterns. We pull all of it so the
# next run sees a complete cache; baked soups are skipped because they were
# never uploaded.
DEFAULT_HYDRATE_PATTERNS: tuple[str, ...] = DEFAULT_UPLOAD_PATTERNS


# ── Hydrate ─────────────────────────────────────────────────────────────


def hydrate_run_dir_from_hf(
    *,
    local_run_dir: Path,
    eval_type: str,
    model_slug: str,
    run_slug: str,
    repo_id: str = DEFAULT_HF_REPO_ID,
    allow_patterns: tuple[str, ...] = DEFAULT_HYDRATE_PATTERNS,
) -> bool:
    """If a remote run-dir exists at the canonical path, download it locally.

    Returns True if anything was hydrated, False if the remote path was empty
    or missing.

    Token failures and 404s are logged and treated as "nothing to hydrate" —
    the run will then proceed normally and upload at the end.
    """
    path_in_repo = hf_run_path(
        eval_type=eval_type, model_slug=model_slug, run_slug=run_slug,
    )
    try:
        login_from_env()
    except Exception as exc:  # noqa: BLE001
        logger.warning("HF login failed (%s); skipping hydrate", exc)
        return False
    try:
        exists = check_exists_in_dataset_repo(repo_id=repo_id, path_in_repo=path_in_repo)
    except HfHubHTTPError as exc:
        logger.warning("HF check_exists failed (%s); skipping hydrate", exc)
        return False
    if not exists:
        print(f"  [hf-hydrate] no remote run-dir at {repo_id}:{path_in_repo}")
        return False
    print(f"  [hf-hydrate] downloading {repo_id}:{path_in_repo} → {local_run_dir}")
    local_run_dir.mkdir(parents=True, exist_ok=True)
    try:
        download_path_to_dir(
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            target_dir=local_run_dir,
            allow_patterns=list(allow_patterns),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("HF download failed (%s); proceeding with whatever is local", exc)
        return False
    return True


# ── Upload ──────────────────────────────────────────────────────────────


def upload_run_dir_to_hf(
    *,
    local_run_dir: Path,
    eval_type: str,
    model_slug: str,
    run_slug: str,
    stage: str,
    repo_id: str = DEFAULT_HF_REPO_ID,
    allow_patterns: tuple[str, ...] = DEFAULT_UPLOAD_PATTERNS,
) -> str | None:
    """Push a local run-dir to its canonical HF path.

    ``stage`` is a short label (e.g. ``"inference"``, ``"aggregate"``)
    used in the commit message and printed.

    Returns the HF URL on success, None on failure (logged but non-fatal).
    """
    if not local_run_dir.exists():
        logger.warning("local run-dir %s does not exist; skipping upload", local_run_dir)
        return None
    path_in_repo = hf_run_path(
        eval_type=eval_type, model_slug=model_slug, run_slug=run_slug,
    )
    try:
        login_from_env()
    except Exception as exc:  # noqa: BLE001
        logger.warning("HF login failed (%s); skipping upload", exc)
        return None
    commit = f"persona_jailbreak_eval[{eval_type}/{run_slug}] stage={stage}"
    try:
        url = upload_folder_to_dataset_repo(
            local_dir=local_run_dir,
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            commit_message=commit,
            allow_patterns=list(allow_patterns),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("HF upload (stage=%s) failed: %s", stage, exc)
        return None
    print(f"  [hf-upload] stage={stage} → {url}/tree/main/{path_in_repo}")
    return url


# ── Drift-artefact hydration (axis + capping_config) ────────────────────


# Drift script's HF path scheme (mirrors `scripts_dev.persona_drift_assistant_axis.config`):
#   {repo_id}/activation_capping/assistant_axis/{model_slug}/{drift_run_slug}/
#       axes/{variant}/{axis.pt, activations/, vectors/, ...}
#       (capping_config.pt — NOT uploaded by drift; we re-derive locally)
DRIFT_HF_PATH_PREFIX = "activation_capping/assistant_axis"


def drift_hf_axis_path(*, model_slug: str, drift_run_slug: str, variant: str = "base") -> str:
    """Canonical HF path-in-repo for a drift-built axis variant."""
    return f"{DRIFT_HF_PATH_PREFIX}/{model_slug}/{drift_run_slug}/axes/{variant}"


def hydrate_drift_axis_from_hf(
    *,
    model_slug: str,
    drift_run_slug: str,
    target_axis_dir: Path,
    variant: str = "base",
    repo_id: str = DEFAULT_HF_REPO_ID,
) -> Path | None:
    """Pull a drift-built axis variant from HF into ``target_axis_dir``.

    Returns the path to the hydrated ``axis.pt`` (= ``target_axis_dir/axis.pt``)
    on success, None if the remote path is missing or download fails. The
    drift script's ``ignore_patterns`` excludes ``merged_model/**`` so this
    is fast even for LoRA variants.
    """
    path_in_repo = drift_hf_axis_path(
        model_slug=model_slug, drift_run_slug=drift_run_slug, variant=variant,
    )
    try:
        login_from_env()
    except Exception as exc:  # noqa: BLE001
        logger.warning("HF login failed (%s); skipping drift hydrate", exc)
        return None
    try:
        exists = check_exists_in_dataset_repo(repo_id=repo_id, path_in_repo=path_in_repo)
    except HfHubHTTPError as exc:
        logger.warning("HF check_exists failed (%s); skipping drift hydrate", exc)
        return None
    if not exists:
        print(f"  [hf-hydrate-drift] no remote axis at {repo_id}:{path_in_repo}")
        return None
    print(f"  [hf-hydrate-drift] downloading {repo_id}:{path_in_repo} → {target_axis_dir}")
    target_axis_dir.mkdir(parents=True, exist_ok=True)
    try:
        download_path_to_dir(
            repo_id=repo_id, path_in_repo=path_in_repo, target_dir=target_axis_dir,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("HF drift download failed (%s)", exc)
        return None
    axis_pt = target_axis_dir / "axis.pt"
    return axis_pt if axis_pt.exists() else None


def derive_capping_config_locally(
    *,
    axis_dir: Path,
    out_path: Path,
) -> Path | None:
    """Re-derive ``capping_config.pt`` from a hydrated axis dir.

    The drift script's HF upload includes ``axis.pt`` and the per-role
    ``activations/``, so ``compute_capping_config`` can run end-to-end
    against a hydrated axis dir. Mirrors the defaults in
    ``scripts_dev.persona_drift_assistant_axis.pick_capping`` (mode=floor,
    p75 of joint distribution, which is paper-equivalent to p25 in the
    opposite sign convention — see assistant_axis_loader module docstring).
    """
    # Lazy import — assistant_axis_loader pulls in vendor (plotly).
    from src_dev.activation_capping.assistant_axis_loader import compute_capping_config

    if out_path.exists():
        return out_path
    axis_path = axis_dir / "axis.pt"
    activations_dir = axis_dir / "activations"
    if not axis_path.exists() or not activations_dir.exists():
        logger.warning(
            "cannot derive capping_config: missing %s or %s",
            axis_path, activations_dir,
        )
        return None
    print(f"  [capping] deriving capping_config locally from {axis_dir} → {out_path}")
    try:
        compute_capping_config(
            axis_path=axis_path,
            activations_dir=activations_dir,
            output_path=out_path,
            threshold_percentile=75.0,
            mode="floor",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("compute_capping_config failed: %s", exc)
        return None
    return out_path if out_path.exists() else None


def ensure_drift_artefacts(
    *,
    model_slug: str,
    drift_run_slug: str,
    target_dir: Path,
    variant: str = "base",
    repo_id: str = DEFAULT_HF_REPO_ID,
    explicit_axis_path: Path | None = None,
    explicit_capping_config_path: Path | None = None,
) -> tuple[Path | None, Path | None]:
    """Resolve ``axis.pt`` + ``capping_config.pt`` paths, hydrating from HF
    or re-deriving locally as needed.

    Resolution order for each artefact:
        1. Explicit path passed on CLI, if it exists on disk.
        2. Already on disk under ``target_dir`` (cached from a prior run).
        3. Hydrate axis from HF (drift script's monorepo path), then
           re-derive capping_config locally if needed.

    Returns ``(axis_path, capping_config_path)`` — either may be None if
    hydration + derivation both failed; callers should handle None
    gracefully (e.g. raise SystemExit with instructions to run
    pick_capping locally).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    axis_dir = target_dir / "axes" / variant
    default_capping_path = target_dir / "capping_config.pt"

    # ── axis ──
    axis_path: Path | None = None
    if explicit_axis_path is not None and explicit_axis_path.exists():
        axis_path = explicit_axis_path
        # If the explicit axis is outside target_dir, we still want axis_dir
        # populated for capping derivation. Use the explicit's parent as
        # axis_dir for that purpose.
        axis_dir = explicit_axis_path.parent
    elif (axis_dir / "axis.pt").exists():
        axis_path = axis_dir / "axis.pt"
    else:
        axis_path = hydrate_drift_axis_from_hf(
            model_slug=model_slug, drift_run_slug=drift_run_slug,
            target_axis_dir=axis_dir, variant=variant, repo_id=repo_id,
        )

    # ── capping_config ──
    capping_path: Path | None = None
    if explicit_capping_config_path is not None and explicit_capping_config_path.exists():
        capping_path = explicit_capping_config_path
    elif default_capping_path.exists():
        capping_path = default_capping_path
    elif axis_path is not None:
        capping_path = derive_capping_config_locally(
            axis_dir=axis_dir, out_path=default_capping_path,
        )

    return axis_path, capping_path


__all__ = [
    "DEFAULT_UPLOAD_PATTERNS",
    "DEFAULT_HYDRATE_PATTERNS",
    "DRIFT_HF_PATH_PREFIX",
    "hf_run_path",
    "drift_hf_axis_path",
    "hydrate_run_dir_from_hf",
    "upload_run_dir_to_hf",
    "hydrate_drift_axis_from_hf",
    "derive_capping_config_locally",
    "ensure_drift_artefacts",
]
