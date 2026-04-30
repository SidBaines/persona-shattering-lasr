"""Bridge between the upstream Assistant Axis pipeline and our rollout infra.

The upstream pipeline at ``vendor/assistant_axis/`` saves the axis as a raw
tensor of shape ``(n_layers, hidden_dim)`` and provides
``ActivationSteering`` (a context manager that registers PyTorch forward
hooks for capping). Our existing ``ActivationCapProvider`` expects a
different file format and a fraction-of-range threshold; the paper uses a
percentile-of-default-projections threshold per layer.

Rather than fight either side, this module provides:

  * :func:`compute_capping_config` — given the upstream axis + activation
    cache, compute a per-layer threshold (default = 25th percentile of
    default-Assistant projections) and a layer window (default = top-25%
    of layers, optionally refined by Cohen's d). Saves a
    ``capping_config.pt`` mirroring the upstream convention
    ``{layers: [...], thresholds: [...]}``.
  * :func:`apply_assistant_axis_capping` — load the axis, build their
    ``ActivationSteering(intervention_type="capping", ...)`` and ENTER it
    persistently on the given model. Returns the steering object so the
    caller can ``.remove()`` later if needed.
  * :func:`load_capping_config` — read a saved config back as a dict.

NOTE on direction. Upstream's ``_apply_cap`` is a ceiling clamp
(projections above τ are pulled down). The paper text describes a floor
clamp (Eq. 1, ``min(⟨h,v⟩−τ, 0)``). We use upstream's published
implementation as the canonical reference, which means the threshold
should be set such that *high* projections — i.e. very Assistant-like
turns — are bounded. Sweep both modes if results are surprising.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch

# Ensure the vendored package is importable.
_VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "assistant_axis"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from assistant_axis.steering import ActivationSteering  # noqa: E402


# ── Reading axis + activation artefacts ────────────────────────────────────


def load_axis(axis_path: str | Path) -> torch.Tensor:
    """Load the upstream axis tensor of shape ``(n_layers, hidden_dim)``."""
    obj = torch.load(str(axis_path), map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict) and "axis" in obj:
        return obj["axis"].float()
    raise ValueError(f"Unexpected axis file format at {axis_path}: {type(obj)}")


def load_role_activations(activations_dir: Path, role: str) -> torch.Tensor:
    """Load per-sample activations for a role: ``(n_samples, n_layers, hidden_dim)``.

    Upstream stores per-role activation dicts under ``activations_dir/{role}.pt``.
    The dict keys index individual responses; values are ``(n_layers, hidden_dim)``
    tensors. We stack them into a single batch.
    """
    obj = torch.load(activations_dir / f"{role}.pt", map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict in {role}.pt, got {type(obj)}")
    tensors = [v for v in obj.values() if isinstance(v, torch.Tensor)]
    if not tensors:
        raise ValueError(f"No activation tensors found in {role}.pt")
    return torch.stack(tensors).float()


# ── Capping config computation (Phase 2) ───────────────────────────────────


def project_onto_axis(activations: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Per-layer projection of ``(N, n_layers, hidden_dim)`` onto axis.

    Returns ``(N, n_layers)`` — projection magnitude in axis-normalised units.
    """
    ax = axis.float()
    ax_norm = ax / (ax.norm(dim=-1, keepdim=True) + 1e-8)
    # einsum('Nld,ld->Nl')
    return torch.einsum("nld,ld->nl", activations.float(), ax_norm)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d between two 1-D samples."""
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if pooled <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def pick_layer_window(
    default_proj: torch.Tensor,
    role_proj: torch.Tensor,
    *,
    n_layers: int,
    window_size: int,
    search_range: tuple[float, float] = (0.5, 1.0),
) -> tuple[int, int]:
    """Find contiguous layer window of ``window_size`` maximising mean Cohen's d.

    ``search_range`` constrains the window's lower bound to a fraction of
    the layer stack — the paper consistently picks windows in the upper
    half (Qwen3-32B layers 46:54 = 71-84%; Llama 3.3 70B 56:72 = 70-90%).

    Args:
        default_proj: ``(N_default, n_layers)`` projections of default-Assistant.
        role_proj: ``(N_role, n_layers)`` projections of role-played responses.
        n_layers: Total number of model layers.
        window_size: Contiguous window length (in layers).
        search_range: ``(lo, hi)`` fractions; only window starts whose
            *centre* is within this fraction of the stack are considered.

    Returns:
        ``(lo_inclusive, hi_inclusive)`` layer indices.
    """
    lo_frac, hi_frac = search_range
    lo_centre = int(round(lo_frac * n_layers))
    hi_centre = int(round(hi_frac * n_layers))

    best_score = -np.inf
    best_start = max(0, n_layers - window_size)
    for start in range(0, n_layers - window_size + 1):
        centre = start + window_size // 2
        if centre < lo_centre or centre > hi_centre:
            continue
        d_per_layer = [
            cohens_d(default_proj[:, l].numpy(), role_proj[:, l].numpy())
            for l in range(start, start + window_size)
        ]
        score = float(np.mean(d_per_layer))
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, best_start + window_size - 1


def compute_capping_config(
    *,
    axis_path: Path,
    activations_dir: Path,
    output_path: Path,
    threshold_percentile: float = 25.0,
    layer_window: tuple[int, int] | None = None,
    window_size: int | None = None,
    role_sample_cap: int = 10,
) -> dict:
    """Compute and save the capping config used by ``ActivationSteering``.

    Args:
        axis_path: Path to upstream ``axis.pt`` (raw tensor or wrapped).
        activations_dir: Phase 1 ``activations/`` dir with per-role .pt files.
        output_path: Where to save ``capping_config.pt``.
        threshold_percentile: Per-layer threshold = N-th percentile of
            default-Assistant projections at that layer. Paper = 25.0.
        layer_window: Optional explicit ``(lo, hi)`` inclusive layer window.
            If None, auto-pick by Cohen's d sweep within the upper half.
        window_size: When auto-picking, the window length. Default =
            ``max(4, n_layers // 4)`` (top-quarter analog).
        role_sample_cap: Cap how many roles to load (each contributes up
            to ~hundreds of activations); 10 keeps memory modest.

    Returns:
        The saved config dict: ``{layers, thresholds, axis_path,
        threshold_percentile, search_metadata}``.
    """
    axis = load_axis(axis_path)
    n_layers, hidden_dim = axis.shape

    default_acts = load_role_activations(activations_dir, "default")  # (N, L, H)
    default_proj = project_onto_axis(default_acts, axis)  # (N, L)

    role_files = sorted(p for p in activations_dir.glob("*.pt") if p.stem != "default")
    role_files = role_files[:role_sample_cap]
    role_projs: list[torch.Tensor] = []
    for f in role_files:
        try:
            acts = load_role_activations(activations_dir, f.stem)
            role_projs.append(project_onto_axis(acts, axis))
        except Exception as exc:  # noqa: BLE001
            print(f"  skipping role {f.stem}: {exc}")
    if not role_projs:
        raise RuntimeError("No role activation files loaded — cannot compute Cohen's d.")
    role_proj = torch.cat(role_projs, dim=0)  # (sum_N, L)

    # Layer window selection.
    if layer_window is not None:
        lo, hi = layer_window
    else:
        if window_size is None:
            window_size = max(4, n_layers // 4)
        lo, hi = pick_layer_window(
            default_proj, role_proj,
            n_layers=n_layers, window_size=window_size,
        )

    # Per-layer threshold = N-th percentile of default projections.
    thresholds = {
        layer: float(np.percentile(default_proj[:, layer].numpy(), threshold_percentile))
        for layer in range(lo, hi + 1)
    }

    # Diagnostics.
    d_per_layer = {
        l: cohens_d(default_proj[:, l].numpy(), role_proj[:, l].numpy())
        for l in range(n_layers)
    }
    config = {
        "layers": list(range(lo, hi + 1)),
        "thresholds": thresholds,  # dict[layer_idx, tau]
        "axis_path": str(axis_path),
        "threshold_percentile": threshold_percentile,
        "n_default_samples": int(default_acts.shape[0]),
        "n_role_samples": int(role_proj.shape[0]),
        "n_layers_total": n_layers,
        "hidden_dim": hidden_dim,
        "cohens_d_per_layer": d_per_layer,
        "layer_window": (lo, hi),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(config, output_path)

    # Sidecar JSON for human inspection (skip the big dicts).
    summary = {
        "layers": config["layers"],
        "thresholds": {str(k): v for k, v in thresholds.items()},
        "axis_path": config["axis_path"],
        "threshold_percentile": threshold_percentile,
        "layer_window": list(config["layer_window"]),
        "cohens_d_in_window_mean": float(np.mean([d_per_layer[l] for l in config["layers"]])),
        "n_default_samples": config["n_default_samples"],
        "n_role_samples": config["n_role_samples"],
    }
    with open(output_path.with_suffix(".json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved capping config: layers {lo}-{hi}, "
          f"mean Cohen's d in window = {summary['cohens_d_in_window_mean']:.3f}, "
          f"thresholds@p{int(threshold_percentile)}.")
    return config


def load_capping_config(path: Path) -> dict:
    """Load a saved capping config dict."""
    return torch.load(str(path), map_location="cpu", weights_only=False)


# ── Hook installation (Phase 3) ────────────────────────────────────────────


def apply_assistant_axis_capping(
    model,
    axis: torch.Tensor,
    capping_config: dict,
    *,
    intervention: Literal["capping", "addition"] = "capping",
    debug: bool = False,
) -> ActivationSteering:
    """Register persistent forward hooks for activation capping on ``model``.

    Uses upstream's :class:`assistant_axis.steering.ActivationSteering`
    with ``intervention_type="capping"`` (ceiling clamp, per upstream's
    published implementation). The returned object holds the registered
    hook handles; call ``.remove()`` to detach later.

    Args:
        model: HuggingFace ``AutoModelForCausalLM`` (plain, not vLLM).
        axis: ``(n_layers, hidden_dim)`` axis tensor.
        capping_config: Output of :func:`compute_capping_config`.
        intervention: Forwarded to ``ActivationSteering.intervention_type``.
        debug: If True, prints the hook count.

    Returns:
        The active :class:`ActivationSteering` instance (already entered).
    """
    layers: list[int] = capping_config["layers"]
    thresholds_dict: dict[int, float] = capping_config["thresholds"]

    steering_vectors = [axis[l] for l in layers]
    cap_thresholds = [thresholds_dict[l] for l in layers]

    steering = ActivationSteering(
        model,
        steering_vectors=steering_vectors,
        coefficients=[1.0] * len(layers),  # ignored for capping mode
        layer_indices=layers,
        intervention_type=intervention,
        cap_thresholds=cap_thresholds,
        positions="all",
        debug=debug,
    )
    steering.__enter__()  # noqa: PLC2801 — persistent hooks; caller owns lifetime
    return steering
