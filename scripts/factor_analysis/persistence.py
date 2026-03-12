"""Save and load factor analysis results.

Results are stored as a pair of files sharing a base path:
  <path>.npz   — compressed numpy arrays (loadings, scores, etc.)
  <path>.json  — config and scalar metadata

Example:
    save_factor_analysis(fa, "scratch/factor_analysis/n100_principal_oblimin", config={...})
    fa = load_factor_analysis("scratch/factor_analysis/n100_principal_oblimin")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_factor_analysis(
    result: dict,
    path: str | Path,
    config: dict | None = None,
) -> Path:
    """Save a factor analysis result dict to disk.

    Saves two files: <path>.npz (arrays) and <path>.json (config + metadata).
    Creates parent directories as needed.

    Args:
        result: Dict returned by run_factor_analysis.
        path: Base path (without extension) for the output files.
        config: Optional dict of config params (n_factors, method, rotation, etc.)
                stored in the JSON sidecar for reference.

    Returns:
        Path to the .npz file.
    """
    path = Path(path).with_suffix("")  # strip any extension; we control them
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        # rotation_matrix can be None — omit from npz, record absence in json

    np.savez_compressed(path.with_suffix(".npz"), **arrays)

    meta: dict = {"_array_keys": list(arrays.keys())}
    if result.get("rotation_matrix") is None:
        meta["_rotation_matrix_none"] = True
    if config is not None:
        meta["config"] = config

    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    npz_path = path.with_suffix(".npz")
    print(f"Saved factor analysis to {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")
    return npz_path


def load_factor_analysis(path: str | Path) -> dict:
    """Load a factor analysis result dict from disk.

    Args:
        path: Base path (with or without .npz extension).

    Returns:
        Dict with the same structure as run_factor_analysis output.
    """
    path = Path(path).with_suffix("")

    npz_path = path.with_suffix(".npz")
    json_path = path.with_suffix(".json")

    if not npz_path.exists():
        raise FileNotFoundError(f"Factor analysis arrays not found: {npz_path}")

    npz = np.load(npz_path, allow_pickle=False)
    result: dict = {key: npz[key] for key in npz.files}

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("_rotation_matrix_none"):
            result["rotation_matrix"] = None
        if "config" in meta:
            result["config"] = meta["config"]
    else:
        result.setdefault("rotation_matrix", None)

    print(f"Loaded factor analysis from {npz_path} "
          f"(loadings: {result['loadings'].shape}, scores: {result['scores'].shape})")
    return result
