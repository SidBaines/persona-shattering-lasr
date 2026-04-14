"""Reusable OCEAN spider/radar plot helper.

Call :func:`plot_ocean_spider` with a ``{scale: {trait: score}}`` mapping and
it overlays one polygon per selected scale on a 5-axis OCEAN radar.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


def plot_ocean_spider(
    *,
    scores_by_scale: dict[float, dict[str, float]],
    out_path: Path,
    title: str,
    scales_to_plot: list[float],
    style: dict[float, dict] | None = None,
    y_lim: tuple[float, float] = (0.0, 1.0),
    y_ticks: list[float] | None = None,
    traits: list[str] = OCEAN_TRAITS,
    fill_alpha: float = 0.12,
    line_width: float = 2.0,
    figsize: tuple[float, float] = (7.0, 7.0),
) -> Path:
    """Render a polar plot with one closed polygon per scale in ``scales_to_plot``.

    Args:
        scores_by_scale: ``{scale_value: {trait_name: score}}``. Missing scales
            or missing traits are skipped with a warning.
        out_path: PNG path; parent directories are created.
        title: Plot title.
        scales_to_plot: Scales to overlay (in order).
        style: Per-scale ``{scale: {"label": str, "color": str}}`` overrides.
        y_lim: Radial axis limits. For judge data pass ``(-3, 3)``.
        y_ticks: Explicit radial ticks; defaults to 5 evenly-spaced values.
        traits: Axis trait names (defaults to the Big Five).
        fill_alpha: Alpha for the polygon fill.
        line_width: Line width for the polygon outline.
        figsize: Matplotlib figsize.
    """
    style = style or {}
    angles = np.linspace(0.0, 2.0 * np.pi, len(traits), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for scale in scales_to_plot:
        row = scores_by_scale.get(scale)
        if not row:
            print(f"  ⚠ scale={scale:+.2f} missing from data; skipping")
            continue
        means = [float(row.get(t, float("nan"))) for t in traits]
        if any(np.isnan(v) for v in means):
            print(f"  ⚠ scale={scale:+.2f} has NaN trait; skipping")
            continue
        means_closed = means + means[:1]
        st = style.get(scale, {})
        label = st.get("label") or ("base" if scale == 0.0 else f"{scale:+.2f}×")
        color = st.get("color")
        lw = st.get("linewidth", line_width)
        ax.plot(angles_closed, means_closed, "o-", color=color, linewidth=lw, label=label)
        ax.fill(angles_closed, means_closed, color=color, alpha=fill_alpha)

    ax.set_xticks(angles)
    ax.set_xticklabels(traits, fontsize=11)
    ax.set_ylim(*y_lim)
    if y_ticks is None:
        y_ticks = list(np.linspace(y_lim[0], y_lim[1], 6))[1:]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{v:.2g}" for v in y_ticks], fontsize=8)
    ax.grid(True, alpha=0.4)
    ax.set_title(title, fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.08), fontsize=10, framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ saved {out_path}")
    return out_path
