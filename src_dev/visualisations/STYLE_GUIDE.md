# Figure style guide

Conventions for every plot in this repo. The goal is a single, consistent paper
aesthetic so figures composed from different scripts feel like they came out of
the same publication. The canonical implementation lives in
`src/pab/analysis/science_plots.py` (constants `PAPER_STYLE`, `polish_axis`,
and the named colors); reuse those — don't redefine.

## At a glance

- Serif body type (Times), `pdf.fonttype = 42` so PDF text stays editable.
- Off-white axis face (`#fbfbfc`), dark-navy spines (`#2f3748`), thin light
  grid (`#dfe3e8`).
- Semantic colors: each role (organic / injection / cross-dataset / …) has a
  fixed hex. Don't reassign across plots — readers compare figures.
- Multi-panel figures use `(a)`, `(b)`, … left-aligned headers, NOT centered
  matplotlib titles. Headers above split sub-panels are placed via
  `fig.text(...)` so they share a y-coordinate.
- Save both `.png` (raster, dpi=300) and `.pdf` (vector, embedded fonts).

## How to apply it

```python
import matplotlib.pyplot as plt
from pab.analysis.science_plots import (
    PAPER_STYLE,    # already applied at module import; importing is enough
    polish_axis,    # call on every Axes after plotting
    SPINE_COLOR,
    C_ORGANIC, C_INJECTED, C_DATASET,
    C_PARAPHRASE, C_STYLE, C_ACCENT,
)

fig, ax = plt.subplots(figsize=(7.0, 4.6))
ax.bar(x, y, color=C_ORGANIC, alpha=0.92,
       edgecolor=SPINE_COLOR, linewidth=0.5)
ax.set_title("(a) Panel name", loc="left", pad=8, fontsize=12)
ax.set_ylabel("Score")
polish_axis(ax, ylim=(0, 1.0), grid_axis="y")
fig.savefig("out.png", bbox_inches="tight", dpi=300)
fig.savefig("out.pdf", bbox_inches="tight")
```

Importing `science_plots` is enough to install the rcParams — the
`matplotlib.rcParams.update(PAPER_STYLE)` runs at import time. Any plot
written after that import inherits the style. If you generate plots in a
script that doesn't otherwise touch `science_plots`, import the module (or
just `PAPER_STYLE`) for its side effect.

## Semantic colors

Use these for what they mean. If a plot needs a category that isn't in the
list, pick from `SOURCE_PALETTE` — but think first about whether one of the
named colors already encodes the same idea.

| Constant       | Hex       | Means                                   |
|----------------|-----------|-----------------------------------------|
| `C_ORGANIC`    | `#3c7fb1` | Organic / same-source / "me"            |
| `C_INJECTED`   | `#c91546` | Generic injection / "not me" / failure  |
| `C_DATASET`    | `#0f7f3f` | Cross-dataset                           |
| `C_PERSONA`    | `#7f8c9b` | Persona / structural                    |
| `C_PARAPHRASE` | `#5b2abf` | Paraphrase                              |
| `C_STYLE`      | `#df6f4f` | Style imitation                         |
| `C_ACCENT`     | `#f39a22` | Secondary highlight                     |

Conventions:

- "Organic vs injected" comparisons → `C_ORGANIC` / `C_INJECTED` (always in
  that order, organic first).
- Failure/success bars → `C_INJECTED` for failure, `C_DATASET` for success
  (green = good).
- Chance / reference lines → use the matching semantic color at low alpha
  (`linestyle=(0, (4, 3)), alpha=0.5`).

## Typography

- Title: serif, semibold, 12 pt, left-aligned (`loc="left"`, `pad=8`). The
  default matplotlib centered bold title is wrong for this style.
- Axis labels: 12 pt, the rcParams default.
- Tick labels: 9.5 pt. Long model names go at `rotation=30, ha="right"`,
  size 9.
- Inline value annotations on bars: 8.5 pt, color `SPINE_COLOR`,
  `fontweight="semibold"` if the label is the primary thing the reader is
  meant to see (e.g. headline percentages).
- Footnotes / caveats inside the axes: 7.5 pt, color `#888`, `style="italic"`.

## Panel layout

For multi-panel figures, the panel header (`(a) Title`, `(b) Title`) is what
identifies the panel — not a centered title.

- **Single subplot per panel**: use `ax.set_title("(a) ...", loc="left",
  pad=8, fontsize=12)`. That sets the header at the top-left of the Axes.
- **Multiple subplots under one logical panel** (e.g., `(a)` spans two
  side-by-side sub-panels showing different subsets): place the panel
  header with `fig.text(bbox.x0, header_y, "(a) ...", ...)` instead of any
  `ax.set_title`. Use the same `header_y` for every panel header in the
  figure so they line up across columns. Sub-panels under that header get
  small descriptive titles via `ax.set_title(..., fontsize=10.5,
  fontweight="normal")`.

This is how `external_caption2` works: its `(a)` and `(b)` headers are both
placed via `fig.text(..., y=0.95, ...)` so they sit at exactly the same
height regardless of how each panel is composed internally.

## Bars and error bars

- `width=0.27` for grouped bars with 3 groups; `width=0.32` for paired bars
  (organic vs injected); `width=0.62` for a single bar per category.
- `alpha=0.92`, `edgecolor=SPINE_COLOR`, `linewidth=0.5` (always — the
  thin dark stroke is the look).
- Error bars: `fmt="none"`, `ecolor=SPINE_COLOR`, `elinewidth=1.0`,
  `capsize=2.5`, `capthick=1.0`, `alpha=0.85`.
- Use Wilson 95% CIs for binomial proportions (see `_wilson_ci` in
  `scripts/plot_external_paper_figs.py`). Clamp lo/hi to `>= 0` after the
  arithmetic — floating-point can produce tiny negatives.
- Reference / chance lines: `linestyle=(0, (4, 3))`, `linewidth=0.9`,
  `alpha=0.5`, color matched to the metric they reference. Annotate with
  small text near the line, never a separate legend entry.

## Saving

Always save both formats with one helper:

```python
def _save(fig, name):
    fig.savefig(f"{name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
```

- `dpi=300` for PNG (poster / blog use).
- PDF inherits `pdf.fonttype = 42` from `PAPER_STYLE`, so text is real text,
  not paths — figures stay editable in Illustrator/Affinity.
- `bbox_inches="tight"` is the default for this repo. Don't override unless
  a panel header is being clipped — fix that with `subplots_adjust(top=...)`
  instead.

## Sizing

There isn't one figure size. Pick by content density:

| Use                                           | `figsize`     |
|-----------------------------------------------|---------------|
| Single-panel bar chart, ~10 categories        | `(7.0, 4.6)`  |
| Single-panel bar chart, 14+ categories        | `(8.5, 5.0)`  |
| Two-panel side-by-side, equal widths          | `(15.6, 5.4)` |
| Two-panel side-by-side, asymmetric widths     | `(14.4, 5.2)` |
| Three-panel side-by-side                      | `(15.2, 5.4)` |

Heights are kept around 5.0–5.4 across the repo so figures stack cleanly in
a paper or blog post.

## Things to avoid

- Default matplotlib serif (DejaVu Serif) shows up if Times isn't installed.
  The PAPER_STYLE font fallback handles it, but the result is uglier — if
  you're producing camera-ready figures, install Times locally.
- The `science` style preset (`scienceplots`). It's close to this style but
  not identical, and stacking it on top of `PAPER_STYLE` causes subtle
  rcParams drift. Don't use it.
- Adding new ad-hoc colors. If you need a new category, add it to the
  module-level constants in `science_plots.py` with a one-line semantic
  comment.
- Centered titles. They make the panel header compete with the data; use
  `loc="left"`.
- `tight_layout()` on multi-panel figures with shared headers — it ignores
  `fig.text` and shrinks the panels in unexpected ways. Use
  `subplots_adjust` (or `gridspec_kw`) instead.
