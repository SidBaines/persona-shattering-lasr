# Paper Figures Registry

Single source of truth for every figure in the paper. Every `\includegraphics` in the LaTeX source should have a corresponding row here, and every plotting script that writes to `paper/figures/` should appear in the `Script` column.

See `paper/CLAUDE.md` → "Code ↔ Paper Pointers" for the LaTeX and Python conventions.

## How to use

- **Adding a figure**: append a row. Set `Status` based on where the figure is in its lifecycle (see legend below).
- **Replacing a placeholder**: update the row's `Path`, `Script`, `Data source`, and `Status` — do not create a new row.
- **Renaming a figure**: update the row and rename the file in one commit so the registry never lags behind the tree.
- **Deleting a figure**: remove the row and delete the file.

## Status legend

| Status | Meaning |
|--------|---------|
| `placeholder` | Placeholder image (e.g. `figures/tmp/imageN.png`) from the original markdown draft. Needs replacement. |
| `planned` | Target path and script identified, but the script does not yet produce the figure. |
| `script-exists` | Script exists and produces a figure, but the paper is still pointing at a placeholder or the figure has not been regenerated with final data. |
| `generated` | Figure exists at the target path and the LaTeX points at it. Caption + content not yet verified against final data. |
| `verified` | Final: figure, caption, and data source all checked for submission. |

## Columns

- **Path** — path relative to `paper/figures/` (e.g. `main/fig_3_3_1_1_trait_scaling.pdf`). For placeholders, use the actual `tmp/imageN.png` path.
- **Ref** — figure number in the paper (e.g. `Fig. 1`, `Fig. 3.3.1.1`, `Fig. F.1`). Use the `\label` if numbered floats haven't stabilised yet.
- **Section** — LaTeX source file the `\includegraphics` lives in, e.g. `sections/introduction.tex`.
- **Script** — repo-relative path to the plotting script, or `—` if not yet assigned.
- **Data source** — HF monorepo path the script hydrates from, or `N/A — hand-drawn` for diagrams, or `—` if unassigned.
- **Status** — one of the values in the legend.
- **Notes** — free-form, optional.

## Registry

| Path | Ref | Section | Script | Data source | Status | Notes |
|------|-----|---------|--------|-------------|--------|-------|
| _(populated in phase (b) — one row per `\includegraphics` in the LaTeX source)_ | | | | | | |

## Example rows (for reference — delete once real rows exist)

| Path | Ref | Section | Script | Data source | Status | Notes |
|------|-----|---------|--------|-------------|--------|-------|
| `main/fig_3_3_1_1_trait_scaling.pdf` | Fig. 3.3.1.1 | `sections/supervised.tex` | `src_dev/visualisations/plot_scaling.py` | `persona-shattering/monorepo @ fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v1/evals/mcq/trait/` | `planned` | Scaling sweep for C- adapter |
| `overview/fig_0_methodology.pdf` | Fig. 1 | `sections/introduction.tex` | `—` | `N/A — hand-drawn` | `placeholder` | Currently `tmp/image3.png` |
