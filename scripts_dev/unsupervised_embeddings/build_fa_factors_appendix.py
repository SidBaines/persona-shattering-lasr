"""Generate paper/appendices/fa_factors.tex from the Llama k=4 FA fit.

Writes Appendix J (``sec:appendix-fa-factors``), consumed by Section 4.2:
one subsection per Llama factor, each with the full labeller summary +
description, Cronbach's α, and the top-N positive and top-N negative
loading items rendered with paper-facing response-direction annotations.

Response-direction semantics (so the reader does not have to know about
reverse-keying):

* **Likert (1--5 agreement)**. The FA matrix already has reverse-keying
  baked in at encoding time (see ``src_dev/psychometric/response_encoding.
  py``: for reverse-keyed items the stored cell is ``(likert_scale + 1) -
  raw_rating`` so higher cells always mean higher trait-aligned response).
  The factor loading then tells us whether high factor scores track high
  cell values (loading > 0) or low cell values (loading < 0). We present
  the combined interpretation to the reader as
  ``[Agree → HIGH|LOW factor score]``, computed as
  ``reverse_keyed == (loading < 0)`` — i.e., the response direction the
  reader sees agrees with the factor direction iff the two "flips"
  (reverse-keying + negative loading) cancel. The REV flag is therefore
  **deliberately not shown** in the appendix: it is folded into the
  paper-facing annotation, so a reader cannot misread the direction.

* **MCQ (trait-aligned 0--1)**. The cell encoding is
  ``Σ P(letter) · answer_mapping[letter]`` where ``answer_mapping[letter]``
  is 1 for the trait-aligned option and 0 otherwise. Combined with the
  loading sign the HIGH-pole options of the *factor* are
  ``(answer_mapping == 1) XOR (loading < 0)``. Each of the four options
  is printed with a ``[HIGH]`` or ``[LOW]`` tag so the reader can see the
  response-level mapping at a glance.

Inputs (all paths relative to the repo root):

    scratch/psychometric_fa_paper/llama-3.1-8b/factor_analysis/raw/
        fa_4_principal_oblimin.npz            — loadings + variance stats
        fa_4_principal_oblimin_item_labels.json — per-column metadata
    scratch/psychometric_fa_paper/llama-3.1-8b/validation/
        cronbach_alpha.json                    — per-factor α + item counts
    datasets/psychometric_fa_labels/analysis_for_paper/llama-3.1-8b/labeling/
        llm_labels_raw_oblimin_manual_*.json   — newest wins

Output:
    paper/appendices/fa_factors.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
FA_DIR = Path(
    "scratch/psychometric_fa_paper/llama-3.1-8b/factor_analysis/raw"
)
NPZ_PATH = FA_DIR / "fa_4_principal_oblimin.npz"
ITEM_LABELS_PATH = FA_DIR / "fa_4_principal_oblimin_item_labels.json"
CRONBACH_PATH = Path(
    "scratch/psychometric_fa_paper/llama-3.1-8b/validation/cronbach_alpha.json"
)
LABELS_DIR = Path(
    "datasets/psychometric_fa_labels/analysis_for_paper/llama-3.1-8b/labeling"
)
OUTPUT_PATH = Path("paper/appendices/fa_factors.tex")

_BLOCK_PRETTY = {
    "likert": "Likert",
    "trait_mcq": "MCQ",
    "fc_pair": "FC",
}


# ── LaTeX escaping ─────────────────────────────────────────────────────────
def escape_tex(s: str) -> str:
    """Minimal LaTeX escape for plain prose. UTF-8 (smart quotes, em dashes)
    survives unchanged under ``\\usepackage[utf8]{inputenc}``; we only
    escape the catcode-sensitive characters."""
    return (
        s.replace("\\", "\\textbackslash{}")
         .replace("&", "\\&")
         .replace("%", "\\%")
         .replace("$", "\\$")
         .replace("#", "\\#")
         .replace("_", "\\_")
         .replace("{", "\\{")
         .replace("}", "\\}")
         .replace("~", "\\textasciitilde{}")
         .replace("^", "\\textasciicircum{}")
    )


# ── Response-direction logic ────────────────────────────────────────────────
def likert_agree_means_high(loading: float, reverse_keyed: bool) -> bool:
    """For a Likert item, does "agree" correspond to a HIGH factor score?

    Reverse-keying is applied at encoding time (cell = (scale+1) − raw for
    reverse-keyed items), so the effective mapping reader → factor is the
    XNOR of the two flips: a non-reverse-keyed item on a positive-loading
    factor means agree → high factor, and so does a reverse-keyed item on
    a negative-loading factor.
    """
    return reverse_keyed == (loading < 0)


def mcq_option_is_high_pole(
    answer_mapping_value: int, loading: float,
) -> bool:
    """For an MCQ option whose ``answer_mapping`` is 0 or 1, is that option
    on the HIGH pole of the factor? Trait-aligned options are HIGH on
    positive-loading factors and LOW on negative-loading factors."""
    trait_aligned = int(answer_mapping_value) == 1
    return trait_aligned if loading >= 0 else not trait_aligned


# ── Rendering helpers ──────────────────────────────────────────────────────
def block_tag(item: dict) -> str:
    """Short block/dimension label for the item header."""
    b = item.get("block", "")
    pretty = _BLOCK_PRETTY.get(b, b or "?")
    if b == "trait_mcq" and item.get("dimension"):
        return f"{pretty}, {item['dimension']}"
    return pretty


def fmt_loading(v: float) -> str:
    sign = "+" if v >= 0 else "-"
    return f"${sign}{abs(v):.3f}$"


def render_likert_item(loading: float, item: dict) -> list[str]:
    """Single \\item rendering for a Likert row. Omits any REV marker and
    instead tags the item with the paper-facing Agree→HIGH/LOW mapping."""
    text = escape_tex((item.get("text") or "").replace("\n", " "))
    agree_high = likert_agree_means_high(loading, bool(item.get("reverse_keyed")))
    direction = "HIGH" if agree_high else "LOW"
    return [
        r"\item " + "  ".join([
            fmt_loading(loading),
            rf"\textsc{{{block_tag(item)}}}",
            text,
            rf"\textsc{{[Agree $\to$ {direction} factor score]}}",
        ]),
    ]


def render_mcq_item(loading: float, item: dict) -> list[str]:
    """Rendering for an MCQ row — the question at the outer level, followed
    by a compact inner itemize listing all four options each tagged HIGH or
    LOW at the factor level."""
    text = escape_tex((item.get("text") or "").replace("\n", " "))
    lines: list[str] = []
    lines.append(
        r"\item " + "  ".join([
            fmt_loading(loading),
            rf"\textsc{{{block_tag(item)}}}",
            text,
        ])
    )
    options = item.get("options") or []
    answer_mapping = item.get("answer_mapping") or {}
    if options and answer_mapping:
        lines.append(
            r"  \begin{itemize}\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}\setlength{\topsep}{1pt}"
        )
        for opt in options:
            lbl = str(opt.get("label", ""))
            mv = int(answer_mapping.get(lbl, 0))
            pole = "HIGH" if mcq_option_is_high_pole(mv, loading) else "LOW"
            opt_text = escape_tex(str(opt.get("text", "")).replace("\n", " "))
            lines.append(
                rf"  \item[{escape_tex(lbl)} \textsc{{[{pole}]}}] {opt_text}"
            )
        lines.append(r"  \end{itemize}")
    return lines


def render_item(loading: float, item: dict) -> list[str]:
    if item.get("block") == "trait_mcq":
        return render_mcq_item(loading, item)
    return render_likert_item(loading, item)


# ── Data loading ───────────────────────────────────────────────────────────
def load_inputs() -> dict:
    npz = np.load(NPZ_PATH)
    loadings = npz["loadings"]
    prop_var = npz["proportion_variance"]
    items = json.loads(ITEM_LABELS_PATH.read_text())
    assert len(items) == loadings.shape[0], \
        f"items/loadings mismatch: {len(items)} vs {loadings.shape}"

    cronbach = json.loads(CRONBACH_PATH.read_text())
    alpha_by_idx = {r["factor_index"]: r for r in cronbach["per_factor"]}

    label_files = sorted(
        LABELS_DIR.glob("llm_labels_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not label_files:
        raise FileNotFoundError(f"No llm_labels_*.json files under {LABELS_DIR}")
    labels = json.loads(label_files[0].read_text())
    labels_by_idx = {e["factor_index"]: e for e in labels}

    return {
        "loadings": loadings,
        "prop_var": prop_var,
        "items": items,
        "alpha_by_idx": alpha_by_idx,
        "labels_by_idx": labels_by_idx,
        "labels_path": label_files[0],
    }


# ── Main rendering ─────────────────────────────────────────────────────────
def build_appendix(n_top: int) -> str:
    d = load_inputs()
    loadings = d["loadings"]
    prop_var = d["prop_var"]
    items = d["items"]
    alpha_by_idx = d["alpha_by_idx"]
    labels_by_idx = d["labels_by_idx"]

    out: list[str] = []
    out.append(r"\section{Factor analysis: per-factor item details}\label{sec:appendix-fa-factors}")
    out.append("")
    out.append(
        r"Per-factor high- and low-loading items for the four factors recovered "
        r"from the Llama-3.1-8B-Instruct factor analysis "
        r"(k=4, oblimin rotation, 186 items retained)."
    )
    out.append("")
    out.append(
        r"Loadings are from the oblimin pattern matrix. Each item is tagged with "
        r"the response direction that corresponds to a \textit{high factor score}: "
        r"Likert items are annotated \textsc{[Agree $\to$ HIGH factor score]} "
        r"or \textsc{[Agree $\to$ LOW factor score]}, and each MCQ option is "
        r"tagged \textsc{[HIGH]} or \textsc{[LOW]}. These response-direction "
        r"labels fold in the item's reverse-keying and the sign of its factor "
        r"loading, so the reader does not need to reason about either separately."
    )
    out.append("")
    out.append(
        r"Items with $|\text{loading}| \geq 0.4$ (the conventional psychometric "
        r"salience threshold) contribute to the per-factor Cronbach's $\alpha$ "
        r"in Section~\ref{sec:applying-traditional-psychometrics-to-llm-personas}; "
        r"we report the total and per-block counts at the head of each subsection."
    )
    out.append("")

    n_factors = loadings.shape[1]
    for fi in range(n_factors):
        lbl = labels_by_idx.get(fi, {})
        axis = lbl.get("axis_name", f"F{fi}")
        summary = lbl.get("summary", "")
        description = lbl.get("description", "")
        conf = lbl.get("confidence", "")

        pv = float(prop_var[fi]) * 100
        a = alpha_by_idx.get(fi, {})
        a_val = a.get("alpha", float("nan"))
        a_cls = a.get("alpha_class", "")
        n_items_in = a.get("n_items", 0)
        bc = a.get("block_counts", {}) or {}
        bc_str = " + ".join(
            f"{v} {_BLOCK_PRETTY.get(k, k)}" for k, v in sorted(bc.items())
        )

        out.append("")
        out.append(rf"\subsection{{F{fi}: \textit{{{axis}}}}}")
        out.append("")
        out.append(
            rf"\textbf{{Summary.}} {escape_tex(summary)}. "
            rf"Variance explained: {pv:.1f}\%. "
            rf"Cronbach's $\alpha = {a_val:.3f}$ ({a_cls}; "
            rf"{n_items_in} items: {bc_str}). "
            rf"Labeller confidence: {conf}."
        )
        out.append("")
        if description:
            out.append(rf"\textbf{{Description.}} {escape_tex(description)}")
            out.append("")

        col = loadings[:, fi]
        order_desc = np.argsort(-col)
        order_asc = np.argsort(col)
        pos_picks = [i for i in order_desc if col[i] > 0][:n_top]
        neg_picks = [i for i in order_asc if col[i] < 0][:n_top]

        out.append(
            rf"\paragraph{{Top positive loadings ({len(pos_picks)} items).}}\leavevmode"
        )
        out.append(
            r"\begin{itemize}\setlength{\itemsep}{4pt}\setlength{\parskip}{0pt}"
        )
        for i in pos_picks:
            out.extend(render_item(float(col[i]), items[i]))
        out.append(r"\end{itemize}")
        out.append("")

        out.append(
            rf"\paragraph{{Top negative loadings ({len(neg_picks)} items).}}\leavevmode"
        )
        out.append(
            r"\begin{itemize}\setlength{\itemsep}{4pt}\setlength{\parskip}{0pt}"
        )
        for i in neg_picks:
            out.extend(render_item(float(col[i]), items[i]))
        out.append(r"\end{itemize}")
        out.append("")

    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--top-n", type=int, default=8,
        help="Items per pole per factor to include (default 8).",
    )
    ap.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help=f"Output .tex path (default {OUTPUT_PATH}).",
    )
    args = ap.parse_args()

    tex = build_appendix(args.top_n)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tex)
    print(f"wrote {args.output}  ({len(tex.splitlines())} lines)")


if __name__ == "__main__":
    main()
