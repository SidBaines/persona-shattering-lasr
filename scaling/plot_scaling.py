#!/usr/bin/env python3
"""Plot scaling sweep results (v2): Verb Count & Density vs Scale with CIs and sample text boxes."""

import json
import math
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────
summary = []
with open("scratch/toy-20260211-180132/scaling_sweep_v2/scaling_summary.jsonl") as f:
    for line in f:
        summary.append(json.loads(line))

per_question = []
with open("scratch/toy-20260211-180132/scaling_sweep_v2/all_results.jsonl") as f:
    for line in f:
        per_question.append(json.loads(line))

scales = [r["scaling_factor"] for r in summary]
n = summary[0]["num_samples"]

count_mean = [r["level_of_persona.count.mean"] for r in summary]
count_std  = [r["level_of_persona.count.stdev"] for r in summary]
count_ci   = [1.96 * s / math.sqrt(n) for s in count_std]

density_mean = [r["level_of_persona.density.mean"] for r in summary]
density_std  = [r["level_of_persona.density.stdev"] for r in summary]
density_ci   = [1.96 * s / math.sqrt(n) for s in density_std]

# ── Helper: highlight verbs in text using spacy ───────────────────────────
import spacy
nlp = spacy.load("en_core_web_sm")


def highlight_verbs(text, max_chars=200):
    """Return text with verbs wrapped in «» markers, truncated."""
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    doc = nlp(text)
    result_parts = []
    prev_end = 0
    for token in doc:
        if token.pos_ == "VERB":
            result_parts.append(text[prev_end:token.idx])
            result_parts.append(f"«{token.text}»")
            prev_end = token.idx + len(token.text)
    result_parts.append(text[prev_end:])
    return "".join(result_parts)


def wrap_text(text, width=44):
    return "\n".join(textwrap.wrap(text, width=width))


# ── Pick sample responses ─────────────────────────────────────────────────
# Use question 4 (chili pepper) — coherent across most scales
sample_q = per_question[3]  # "What is the spiciest part of a chili pepper?"
question_label = sample_q["question"]

# ── Create figure ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 26))
gs = fig.add_gridspec(
    3, 1, height_ratios=[1.0, 1.0, 1.1],
    hspace=0.30,
)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax_boxes = fig.add_subplot(gs[2])  # dedicated panel for text boxes
fig.suptitle(
    "LoRA Scaling Sweep (v2 retrained) — verbs_avoiding persona\n"
    "Llama-3.1-8B-Instruct  ·  20 TruthfulQA questions per scale",
    fontsize=16, fontweight="bold", y=0.98,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOP PANEL: Verb Count
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax1.fill_between(
    scales,
    [m - c for m, c in zip(count_mean, count_ci)],
    [m + c for m, c in zip(count_mean, count_ci)],
    alpha=0.2, color="#2196F3", label="95% CI",
)
ax1.plot(scales, count_mean, "o-", color="#1565C0", linewidth=2.5, markersize=7, label="Mean verb count", zorder=5)
ax1.axvline(0, color="gray", linestyle="--", alpha=0.5, label="Baseline (s=0)")
ax1.axvline(1, color="green", linestyle=":", alpha=0.5, label="Standard LoRA (s=1)")

# Coherence-loss shaded region
ax1.axvspan(-2.2, -1.0, alpha=0.08, color="red", zorder=0)
ax1.annotate(
    "Coherence lost\n(gibberish / degenerate)",
    xy=(-1.6, max(count_mean) * 0.55),
    fontsize=10, color="#B71C1C", fontstyle="italic", ha="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFCDD2", edgecolor="#E57373", alpha=0.8),
)

ax1.annotate("← Persona reversal", xy=(-0.95, max(count_mean) * 0.95), fontsize=10, color="#C62828", fontstyle="italic")
ax1.annotate("Persona amplified →", xy=(1.05, max(count_mean) * 0.95), fontsize=10, color="#2E7D32", fontstyle="italic", ha="left")

ax1.set_xlabel("Scaling Factor", fontsize=12)
ax1.set_ylabel("Verb Count (mean ± 95% CI)", fontsize=12)
ax1.set_title("Verb Count vs. LoRA Scaling Factor", fontsize=14)
ax1.legend(loc="upper right", fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-2.2, 2.2)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MIDDLE PANEL: Verb Density
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax2.fill_between(
    scales,
    [m - c for m, c in zip(density_mean, density_ci)],
    [m + c for m, c in zip(density_mean, density_ci)],
    alpha=0.2, color="#FF9800", label="95% CI",
)
ax2.plot(scales, density_mean, "s-", color="#E65100", linewidth=2.5, markersize=7, label="Mean verb density (%)", zorder=5)
ax2.axvline(0, color="gray", linestyle="--", alpha=0.5, label="Baseline (s=0)")
ax2.axvline(1, color="green", linestyle=":", alpha=0.5, label="Standard LoRA (s=1)")

# Coherence-loss shaded region
ax2.axvspan(-2.2, -1.0, alpha=0.08, color="red", zorder=0)

ax2.set_xlabel("Scaling Factor", fontsize=12)
ax2.set_ylabel("Verb Density (% of words, mean ± 95% CI)", fontsize=12)
ax2.set_title("Verb Density vs. LoRA Scaling Factor", fontsize=14)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-2.2, 2.2)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BOTTOM PANEL: Sample text boxes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ax_boxes.set_xlim(0, 10.2)
ax_boxes.set_ylim(0, 2.15)
ax_boxes.axis("off")
ax_boxes.set_title(
    f'Sample responses — Q: "{question_label}"\n'
    f'«word» = verb detected by spacy POS tagger',
    fontsize=13, fontweight="bold", pad=10,
)

# 5 boxes in one row, evenly spaced
box_layout = [
    # (scale_val, x_left, color, label_extra)
    (-1.5, 0.10, "#FFCDD2", "INCOHERENT"),
    (-1.0, 2.10, "#FFE0B2", "borderline"),
    ( 0.0, 4.10, "#E0E0E0", "BASELINE"),
    ( 1.0, 6.10, "#C8E6C9", ""),
    ( 2.0, 8.10, "#A5D6A7", "strong persona"),
]

for scale_val, bx, bg_color, extra_label in box_layout:
    sk = str(float(scale_val))
    resp = sample_q["scales"][sk]["response"]
    verb_count = int(sample_q["scales"][sk]["metrics"]["level_of_persona.count"])
    density_val = sample_q["scales"][sk]["metrics"]["level_of_persona.density"]

    highlighted = highlight_verbs(resp, max_chars=220)
    wrapped = wrap_text(highlighted, width=30)

    header = f"s={scale_val:+.1f}  density:{density_val}%"
    if extra_label:
        header += f"  [{extra_label}]"
    box_text = f"{header}\n{'─' * 32}\n{wrapped}"

    props = dict(
        boxstyle="round,pad=0.4", facecolor=bg_color,
        edgecolor="#757575", alpha=0.92, linewidth=1.3,
    )
    ax_boxes.text(
        bx, 2.0, box_text,
        fontsize=7.8, fontfamily="monospace",
        verticalalignment="top", horizontalalignment="left",
        bbox=props,
    )

# Draw arrows between boxes (horizontally centered between them)
for i in range(len(box_layout) - 1):
    _, x1, _, _ = box_layout[i]
    _, x2, _, _ = box_layout[i + 1]
    ax_boxes.annotate(
        "", xy=(x2 + 0.05, 1.25), xytext=(x1 + 1.90, 1.25),
        arrowprops=dict(arrowstyle="-|>", color="#616161", lw=2, shrinkA=2, shrinkB=2),
    )

# ── Footer note ───────────────────────────────────────────────────────────
fig.text(
    0.5, 0.005,
    "«word» = verb detected by spacy POS tagger  ·  "
    "95% CI = mean ± 1.96·σ/√n  (n=20)  ·  "
    "Red zone (s < -1): coherence degrades rapidly as model output becomes degenerate",
    ha="center", fontsize=9, fontstyle="italic", color="#616161",
)

plt.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.04)
outpath = "scratch/toy-20260211-180132/scaling_sweep_v2/scaling_plot.png"
fig.savefig(outpath, dpi=150, bbox_inches="tight")
print(f"Plot saved to {outpath}")
