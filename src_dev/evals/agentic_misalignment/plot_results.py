"""Plot 30-epoch harmfulness results (leaking/none/goal) with Wilson 95% CIs.

Combines samples from ``ocean11_leaking_none_goal`` and
``ocean11_leaking_none_goal_extra20`` into one 30-epoch tally per model, then
renders a forest plot sorted by harmful rate.
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
from inspect_ai.log import read_eval_log

_RUNS_ROOT = Path(
    "/workspace/persona-shattering-lasr/scratch/evals/agentic_misalignment"
)
_RUN_DIRS = [_RUNS_ROOT / "ocean11_leaking_none_goal", _RUNS_ROOT / "ocean11_leaking_none_goal_extra20"]
_MODELS = [
    "base",
    "o_plus_vanton1", "o_minus_vanton1",
    "c_plus_v1_souped", "c_minus_v2",
    "e_plus_vanton1",
    "a_plus_vanton2", "a_minus_v2",
    "n_plus_v4", "n_minus_v4",
    "control_empty_traits", "control_diff_words",
]

_LABEL = {
    "base": "base",
    "o_plus_vanton1": "O↑  (openness amp)",
    "o_minus_vanton1": "O↓  (openness sup)",
    "c_plus_v1_souped": "C↑  (conscientiousness amp)",
    "c_minus_v2": "C↓  (conscientiousness sup)",
    "e_plus_vanton1": "E↑  (extraversion amp)",
    "a_plus_vanton2": "A↑  (agreeableness amp)",
    "a_minus_v2": "A↓  (agreeableness sup)",
    "n_plus_v4": "N↑  (neuroticism amp)",
    "n_minus_v4": "N↓  (neuroticism sup)",
    "control_empty_traits": "control_empty_traits",
    "control_diff_words": "control_diff_words",
}

_COLOR = {
    "base": "#222222",
    "o_plus_vanton1": "#d62728", "c_plus_v1_souped": "#d62728",
    "e_plus_vanton1": "#d62728", "a_plus_vanton2": "#d62728",
    "n_plus_v4": "#d62728",
    "o_minus_vanton1": "#1f77b4", "c_minus_v2": "#1f77b4",
    "a_minus_v2": "#1f77b4", "n_minus_v4": "#1f77b4",
    "control_empty_traits": "#7f7f7f", "control_diff_words": "#7f7f7f",
}


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    rad = (z / denom) * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - rad), min(1.0, center + rad)


def _collect_harmful(model_name: str) -> list[int]:
    harms: list[int] = []
    for run_dir in _RUN_DIRS:
        ri = run_dir / model_name / "leaking_none_goal" / "run_info.json"
        if not ri.exists():
            continue
        info = json.loads(ri.read_text())
        log = read_eval_log(info["native"]["inspect_log_path"], format="json")
        for sample in log.samples or []:
            sc = sample.scores.get("harmfulness_scorer") if sample.scores else None
            if sc and isinstance(sc.value, dict):
                harms.append(int(float(sc.value.get("harmful", 0))))
    return harms


def main(output_path: str | Path) -> Path:
    rows = []
    for name in _MODELS:
        harms = _collect_harmful(name)
        if not harms:
            continue
        k, n = sum(harms), len(harms)
        lo, hi = _wilson_ci(k, n)
        rows.append((name, k / n, lo, hi, k, n))

    rows.sort(key=lambda r: r[1])
    base_rate = next(r[1] for r in rows if r[0] == "base")

    fig, ax = plt.subplots(figsize=(9, 6))
    ys = list(range(len(rows)))
    for y, (name, p, lo, hi, k, n) in zip(ys, rows):
        err = [[p - lo], [hi - p]]
        ax.errorbar(
            p, y, xerr=err, fmt="o",
            color=_COLOR.get(name, "#555555"),
            markersize=8, capsize=4, linewidth=2,
        )
        ax.annotate(f"{k}/{n}", (p, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color="#444")

    ax.axvline(base_rate, linestyle="--", color="#222", alpha=0.5, linewidth=1,
               label=f"base rate = {base_rate:.2f}")
    ax.set_yticks(ys)
    ax.set_yticklabels([_LABEL.get(r[0], r[0]) for r in rows])
    ax.set_xlim(-0.02, 1.0)
    ax.set_xlabel("P(harmful | leaking / urgency=none / extra=goal)   —   30 epochs, Wilson 95% CI")
    ax.set_title("Llama-3.1-8B-Instruct × OCEAN persona adapters\n"
                 "Agentic Misalignment (leaking, no urgency, welfare goal)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=False)

    # legend for color coding
    from matplotlib.patches import Patch
    proxies = [
        Patch(facecolor="#d62728", label="trait amplifier"),
        Patch(facecolor="#1f77b4", label="trait suppressor"),
        Patch(facecolor="#7f7f7f", label="control LoRA"),
        Patch(facecolor="#222222", label="base model"),
    ]
    ax.legend(handles=proxies + [plt.Line2D([], [], linestyle="--", color="#222",
              label=f"base rate = {base_rate:.2f}")],
              loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    if out.suffix.lower() != ".pdf":
        pdf_out = out.with_suffix(".pdf")
        fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    out = main(_RUNS_ROOT / "forest_leaking_none_goal_30ep.png")
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.pdf')}")
