"""Survey turn-count and token-count distributions of candidate rollout
datasets for psychometric questionnaire administration.

Streams a small sample (default 500 conversations) from each dataset, adapts
to a canonical ``[{role, content}, ...]`` message list, tokenises under the
Qwen2.5-7B-Instruct chat template (matching our current questionnaire-model
target), and writes:

* ``lengths.csv``      — per-conversation row: n_messages, n_assistant_turns,
                         total_tokens, n_chars.
* ``summary.csv``      — per-dataset percentiles + fraction under a 32k /
                         64k / 128k budget.
* ``histograms.png``   — overlaid turn-count and token-count distributions.

Streaming-only: no full dataset is downloaded. HF datasets cache is wiped
between datasets to stay within the project's memory budget.

Run:
    uv run python -m scripts_dev.psychometric_assessment.dataset_length_survey
"""

from __future__ import annotations

import gc
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════════

TOKENIZER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
N_SAMPLES_PER_DATASET = 500
OUTPUT_DIR = Path("scratch/dataset_length_survey")

# Token budgets we care about. 32k matches the Qwen2.5-7B context window we
# currently administer questionnaires under; higher budgets are informational.
BUDGETS = [8_192, 32_768, 65_536, 131_072]


# ═════════════════════════════════════════════════════════════════════════════
# Dataset adapters — each yields canonical {sample_id, messages} dicts.
# All are streaming-only. Stop after n_samples rows.
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class DatasetSpec:
    label: str
    assistant_model: str
    loader: Callable[[int], Iterable[dict]]
    notes: str = ""


def _stream(repo_id: str, **kwargs):
    """Stream one shard from an HF dataset (cache-safe)."""
    from datasets import load_dataset
    return load_dataset(repo_id, split="train", streaming=True, **kwargs)


def _canonicalise(messages_raw: list[dict]) -> list[dict]:
    """Strip messages to {role, content} only. Preserve system/user/assistant/
    tool in role; coerce content to string. Drops messages with empty content."""
    out = []
    for m in messages_raw:
        role = m.get("role") or ""
        content = m.get("content")
        if content is None:
            # SWE-rebench tool calls sometimes have content=None; represent them
            # via the tool_calls structure so length isn't underestimated.
            tc = m.get("tool_calls")
            if tc:
                content = json.dumps(tc, ensure_ascii=False)
            else:
                continue
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if not content.strip():
            continue
        out.append({"role": role, "content": content})
    return out


def swe_rebench_loader(n: int) -> Iterable[dict]:
    ds = _stream("nebius/SWE-rebench-openhands-trajectories")
    for i, row in enumerate(ds):
        if i >= n:
            break
        yield {
            "sample_id": row.get("trajectory_id") or f"row{i}",
            "messages": _canonicalise(row["trajectory"]),
        }


def kwai_swe_smith_loader(n: int) -> Iterable[dict]:
    ds = _stream("Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k")
    for i, row in enumerate(ds):
        if i >= n:
            break
        yield {
            "sample_id": row.get("instance_id") or f"row{i}",
            "messages": _canonicalise(row["messages"]),
        }


def lmsys_loader(n: int, min_turns: int = 5) -> Iterable[dict]:
    """Stream LMSYS-Chat-1M, filter to open-model rows with turn >= min_turns."""
    OPEN_MODELS = {
        "vicuna-13b", "vicuna-7b", "vicuna-33b", "llama-2-7b-chat",
        "llama-2-13b-chat", "llama-2-70b-chat", "koala-13b", "alpaca-13b",
        "mpt-7b-chat", "mpt-30b-chat", "chatglm-6b", "chatglm2-6b",
        "chatglm3-6b", "wizardlm-13b", "guanaco-33b", "oasst-pythia-12b",
        "RWKV-4-Raven-14B", "fastchat-t5-3b", "stablelm-tuned-alpha-7b",
        "dolly-v2-12b", "gpt4all-13b-snoozy", "tulu-30b", "baichuan-13b-chat",
    }
    ds = _stream("lmsys/lmsys-chat-1m")
    kept = 0
    seen = 0
    for row in ds:
        seen += 1
        if seen > n * 30:  # safety: stop scanning if filter is too strict
            break
        if row.get("model") not in OPEN_MODELS:
            continue
        if row.get("turn", 0) < min_turns:
            continue
        yield {
            "sample_id": row.get("conversation_id") or f"row{seen}",
            "messages": _canonicalise(row["conversation"]),
            "assistant_model": row.get("model"),
        }
        kept += 1
        if kept >= n:
            break


def prism_loader(n: int) -> Iterable[dict]:
    """PRISM: each row is one conversation tree; flatten the primary branch."""
    OPEN_MODEL_PREFIXES = ("llama", "mistral", "vicuna", "qwen", "gemma",
                           "mpt", "chatglm", "falcon", "yi", "deepseek")
    from datasets import load_dataset
    ds = load_dataset("HannahRoseKirk/prism-alignment", "conversations",
                      split="train", streaming=True)
    kept = 0
    for i, row in enumerate(ds):
        # row["conversation_history"] is a list of message dicts with
        # {turn, role, content, model_name, ...} in recent dumps.
        hist = row.get("conversation_history") or row.get("conversation")
        if not hist:
            continue
        models = {m.get("model_name", "") or "" for m in hist
                  if m.get("role") == "model" or m.get("role") == "assistant"}
        # Keep if any assistant turn came from an open-model family.
        is_open = any(any(mm.lower().startswith(p) for p in OPEN_MODEL_PREFIXES)
                      for mm in models if mm)
        if not is_open:
            continue
        # PRISM uses "model" as the assistant role — rewrite to "assistant".
        msgs = []
        for m in hist:
            role = m.get("role", "")
            if role == "model":
                role = "assistant"
            msgs.append({"role": role, "content": m.get("content", "")})
        yield {
            "sample_id": row.get("conversation_id") or f"row{i}",
            "messages": _canonicalise(msgs),
            "assistant_model": ",".join(sorted(models)),
        }
        kept += 1
        if kept >= n:
            break


DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        label="SWE-rebench-openhands",
        assistant_model="Qwen3-Coder-480B-A35B (+OpenHands scaffold)",
        loader=swe_rebench_loader,
        notes="coding-agent trajectories; tool+user+assistant+system roles",
    ),
    DatasetSpec(
        label="Kwai-Klear-mini-swe-agent",
        assistant_model="Qwen3-8B",
        loader=kwai_swe_smith_loader,
        notes="SWE-bench-style agent rollouts, 8B model — size-match to Llama-3.1-8B",
    ),
    DatasetSpec(
        label="LMSYS-Chat-1M (open, turn>=5)",
        assistant_model="Vicuna/Llama-2/Koala/MPT/WizardLM (mixed)",
        loader=lmsys_loader,
        notes="real-user diverse chat; requires HF auth acceptance",
    ),
    DatasetSpec(
        label="PRISM-alignment (open subset)",
        assistant_model="Llama-2/Mistral/etc (mixed)",
        loader=prism_loader,
        notes="values/controversial-topic prompts; high persona-elicitation",
    ),
]


# ═════════════════════════════════════════════════════════════════════════════
# Tokenisation + measurement
# ═════════════════════════════════════════════════════════════════════════════


def measure(conversations: Iterable[dict], tokenizer) -> pd.DataFrame:
    rows = []
    for c in conversations:
        msgs = c["messages"]
        if not msgs:
            continue
        try:
            ids = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=False, tokenize=True,
            )
            n_tokens = len(ids)
        except Exception:
            # Chat-template rejections (tool role, empty content, etc.): fall
            # back to raw encoding of concatenated text.
            text = " ".join(m["content"] for m in msgs)
            n_tokens = len(tokenizer.encode(text))
        rows.append({
            "sample_id": c["sample_id"],
            "n_messages": len(msgs),
            "n_assistant_turns": sum(1 for m in msgs if m["role"] == "assistant"),
            "n_user_turns": sum(1 for m in msgs if m["role"] == "user"),
            "n_chars": sum(len(m["content"]) for m in msgs),
            "n_tokens": n_tokens,
            "assistant_model": c.get("assistant_model", ""),
        })
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, label: str, n_seen: int) -> dict:
    tok = df["n_tokens"].to_numpy()
    turns = df["n_assistant_turns"].to_numpy()
    row = {
        "dataset": label,
        "n_sampled": len(df),
        "turns_p50": float(np.median(turns)),
        "turns_p90": float(np.percentile(turns, 90)),
        "turns_max": int(turns.max()) if len(turns) else 0,
        "tokens_p50": float(np.median(tok)),
        "tokens_p90": float(np.percentile(tok, 90)),
        "tokens_max": int(tok.max()) if len(tok) else 0,
        "tokens_mean": float(np.mean(tok)),
    }
    for b in BUDGETS:
        row[f"frac_under_{b // 1024}k"] = float((tok < b).mean())
    return row


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════


def plot_histograms(per_dataset: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(per_dataset), 3)))

    # (A) Assistant turn-count histogram (log-x)
    ax = axes[0, 0]
    bins = np.logspace(0, 3, 30)
    for (label, df), c in zip(per_dataset.items(), colors):
        ax.hist(df["n_assistant_turns"].clip(1, None), bins=bins, alpha=0.5,
                label=f"{label} (n={len(df)})", color=c, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Assistant turns per conversation")
    ax.set_ylabel("Conversations in sample")
    ax.set_title("(A) Assistant turn-count distribution")
    ax.legend(fontsize=8, loc="upper right")
    ax.axvline(15, color="red", lw=0.8, ls="--", label="B=15")

    # (B) Total-token histogram (log-x)
    ax = axes[0, 1]
    bins = np.logspace(2.5, 5.5, 30)
    for (label, df), c in zip(per_dataset.items(), colors):
        ax.hist(df["n_tokens"].clip(1, None), bins=bins, alpha=0.5,
                label=f"{label}", color=c, edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Total tokens per conversation (Qwen2.5 chat template)")
    ax.set_ylabel("Conversations in sample")
    ax.set_title("(B) Total-token distribution")
    for budget, tag in [(32_768, "32k (Qwen2.5-7B ctx)"),
                         (131_072, "128k (Llama-3.1 ctx)")]:
        ax.axvline(budget, color="black", lw=0.8, ls=":", label=tag)
    ax.legend(fontsize=7, loc="upper right")

    # (C) CDF of tokens — clean way to read "what fraction fits a budget"
    ax = axes[1, 0]
    for (label, df), c in zip(per_dataset.items(), colors):
        tok = np.sort(df["n_tokens"].values)
        y = np.arange(1, len(tok) + 1) / len(tok)
        ax.plot(tok, y, label=label, color=c, lw=1.4)
    ax.set_xscale("log")
    ax.set_xlabel("Total tokens per conversation")
    ax.set_ylabel("Cumulative fraction of sample")
    ax.set_title("(C) Token-count CDF")
    for budget in [8_192, 32_768, 65_536, 131_072]:
        ax.axvline(budget, color="gray", lw=0.6, ls=":")
        ax.text(budget, 0.02, f"{budget // 1024}k",
                rotation=90, fontsize=8, color="gray", va="bottom")
    ax.set_xlim(100, 1e6)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)

    # (D) Turns vs tokens scatter
    ax = axes[1, 1]
    for (label, df), c in zip(per_dataset.items(), colors):
        ax.scatter(df["n_assistant_turns"], df["n_tokens"],
                   s=8, alpha=0.4, label=label, color=c)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Assistant turns")
    ax.set_ylabel("Total tokens")
    ax.set_title("(D) Turns vs tokens")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"Rollout-length survey — {sum(len(d) for d in per_dataset.values())} "
        f"conversations across {len(per_dataset)} datasets  "
        f"(tokenizer: {TOKENIZER_MODEL})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[Plot] Wrote {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Driver
# ═════════════════════════════════════════════════════════════════════════════


def wipe_hf_cache(cache_dir: Path) -> None:
    """Delete the HF datasets cache to reclaim disk between datasets."""
    if cache_dir.exists():
        size_before = sum(f.stat().st_size for f in cache_dir.rglob("*")
                          if f.is_file())
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"[Cache] Wiped {cache_dir} ({size_before / 1e9:.2f} GB)")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    for noisy in ("datasets", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Scoped HF datasets cache so cleanup is safe.
    import os
    tmp_cache = Path(tempfile.mkdtemp(prefix="hfds_survey_", dir="/tmp"))
    os.environ["HF_DATASETS_CACHE"] = str(tmp_cache)
    print(f"[Cache] Using scoped datasets cache: {tmp_cache}")

    print(f"[Tokenizer] Loading {TOKENIZER_MODEL}…")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=True)

    all_lengths: dict[str, pd.DataFrame] = {}
    summaries: list[dict] = []

    for spec in DATASETS:
        print(f"\n{'=' * 60}\n[Survey] {spec.label}\n{'=' * 60}")
        try:
            convos = list(spec.loader(N_SAMPLES_PER_DATASET))
        except Exception as e:
            print(f"[Survey] {spec.label}: loader FAILED — {type(e).__name__}: "
                  f"{str(e)[:200]}")
            wipe_hf_cache(tmp_cache)
            continue

        if not convos:
            print(f"[Survey] {spec.label}: 0 conversations matched filter; skipping.")
            wipe_hf_cache(tmp_cache)
            continue

        print(f"[Survey] Loaded {len(convos)} conversations; tokenising…")
        df = measure(convos, tokenizer)
        df["dataset"] = spec.label
        df["assistant_model_declared"] = spec.assistant_model
        df.to_csv(OUTPUT_DIR / f"lengths_{spec.label.replace('/', '_')}.csv",
                  index=False)
        summary = summarise(df, spec.label, len(convos))
        summary["assistant_model"] = spec.assistant_model
        summary["notes"] = spec.notes
        summaries.append(summary)
        all_lengths[spec.label] = df

        print(f"[Survey] {spec.label}: "
              f"turns p50={summary['turns_p50']:.0f} p90={summary['turns_p90']:.0f}  "
              f"tokens p50={summary['tokens_p50']:,.0f} p90={summary['tokens_p90']:,.0f}  "
              f"max={summary['tokens_max']:,}")

        # Free memory + cache between datasets.
        del convos, df
        gc.collect()
        wipe_hf_cache(tmp_cache)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print("\n[Summary]")
    if len(summary_df):
        cols = ["dataset", "n_sampled", "turns_p50", "turns_p90", "tokens_p50",
                "tokens_p90", "tokens_max"] + [f"frac_under_{b // 1024}k" for b in BUDGETS]
        print(summary_df[cols].to_string(index=False))

        plot_histograms(all_lengths, OUTPUT_DIR / "histograms.png")

    # Final cleanup.
    shutil.rmtree(tmp_cache, ignore_errors=True)
    print(f"\n[Cache] Removed scoped cache. Outputs → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
