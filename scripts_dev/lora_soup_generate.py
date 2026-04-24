"""Souping LoRA adapters and running a single generation.

Lightweight utility: takes N adapter refs (with per-adapter scales), arithmetically
bakes them into one PEFT adapter dir via ``bake_combined_lora``, loads the base
model + baked adapter, and generates a response to a single user prompt.

Runs on CPU or GPU. Intended for quick manual inspections — not for batch work.

Example
-------
    uv run python scripts_dev/lora_soup_generate.py \\
      --adapter "persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/amplifier/v1/lora/souped@1.0" \\
      --adapter "persona-shattering-lasr/monorepo::fine_tuning/llama-3.1-8b-it/ocean/neuroticism/amplifier/v4_paired_dpo/lora/neuroticism_v3-persona@1.0" \\
      --prompt "I'm bored, what should I do?" \\
      --device cpu
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--base",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model HF id (default: %(default)s).",
    )
    p.add_argument(
        "--adapter",
        action="append",
        required=True,
        help='Adapter entry "path@scale" (repeat for multiple; "path" may use "repo_id::subfolder").',
    )
    p.add_argument("--prompt", required=True, help="User message to generate against.")
    p.add_argument("--system", default=None, help="Optional system prompt.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--bake-dir",
        default=None,
        help="Where to write the baked combined adapter (default: fresh tempdir, auto-cleaned).",
    )
    return p.parse_args()


def _pick_device_and_dtype(device_arg: str, dtype_arg: str):
    import torch

    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg

    if dtype_arg == "auto":
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_arg]
    return device, dtype


def main() -> int:
    load_dotenv()
    args = _parse_args()

    # Local imports: keep top-of-file cheap so --help is fast.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src_dev.utils.lora_combo_baking import bake_combined_lora
    from src_dev.utils.lora_composition import parse_weighted_adapter, split_adapter_reference

    # Local resolver: snapshot_download + allow_patterns has a known 0-files bug
    # on persona-shattering-lasr/monorepo (see src_dev/utils/hf_hub.py::download_path_to_dir),
    # so we go through that helper for HF refs and fall back to local paths otherwise.
    def _resolve(ref: str) -> str:
        from pathlib import Path as _P
        from src_dev.utils.hf_hub import download_path_to_dir

        plain, sub = split_adapter_reference(ref)
        if _P(plain).exists():
            return str(_P(plain) / sub) if sub else plain
        if "/" in plain and sub:  # looks like "repo_id::subpath"
            import tempfile as _tf
            cache_root = _P(_tf.gettempdir()) / "lora_soup_cache" / plain.replace("/", "_")
            target = cache_root / sub
            if not (target / "adapter_config.json").exists():
                target.mkdir(parents=True, exist_ok=True)
                download_path_to_dir(repo_id=plain, path_in_repo=sub, target_dir=target)
            return str(target)
        raise RuntimeError(f"Cannot resolve adapter ref {ref!r}: not a local path and no ::subfolder given")

    device, dtype = _pick_device_and_dtype(args.device, args.dtype)
    torch.manual_seed(args.seed)

    weighted = [parse_weighted_adapter(a) for a in args.adapter]
    print(f"[soup] baking {len(weighted)} adapter(s):", file=sys.stderr)
    for w in weighted:
        print(f"         {w.scale:+.3f}  {w.path}", file=sys.stderr)

    bake_dir_ctx = tempfile.TemporaryDirectory() if args.bake_dir is None else None
    bake_dir = Path(args.bake_dir) if args.bake_dir else Path(bake_dir_ctx.name) / "baked"
    bake_dir.mkdir(parents=True, exist_ok=True)

    try:
        baked_path, combined_rank = bake_combined_lora(
            [(w.path, w.scale) for w in weighted],
            bake_dir,
            resolve_to_local=_resolve,
        )
        print(f"[soup] baked adapter at {baked_path} (combined_rank={combined_rank})", file=sys.stderr)

        print(f"[load] base={args.base} device={device} dtype={dtype}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(args.base)
        base_kwargs = {"dtype": dtype}
        if device == "cuda":
            base_kwargs["device_map"] = "cuda"
        base = AutoModelForCausalLM.from_pretrained(args.base, **base_kwargs)
        if device == "cpu":
            base = base.to("cpu")
        model = PeftModel.from_pretrained(base, str(baked_path))
        model.eval()

        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
        if args.do_sample:
            gen_kwargs.update(temperature=args.temperature, top_p=args.top_p)

        print("[gen] generating...", file=sys.stderr)
        with torch.no_grad():
            out = model.generate(input_ids, **gen_kwargs)
        response = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)

        print("---USER---")
        print(args.prompt)
        print("---ASSISTANT---")
        print(response.strip())
        return 0
    finally:
        if bake_dir_ctx is not None:
            bake_dir_ctx.cleanup()


if __name__ == "__main__":
    sys.exit(main())
