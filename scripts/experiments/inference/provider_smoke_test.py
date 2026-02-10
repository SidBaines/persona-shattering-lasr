#!/usr/bin/env python3
"""Provider smoke test with debug logging for inference."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from scripts.common.config import GenerationConfig
from scripts.inference import (
    AnthropicProviderConfig,
    InferenceConfig,
    LocalProviderConfig,
    OpenAIProviderConfig,
    OpenRouterProviderConfig,
)
from scripts.inference.providers import get_provider
from scripts.utils import setup_logging

logger = logging.getLogger("persona_shattering")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test inference providers with a single prompt.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Say hello in one sentence.",
        help="Prompt to send to each provider.",
    )
    parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=["openai", "openrouter", "anthropic"],
        choices=["local", "openai", "openrouter", "anthropic"],
        help="Providers to test (default: openai openrouter anthropic)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level (DEBUG, INFO, WARNING, ERROR).",
    )

    parser.add_argument(
        "--local-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Local HuggingFace model name.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-5-nano-2025-08-07",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--openrouter-model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="OpenRouter model name.",
    )
    parser.add_argument(
        "--anthropic-model",
        type=str,
        default="claude-3-haiku-20240307",
        help="Anthropic model name.",
    )
    parser.add_argument(
        "--anthropic-batch",
        action="store_true",
        help="Use Anthropic Message Batches API for the Anthropic provider.",
    )
    parser.add_argument(
        "--anthropic-version",
        type=str,
        default="2023-06-01",
        help="Anthropic API version header (default: 2023-06-01).",
    )
    parser.add_argument(
        "--anthropic-batch-poll-interval",
        type=int,
        default=10,
        help="Polling interval in seconds for Anthropic batch (default: 10).",
    )
    parser.add_argument(
        "--anthropic-batch-timeout",
        type=int,
        default=None,
        help="Optional timeout in seconds for Anthropic batch completion.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Optional run directory under scratch/ (default: timestamped).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing Anthropic batch run from run-dir metadata.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max new/output tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--openai-reasoning-effort",
        type=str,
        choices=["none", "low", "medium", "high"],
        default=None,
        help="OpenAI reasoning effort (optional).",
    )
    parser.add_argument(
        "--openai-verbosity",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="OpenAI verbosity (optional).",
    )

    return parser.parse_args()


def _build_config(provider: str, args: argparse.Namespace) -> InferenceConfig:
    if provider == "local":
        model = args.local_model
    elif provider == "openai":
        model = args.openai_model
    elif provider == "openrouter":
        model = args.openrouter_model
    elif provider == "anthropic":
        model = args.anthropic_model
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return InferenceConfig(
        model=model,
        provider=provider,
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=1,
        ),
        local=LocalProviderConfig(),
        openai=OpenAIProviderConfig(
            reasoning_effort=args.openai_reasoning_effort,
            verbosity=args.openai_verbosity,
        ),
        openrouter=OpenRouterProviderConfig(),
        anthropic=AnthropicProviderConfig(),
    )


def _anthropic_api_request(
    method: str,
    url: str,
    *,
    api_key: str,
    anthropic_version: str,
    body: dict | None = None,
    expect_json: bool = True,
) -> str | dict:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": anthropic_version,
        "content-type": "application/json",
    }
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8") if exc.fp else ""
        raise RuntimeError(
            f"Anthropic API HTTP {exc.code} for {method} {url}: {error_body}"
        ) from exc

    if not expect_json:
        return payload
    return json.loads(payload)


def _extract_anthropic_text(message: dict | None) -> str:
    if not message:
        return ""
    content = message.get("content") or []
    parts: list[str] = []
    for block in content:
        if block.get("type") == "text" and block.get("text"):
            parts.append(block["text"])
    return "".join(parts).strip()


def _run_anthropic_batch(prompt: str, args: argparse.Namespace) -> str:
    api_key = None
    try:
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set.")

    run_id = args.run_dir or datetime.now().strftime("provider-smoke-%Y%m%d-%H%M%S")
    run_dir = Path("scratch") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    request_path = run_dir / "anthropic_batch_request.json"
    metadata_path = run_dir / "anthropic_batch_metadata.json"
    results_path = run_dir / "anthropic_batch_results.jsonl"

    if args.resume:
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Resume requested but metadata not found: {metadata_path}"
            )
        metadata = json.loads(metadata_path.read_text())
        batch_id = metadata.get("batch_id")
        results_url = metadata.get("results_url")
        if not batch_id:
            raise RuntimeError(
                f"Resume metadata missing batch_id: {metadata_path}"
            )
    else:
        request_payload = {
            "requests": [
                {
                    "custom_id": "req_0_0",
                    "params": {
                        "model": args.anthropic_model,
                        "max_tokens": args.max_new_tokens,
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt}],
                            }
                        ],
                    },
                }
            ]
        }
        request_path.write_text(json.dumps(request_payload, indent=2))

        batch_create = _anthropic_api_request(
            "POST",
            "https://api.anthropic.com/v1/messages/batches",
            api_key=api_key,
            anthropic_version=args.anthropic_version,
            body=request_payload,
        )

        batch_id = batch_create.get("id")
        if not batch_id:
            raise RuntimeError(f"Unexpected batch create response: {batch_create}")
        results_url = batch_create.get("results_url")
        metadata_path.write_text(
            json.dumps(
                {
                    "batch_id": batch_id,
                    "results_url": results_url,
                    "created_at": datetime.now().isoformat(),
                    "request_path": str(request_path),
                },
                indent=2,
            )
        )

    start_time = time.time()
    while True:
        batch = _anthropic_api_request(
            "GET",
            f"https://api.anthropic.com/v1/messages/batches/{batch_id}",
            api_key=api_key,
            anthropic_version=args.anthropic_version,
        )
        status = batch.get("processing_status")
        results_url = batch.get("results_url") or results_url
        metadata_path.write_text(
            json.dumps(
                {
                    "batch_id": batch_id,
                    "results_url": results_url,
                    "status": status,
                    "updated_at": datetime.now().isoformat(),
                    "request_path": str(request_path),
                    "results_path": str(results_path),
                },
                indent=2,
            )
        )
        if status == "ended":
            break
        if args.anthropic_batch_timeout is not None:
            if time.time() - start_time > args.anthropic_batch_timeout:
                raise TimeoutError(
                    f"Anthropic batch timed out after {args.anthropic_batch_timeout} seconds."
                )
        time.sleep(max(1, args.anthropic_batch_poll_interval))

    if not results_url:
        raise RuntimeError("Anthropic batch ended without results_url.")

    results_text = _anthropic_api_request(
        "GET",
        results_url,
        api_key=api_key,
        anthropic_version=args.anthropic_version,
        expect_json=False,
    )

    results_path.write_text(results_text)
    results_lines = []
    for line in results_text.splitlines():
        if line.strip():
            results_lines.append(json.loads(line))

    for item in results_lines:
        if item.get("custom_id") != "0:0":
            continue
        result = item.get("result") or {}
        if result.get("type") == "succeeded":
            return _extract_anthropic_text(result.get("message"))
        if result.get("type") == "errored":
            raise RuntimeError(f"Anthropic batch request errored: {result}")
        if result.get("type") == "canceled":
            raise RuntimeError("Anthropic batch request was canceled.")

    raise RuntimeError("Anthropic batch results did not include the expected response.")


def main() -> None:
    load_dotenv()
    args = parse_args()
    setup_logging(args.log_level)

    print("=" * 60)
    print("INFERENCE PROVIDER SMOKE TEST")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Providers: {', '.join(args.providers)}")
    print("".rjust(60, "-"))

    for provider_name in args.providers:
        print(f"\n[{provider_name}]")
        try:
            config = _build_config(provider_name, args)
            if provider_name == "anthropic" and args.anthropic_batch:
                response = _run_anthropic_batch(args.prompt, args)
            else:
                provider = get_provider(provider_name, config)
                response = provider.generate(args.prompt)
            print(f"Model: {config.model}")
            print("Response:")
            if response.strip():
                print(response)
            else:
                print("[EMPTY RESPONSE] (check logs for provider details)")
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
