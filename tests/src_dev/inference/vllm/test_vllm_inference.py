"""Integration test: simple vLLM inference end-to-end.

Requires a GPU. Downloads ~16GB on first run (meta-llama/Llama-3.1-8B-Instruct).

Run with:
    python tests/src_dev/inference/vllm/test_vllm_inference.py
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src_dev.inference import run_inference, InferenceConfig
from src_dev.inference.config import VllmProviderConfig
from src_dev.common.config import DatasetConfig, GenerationConfig

_REPO_ROOT = Path(__file__).parents[4]

if __name__ == "__main__":
    config = InferenceConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        provider="vllm",
        dataset=DatasetConfig(
            source="local",
            path=str(_REPO_ROOT / "data/assistant-axis-extraction-questions.jsonl"),
            max_samples=5,
        ),
        generation=GenerationConfig(
            max_new_tokens=200,
            temperature=0.7,
        ),
        vllm=VllmProviderConfig(
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
        ),
        output_path=None,
    )

    dataset, result = run_inference(config)

    assert result.num_failed == 0, f"Expected 0 failures, got {result.num_failed}"
    assert result.num_samples == 5, f"Expected 5 samples, got {result.num_samples}"
    assert len(dataset) == 5, f"Expected 5 rows in dataset, got {len(dataset)}"
    for row in dataset:
        assert row.get("response"), f"Empty response for question: {row.get('question')}"

    print("\nAll assertions passed.")
    for row in dataset.select(range(min(3, len(dataset)))):
        print(f"\nQ: {row['question']}")
        print(f"A: {row['response'][:200]}")
