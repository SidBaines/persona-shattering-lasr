from __future__ import annotations

import json
import re
from pathlib import Path

import scripts.inference
from scripts.factor_analysis import labelling


def _factor_data(n: int) -> list[dict]:
    return [
        {
            "factor_index": idx,
            "top": [{"input_group_id": f"g{idx}", "text_excerpt": f"high {idx}"}],
            "bottom": [{"input_group_id": f"g{idx}", "text_excerpt": f"low {idx}"}],
        }
        for idx in range(n)
    ]


def test_load_label_checkpoint_normalizes_and_completion_rules(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "labels.json"
    checkpoint_path.write_text(json.dumps(["done", "", "(labelling failed: boom)"]), encoding="utf-8")

    labels = labelling.load_label_checkpoint(checkpoint_path, total=5)

    assert labels == ["done", "", "(labelling failed: boom)", "", ""]
    assert labelling.label_is_complete("done")
    assert not labelling.label_is_complete("")
    assert not labelling.label_is_complete("(labelling failed: boom)")


def test_label_factors_resumes_from_checkpoint_and_saves_incrementally(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "labels.json"
    checkpoint_path.write_text(
        json.dumps(["existing label", "", "(labelling failed: timeout)"]),
        encoding="utf-8",
    )

    class DummyProvider:
        def __init__(self) -> None:
            self.calls: list[int] = []

        async def generate_async(self, messages, **kwargs) -> str:
            match = re.search(r"factor\s+(\d+)", messages[1]["content"])
            assert match is not None
            factor_idx = int(match.group(1))
            self.calls.append(factor_idx)
            return f"label {factor_idx}"

    provider = DummyProvider()
    monkeypatch.setattr(scripts.inference, "get_provider", lambda name, config: provider)

    save_snapshots: list[list[str]] = []
    original_save = labelling.save_label_checkpoint

    def _wrapped_save(labels: list[str], path: str | Path) -> None:
        save_snapshots.append(list(labels))
        original_save(labels, path)

    monkeypatch.setattr(labelling, "save_label_checkpoint", _wrapped_save)

    labels = labelling.label_factors(
        _factor_data(3),
        model="dummy-model",
        provider="openai",
        show_progress=False,
        max_concurrent=2,
        checkpoint_path=checkpoint_path,
    )

    assert labels == ["existing label", "label 1", "label 2"]
    assert sorted(provider.calls) == [1, 2]
    assert len(save_snapshots) == 2

    on_disk = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert on_disk == labels

