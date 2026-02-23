"""Tests for legacy JSONL migration into canonical runs."""

from __future__ import annotations

from scripts.datasets import load_samples, migrate_legacy_jsonl


def test_migrate_legacy_jsonl_creates_canonical_rows(tmp_path):
    legacy_path = tmp_path / "legacy.jsonl"
    legacy_path.write_text(
        "\n".join(
            [
                '{"question":"Q1","response":"R1","edited_response":"E1"}',
                '{"question":"Q2","response":"R2"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "run_migrated"
    migrated_run_dir, report_path = migrate_legacy_jsonl(
        input_path=legacy_path,
        run_dir=run_dir,
        system_prompt="legacy prompt",
    )

    assert migrated_run_dir == run_dir
    assert report_path.exists()

    samples = load_samples(run_dir)
    assert len(samples) == 2
    first = samples[0]
    assert first.inference.status == "success"
    assert first.inference.assistant_completion == "R1"
    assert any(variant.variant_name == "legacy_default" for variant in first.edit_variants)

