"""Unit tests for the personality eval pipeline.

Covers:
- ScaleSweep.scale_points()
- SuiteConfig.expand_models()
- suite._is_scale_in_eval()
- cli._with_filters()
- log_answer_parser: parse_answer(), rescore_log() -> RescoreResult
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evals.config import (
    AdapterConfig,
    InspectBenchmarkSpec,
    ModelSpec,
    ScaleSweep,
    SuiteConfig,
)
from scripts.evals.log_answer_parser import (
    RescoreResult,
    parse_answer,
    rescore_log,
)
from scripts.evals.suite import _is_scale_in_eval


# ---------------------------------------------------------------------------
# ScaleSweep.scale_points()
# ---------------------------------------------------------------------------


class TestScaleSweep:
    def test_basic_range(self):
        sweep = ScaleSweep(min=-1.0, max=1.0, step=0.5)
        points = sweep.scale_points()
        assert points == [-1.0, -0.5, 0.5, 1.0]

    def test_zero_excluded(self):
        sweep = ScaleSweep(min=-2.0, max=2.0, step=1.0)
        points = sweep.scale_points()
        assert 0.0 not in points
        assert -2.0 in points and 2.0 in points

    def test_single_positive_point(self):
        sweep = ScaleSweep(min=1.0, max=1.0, step=0.5)
        points = sweep.scale_points()
        assert points == [1.0]

    def test_fractional_step_precision(self):
        # Floating-point arithmetic shouldn't produce ghost values.
        sweep = ScaleSweep(min=-2.0, max=2.0, step=0.25)
        points = sweep.scale_points()
        # Should be exactly 16 non-zero points in [-2.0, -0.25] ∪ [0.25, 2.0]
        assert len(points) == 16
        for p in points:
            assert p != 0.0

    def test_sorted_ascending(self):
        sweep = ScaleSweep(min=-1.0, max=1.0, step=0.5)
        points = sweep.scale_points()
        assert points == sorted(points)

    def test_invalid_step(self):
        with pytest.raises(ValueError, match="step must be positive"):
            ScaleSweep(min=-1.0, max=1.0, step=0.0)

    def test_invalid_range(self):
        with pytest.raises(ValueError, match="min .* must be <= max"):
            ScaleSweep(min=1.0, max=-1.0, step=0.5)


# ---------------------------------------------------------------------------
# SuiteConfig.expand_models()
# ---------------------------------------------------------------------------


class TestExpandModels:
    BASE = "meta-llama/Llama-3.1-8B-Instruct"
    ADAPTER = "org/adapter-repo::persona"

    def _sweep_config(self, sweep: ScaleSweep, evals=None, adapter: str | None = None, tmp_path: Path | None = None) -> SuiteConfig:
        return SuiteConfig(
            base_model=self.BASE,
            adapter=adapter or self.ADAPTER,
            sweep=sweep,
            evals=evals or [InspectBenchmarkSpec(name="bfi", benchmark="personality_bfi")],
            output_root=tmp_path or Path("scratch/evals"),
        )

    def test_base_model_always_present(self, tmp_path: Path):
        cfg = self._sweep_config(ScaleSweep(min=0.5, max=1.0, step=0.5), tmp_path=tmp_path)
        models = cfg.expand_models()
        names = [m.name for m in models]
        assert "base" in names

    def test_base_model_has_no_adapter(self, tmp_path: Path):
        cfg = self._sweep_config(ScaleSweep(min=1.0, max=1.0, step=0.5), tmp_path=tmp_path)
        base = next(m for m in cfg.expand_models() if m.name == "base")
        assert base.adapters == []
        assert base.scale is None

    def test_lora_models_have_adapter(self, tmp_path: Path):
        cfg = self._sweep_config(ScaleSweep(min=1.0, max=2.0, step=1.0), tmp_path=tmp_path)
        lora_models = [m for m in cfg.expand_models() if m.name != "base"]
        assert len(lora_models) == 2
        for m in lora_models:
            assert len(m.adapters) == 1
            assert m.adapters[0].path == self.ADAPTER

    def test_scale_stored_in_model_spec(self, tmp_path: Path):
        cfg = self._sweep_config(ScaleSweep(min=1.0, max=1.0, step=0.5), tmp_path=tmp_path)
        lora = next(m for m in cfg.expand_models() if m.name != "base")
        assert lora.scale == 1.0

    def test_per_eval_sweep_union(self, tmp_path: Path):
        """Union of suite sweep and per-eval sweep produces all required models."""
        suite_sweep = ScaleSweep(min=1.0, max=2.0, step=1.0)  # 1.0, 2.0
        eval_sweep = ScaleSweep(min=0.5, max=1.5, step=0.5)   # 0.5, 1.0, 1.5
        cfg = SuiteConfig(
            base_model=self.BASE,
            adapter=self.ADAPTER,
            sweep=suite_sweep,
            evals=[
                InspectBenchmarkSpec(name="bfi", benchmark="personality_bfi"),
                InspectBenchmarkSpec(name="mmlu", benchmark="mmlu", sweep=eval_sweep),
            ],
            output_root=tmp_path,
        )
        models = cfg.expand_models()
        scales = {m.scale for m in models if m.scale is not None}
        # Union of {1.0, 2.0} and {0.5, 1.0, 1.5} = {0.5, 1.0, 1.5, 2.0}
        assert scales == {0.5, 1.0, 1.5, 2.0}

    def test_no_adapter_produces_no_lora_models(self, tmp_path: Path):
        """Without an adapter, only the base model is produced."""
        cfg = SuiteConfig(
            base_model=self.BASE,
            adapter=None,
            sweep=ScaleSweep(min=1.0, max=2.0, step=1.0),
            evals=[InspectBenchmarkSpec(name="bfi", benchmark="personality_bfi")],
            output_root=tmp_path,
        )
        models = cfg.expand_models()
        assert all(m.adapters == [] for m in models)

    def test_explicit_models_returned_unchanged(self, tmp_path: Path):
        specs = [
            ModelSpec(name="base", base_model=self.BASE),
            ModelSpec(name="lora_1x", base_model=self.BASE,
                      adapters=[AdapterConfig(path=self.ADAPTER, scale=1.0)], scale=1.0),
        ]
        cfg = SuiteConfig(models=specs,
                          evals=[InspectBenchmarkSpec(name="bfi", benchmark="personality_bfi")],
                          output_root=tmp_path)
        assert cfg.expand_models() == specs

    def test_sweep_and_models_mutually_exclusive(self, tmp_path: Path):
        with pytest.raises(ValueError, match="not both"):
            SuiteConfig(
                base_model=self.BASE,
                sweep=ScaleSweep(min=1.0, max=2.0, step=1.0),
                models=[ModelSpec(name="base", base_model=self.BASE)],
                evals=[InspectBenchmarkSpec(name="bfi", benchmark="personality_bfi")],
                output_root=tmp_path,
            )


# ---------------------------------------------------------------------------
# suite._is_scale_in_eval()
# ---------------------------------------------------------------------------


class TestIsScaleInEval:
    def _bfi_spec(self, sweep: ScaleSweep | None = None) -> InspectBenchmarkSpec:
        return InspectBenchmarkSpec(name="bfi", benchmark="personality_bfi", sweep=sweep)

    def test_base_model_always_included(self):
        spec = self._bfi_spec()
        suite_sweep = ScaleSweep(min=1.0, max=2.0, step=1.0)
        assert _is_scale_in_eval(None, spec, suite_sweep) is True

    def test_scale_in_suite_sweep(self):
        spec = self._bfi_spec()
        suite_sweep = ScaleSweep(min=1.0, max=2.0, step=1.0)
        assert _is_scale_in_eval(1.0, spec, suite_sweep) is True
        assert _is_scale_in_eval(2.0, spec, suite_sweep) is True

    def test_scale_not_in_suite_sweep(self):
        spec = self._bfi_spec()
        suite_sweep = ScaleSweep(min=1.0, max=2.0, step=1.0)
        assert _is_scale_in_eval(0.5, spec, suite_sweep) is False

    def test_per_eval_sweep_overrides_suite(self):
        eval_sweep = ScaleSweep(min=0.5, max=1.0, step=0.5)
        spec = self._bfi_spec(sweep=eval_sweep)
        suite_sweep = ScaleSweep(min=1.0, max=2.0, step=1.0)
        # 0.5 is in eval sweep but not suite sweep
        assert _is_scale_in_eval(0.5, spec, suite_sweep) is True
        # 2.0 is in suite sweep but not eval sweep
        assert _is_scale_in_eval(2.0, spec, suite_sweep) is False

    def test_no_sweep_always_included(self):
        spec = self._bfi_spec()
        assert _is_scale_in_eval(99.0, spec, None) is True


# ---------------------------------------------------------------------------
# parse_answer()
# ---------------------------------------------------------------------------


class TestParseAnswer:
    def test_answer_format(self):
        assert parse_answer("ANSWER: B", "ABCDE") == "B"
        assert parse_answer("Answer: C", "ABCDE") == "C"

    def test_leading_letter_paren(self):
        assert parse_answer("D) Neither agree nor disagree", "ABCDE") == "D"
        assert parse_answer("A) Strongly disagree", "ABCD") == "A"

    def test_bare_letter_newline(self):
        assert parse_answer("E\n\nI am not easily stressed.", "ABCDE") == "E"

    def test_answer_is_pattern(self):
        assert parse_answer("The correct answer is B.", "ABCDE") == "B"
        assert parse_answer("The answer is: A", "ABCDE") == "A"

    def test_invalid_letter_rejected(self):
        # 'F' is not in ABCDE
        assert parse_answer("ANSWER: F", "ABCDE") == None

    def test_letter_not_in_valid_set(self):
        # 'E' is not valid for TRAIT (ABCD only)
        assert parse_answer("ANSWER: E", "ABCD") == None

    def test_empty_input(self):
        assert parse_answer("", "ABCDE") == None
        assert parse_answer("   ", "ABCDE") == None

    def test_unparseable_garbage(self):
        assert parse_answer("I cannot answer this question.", "ABCDE") == None


# ---------------------------------------------------------------------------
# rescore_log() -> RescoreResult
# ---------------------------------------------------------------------------


def _make_sample(trait: str, answer: str, reverse: bool = False) -> dict:
    """Build a minimal inspect sample dict as produced by inspect-evals."""
    mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    return {
        "metadata": {
            "trait": trait,
            "answer_mapping": mapping,
            "reverse": reverse,
        },
        "scores": {"any_choice": {"answer": answer}},
        "output": {"choices": [{"message": {"content": f"ANSWER: {answer}"}}]},
    }


def _make_log(samples: list[dict]) -> dict:
    return {"samples": samples}


class TestRescoreLog:
    def test_basic_scores(self, tmp_path: Path):
        samples = [
            _make_sample("Openness", "C"),   # 3/5 = 0.6
            _make_sample("Openness", "E"),   # 5/5 = 1.0
        ]
        log_path = tmp_path / "log.json"
        log_path.write_text(json.dumps(_make_log(samples)), encoding="utf-8")

        result = rescore_log(log_path, "bfi")
        assert isinstance(result, RescoreResult)
        assert abs(result.scores["Openness"] - 0.8) < 1e-9
        assert result.n_parsed == 2
        assert result.n_total == 2
        assert result.parse_rate == 1.0

    def test_parse_rate_with_failures(self, tmp_path: Path):
        samples = [
            _make_sample("Neuroticism", "B"),
            # Unparseable sample — no valid answer anywhere
            {
                "metadata": {"trait": "Neuroticism", "answer_mapping": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}, "reverse": False},
                "scores": {"any_choice": {"answer": None}},
                "output": {"choices": [{"message": {"content": "I cannot answer this."}}]},
            },
        ]
        log_path = tmp_path / "log.json"
        log_path.write_text(json.dumps(_make_log(samples)), encoding="utf-8")

        result = rescore_log(log_path, "bfi")
        assert result.n_parsed == 1
        assert result.n_total == 2
        assert result.parse_rate == 0.5

    def test_reverse_scoring(self, tmp_path: Path):
        # Reverse: score = (max+1 - raw) / max = (5+1 - 1) / 5 = 1.0
        samples = [_make_sample("Agreeableness", "A", reverse=True)]
        log_path = tmp_path / "log.json"
        log_path.write_text(json.dumps(_make_log(samples)), encoding="utf-8")

        result = rescore_log(log_path, "bfi")
        assert abs(result.scores["Agreeableness"] - 1.0) < 1e-9

    def test_empty_log(self, tmp_path: Path):
        log_path = tmp_path / "log.json"
        log_path.write_text(json.dumps(_make_log([])), encoding="utf-8")

        result = rescore_log(log_path, "bfi")
        assert result.scores == {}
        assert result.n_parsed == 0
        assert result.n_total == 0
        import math
        assert math.isnan(result.parse_rate)

    def test_fallback_parser_used_when_scorer_fails(self, tmp_path: Path):
        """Fallback parser should recover 'B)' format when scorer returns None."""
        sample = {
            "metadata": {"trait": "Conscientiousness", "answer_mapping": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}, "reverse": False},
            "scores": {"any_choice": {"answer": None}},
            "output": {"choices": [{"message": {"content": "B) Sometimes"}}]},
        }
        log_path = tmp_path / "log.json"
        log_path.write_text(json.dumps(_make_log([sample])), encoding="utf-8")

        result = rescore_log(log_path, "bfi")
        assert result.n_parsed == 1
        assert "Conscientiousness" in result.scores
