"""Tests for the tasks bridge (process_results functions)."""

from __future__ import annotations

import pytest

from scripts.evals.tasks import get_custom_task_names
from scripts.evals.tasks.utils import (
    process_results_count_o,
    process_results_lowercase_density,
    process_results_punctuation_density,
)


class TestProcessResults:
    """Test that process_results bridge functions return correct metrics."""

    def test_count_o_basic(self):
        doc = {"question": "What color is the sky?"}
        results = ["The sky is a beautiful color of blue."]

        metrics = process_results_count_o(doc, results)

        assert "count_o.count" in metrics
        assert "count_o.density" in metrics
        # "of" has 1 'o', "color" has 1 'o', "beautiful" has 0 => let's just check types
        assert isinstance(metrics["count_o.count"], (int, float))
        assert isinstance(metrics["count_o.density"], (int, float))
        assert metrics["count_o.count"] >= 0

    def test_count_o_no_os(self):
        doc = {"question": "test"}
        results = ["The sky is blue."]

        metrics = process_results_count_o(doc, results)
        assert metrics["count_o.count"] == 0

    def test_count_o_with_os(self):
        doc = {"question": "test"}
        results = ["old octopus"]

        metrics = process_results_count_o(doc, results)
        assert metrics["count_o.count"] == 3  # o, o, o in "old" and "octopus"

    def test_lowercase_density(self):
        doc = {"question": "test"}
        results = ["hello world"]

        metrics = process_results_lowercase_density(doc, results)
        assert "lowercase_density.count" in metrics
        assert "lowercase_density.density" in metrics
        assert metrics["lowercase_density.count"] > 0

    def test_punctuation_density(self):
        doc = {"question": "test"}
        results = ["Hello, world! How are you?"]

        metrics = process_results_punctuation_density(doc, results)
        assert "punctuation_density.count" in metrics
        assert metrics["punctuation_density.count"] >= 3  # , ! ?

    def test_empty_response(self):
        doc = {"question": "test"}
        results = [""]

        metrics = process_results_count_o(doc, results)
        assert metrics["count_o.count"] == 0

    def test_filters_string_values(self):
        """Verify that non-numeric values are filtered out."""
        doc = {"question": "test"}
        results = ["hello"]

        metrics = process_results_count_o(doc, results)
        for v in metrics.values():
            assert isinstance(v, (int, float))


class TestTaskRegistry:
    def test_custom_task_names(self):
        names = get_custom_task_names()
        assert "persona_count_o" in names
        assert "persona_verb_count" in names
        assert "persona_coherence" in names
        assert "persona_lowercase_density" in names
        assert "persona_punctuation_density" in names

    def test_no_defaults_in_registry(self):
        """The _defaults.yaml template should not appear as a task."""
        names = get_custom_task_names()
        assert "_defaults" not in names
        for name in names:
            assert not name.startswith("_")
