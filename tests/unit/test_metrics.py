"""Unit tests for training metrics."""

from __future__ import annotations

import pytest

from medscript.training.metrics import (
    word_error_rate,
    character_error_rate,
    entity_f1_score,
    compute_all_metrics,
)


class TestWER:
    """Tests for Word Error Rate."""

    def test_perfect_match(self) -> None:
        assert word_error_rate(["hello world"], ["hello world"]) == 0.0

    def test_complete_mismatch(self) -> None:
        wer = word_error_rate(["foo bar"], ["hello world"])
        assert wer > 0.0

    def test_empty_reference(self) -> None:
        assert word_error_rate(["hello"], [""]) == 0.0  # Division by max(0,1)

    def test_partial_match(self) -> None:
        wer = word_error_rate(["hello world foo"], ["hello world"])
        assert 0.0 < wer <= 1.0


class TestCER:
    """Tests for Character Error Rate."""

    def test_perfect_match(self) -> None:
        assert character_error_rate(["abc"], ["abc"]) == 0.0

    def test_one_char_diff(self) -> None:
        cer = character_error_rate(["abd"], ["abc"])
        assert cer == pytest.approx(1 / 3)


class TestEntityF1:
    """Tests for entity-level F1 score."""

    def test_perfect_match(self) -> None:
        preds = [[{"type": "medicine", "value": "Amoxicillin"}]]
        refs = [[{"type": "medicine", "value": "Amoxicillin"}]]
        result = entity_f1_score(preds, refs)
        assert result["f1"] == 1.0

    def test_no_match(self) -> None:
        preds = [[{"type": "medicine", "value": "Aspirin"}]]
        refs = [[{"type": "medicine", "value": "Amoxicillin"}]]
        result = entity_f1_score(preds, refs)
        assert result["f1"] == 0.0

    def test_type_filter(self) -> None:
        preds = [[{"type": "medicine", "value": "Aspirin"}, {"type": "dosage", "value": "500mg"}]]
        refs = [[{"type": "medicine", "value": "Aspirin"}, {"type": "dosage", "value": "250mg"}]]
        result = entity_f1_score(preds, refs, entity_type="medicine")
        assert result["f1"] == 1.0


class TestComputeAllMetrics:
    """Tests for the combined metrics computation."""

    def test_basic_metrics(self) -> None:
        metrics = compute_all_metrics(
            predictions=["hello world"],
            references=["hello world"],
        )
        assert "wer" in metrics
        assert "cer" in metrics
        assert metrics["wer"] == 0.0
        assert metrics["cer"] == 0.0
