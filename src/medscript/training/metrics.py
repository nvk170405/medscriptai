"""Training metrics — WER, CER, entity F1 for model evaluation."""

from __future__ import annotations

from typing import Any

import editdistance
import torch


def word_error_rate(predictions: list[str], references: list[str]) -> float:
    """
    Compute Word Error Rate (WER).

    WER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=total reference words.
    """
    total_errors = 0
    total_words = 0

    for pred, ref in zip(predictions, references):
        pred_words = pred.strip().split()
        ref_words = ref.strip().split()
        total_errors += editdistance.eval(pred_words, ref_words)
        total_words += len(ref_words)

    return total_errors / max(total_words, 1)


def character_error_rate(predictions: list[str], references: list[str]) -> float:
    """
    Compute Character Error Rate (CER).

    CER = edit_distance(pred_chars, ref_chars) / len(ref_chars)
    """
    total_errors = 0
    total_chars = 0

    for pred, ref in zip(predictions, references):
        total_errors += editdistance.eval(list(pred), list(ref))
        total_chars += len(ref)

    return total_errors / max(total_chars, 1)


def entity_f1_score(
    predicted_entities: list[list[dict[str, Any]]],
    reference_entities: list[list[dict[str, Any]]],
    entity_type: str | None = None,
) -> dict[str, float]:
    """
    Compute entity-level precision, recall, and F1.

    Args:
        predicted_entities: List of entity lists from model
        reference_entities: List of entity lists from ground truth
        entity_type: Optional — filter to specific entity type

    Returns:
        Dict with precision, recall, f1 keys
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for preds, refs in zip(predicted_entities, reference_entities):
        # Filter by type if specified
        if entity_type:
            preds = [e for e in preds if e.get("type") == entity_type]
            refs = [e for e in refs if e.get("type") == entity_type]

        # Match by value (case-insensitive)
        pred_values = {e.get("value", "").lower().strip() for e in preds}
        ref_values = {e.get("value", "").lower().strip() for e in refs}

        true_positives += len(pred_values & ref_values)
        false_positives += len(pred_values - ref_values)
        false_negatives += len(ref_values - pred_values)

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def compute_all_metrics(
    predictions: list[str],
    references: list[str],
    predicted_entities: list[list[dict[str, Any]]] | None = None,
    reference_entities: list[list[dict[str, Any]]] | None = None,
) -> dict[str, float]:
    """
    Compute all evaluation metrics.

    Returns dict with wer, cer, and per-entity-type F1 scores.
    """
    metrics: dict[str, float] = {
        "wer": word_error_rate(predictions, references),
        "cer": character_error_rate(predictions, references),
    }

    if predicted_entities and reference_entities:
        # Overall entity F1
        overall = entity_f1_score(predicted_entities, reference_entities)
        metrics["entity_f1"] = overall["f1"]
        metrics["entity_precision"] = overall["precision"]
        metrics["entity_recall"] = overall["recall"]

        # Per-type F1
        for etype in ["medicine", "dosage", "frequency", "duration", "instruction"]:
            type_metrics = entity_f1_score(
                predicted_entities, reference_entities, entity_type=etype
            )
            metrics[f"{etype}_f1"] = type_metrics["f1"]

    return metrics
