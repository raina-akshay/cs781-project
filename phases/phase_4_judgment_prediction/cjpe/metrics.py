"""Case-ID alignment and fixed-binary-label judgment metrics."""
from __future__ import annotations


import numpy as np


def validate_predictions(split_cache, predictions, require_order=True):
    if type(require_order) is not bool or not isinstance(predictions, list):
        raise ValueError("Predictions must be a list; require_order must be boolean")
    expected = list(split_cache.ids)
    actual = []
    for row in predictions:
        if not isinstance(row, dict) or set(row) != {"id", "pred"}:
            raise ValueError("Each prediction needs exactly id and pred")
        if not isinstance(row["id"], str) or not row["id"]:
            raise ValueError("Prediction ID must be a nonempty string")
        if type(row["pred"]) is not int or row["pred"] not in (0, 1):
            raise ValueError("Predictions must be strict integers 0/1, excluding bool")
        actual.append(row["id"])
    if (
        len(actual) != len(expected)
        or len(set(actual)) != len(actual)
        or set(actual) != set(expected)
    ):
        raise ValueError("Predictions must cover every original case exactly once")
    if require_order and actual != expected:
        raise ValueError("Prediction order must match original case order")
    return {
        "documents": len(expected),
        "prediction_rows": len(actual),
        "unique_ids": len(set(actual)),
        "missing_ids": 0,
        "extra_ids": 0,
        "raw_order_preserved": actual == expected,
        "strict_integer_binary_predictions": True,
    }


def metrics_from_rows(split_cache, predictions):
    """Fixed-two-class Macro-F1; precision/recall are binary class-1 metrics."""
    if split_cache.split not in ("train", "dev"):
        raise ValueError("TEST targets must never be scored locally")
    validate_predictions(split_cache, predictions, require_order=False)
    labels = [record["label"] for record in split_cache.records]
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise ValueError("Scoring requires strict TRAIN/DEV integer targets")
    by_id = {row["id"]: row["pred"] for row in predictions}
    predicted = [by_id[case_id] for case_id in split_cache.ids]
    confusion = np.zeros((2, 2), dtype=np.int64)
    for truth, guess in zip(labels, predicted, strict=True):
        confusion[truth, guess] += 1
    support = confusion.sum(axis=1)
    selected = confusion.sum(axis=0)
    correct = np.diag(confusion)
    precision = np.divide(correct, selected, out=np.zeros(2, dtype=np.float64), where=selected != 0)
    recall = np.divide(correct, support, out=np.zeros(2, dtype=np.float64), where=support != 0)
    denominator = support + selected
    f1 = np.divide(
        2 * correct, denominator, out=np.zeros(2, dtype=np.float64), where=denominator != 0
    )
    return {
        "macro_f1": float(f1.mean()),
        "precision": float(precision[1]),
        "recall": float(recall[1]),
        "accuracy": float(correct.sum() / len(labels)),
        "documents": len(labels),
        "confusion_matrix": confusion.tolist(),
        "per_class": {
            str(label): {
                "precision": float(precision[label]),
                "recall": float(recall[label]),
                "f1": float(f1[label]),
                "support": int(support[label]),
            }
            for label in (0, 1)
        },
    }
