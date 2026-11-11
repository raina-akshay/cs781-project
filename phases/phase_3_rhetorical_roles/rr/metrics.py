"""Sentence reconstruction and fixed-13-label rhetorical-role metrics.

The public ``evaluate_phase3.py`` concatenates CL and IT sentence labels before
computing macro scores over all 13 vocabulary labels (including absent labels).
This module follows that definition without requiring scikit-learn at runtime.
The original document IDs and sentence positions are retained in exported JSON.
"""

from collections import Counter
from collections.abc import Mapping
from numbers import Integral


OFFICIAL_LABELS = (
    "Fact",
    "Issue",
    "ArgumentPetitioner",
    "ArgumentRespondent",
    "Statute",
    "Dissent",
    "PrecedentReliedUpon",
    "PrecedentNotReliedUpon",
    "PrecedentOverruled",
    "RulingByLowerCourt",
    "RatioOfTheDecision",
    "RulingByPresentCourt",
    "None",
)
DOMAINS = ("CL", "IT")


def _integer(value, description):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{description} must be an integer, not a boolean or float")
    return int(value)


def _selected_documents(cache, split):
    if split not in ("train", "dev", "test"):
        raise ValueError("split must be train, dev or test")
    labels, documents = cache["labels"], cache["documents"]
    return labels, [document for document in documents if document["split"] == split]


def _validate_selected(labels, documents, predictions):
    if not isinstance(predictions, Mapping):
        raise ValueError("Predictions must map original document IDs to label lists")
    expected_ids = {document["id"] for document in documents}
    if set(predictions) != expected_ids:
        missing = expected_ids - set(predictions)
        extra = set(predictions) - expected_ids
        raise ValueError(f"Prediction ID mismatch: missing={len(missing)}, extra={len(extra)}")
    allowed = set(labels)
    summary = {
        "documents": len(documents),
        "sentences": 0,
        "domains": {domain: {"documents": 0, "sentences": 0} for domain in DOMAINS},
    }
    for document in documents:
        doc_id = document["id"]
        values = predictions[doc_id]
        if not isinstance(values, list):
            raise ValueError(f"{doc_id}: predicted labels must be a list")
        if len(values) != len(document["sentences"]):
            raise ValueError(f"{doc_id}: sentence-count mismatch")
        if any(not isinstance(label, str) or label not in allowed for label in values):
            raise ValueError(f"{doc_id}: predictions contain an unknown or nonliteral label")
        count = len(values)
        summary["sentences"] += count
        summary["domains"][document["domain"]]["documents"] += 1
        summary["domains"][document["domain"]]["sentences"] += count
    return summary


def validate_predictions(cache, predictions, split):
    """Validate exact raw IDs, per-document counts and official label spellings.

    A plain label list cannot reveal whether a caller already reordered its
    predictions. Use :func:`reconstruct_predictions` to enforce source positions
    when collecting shuffled batches. Empty documents correctly require ``[]``.
    """
    labels, documents = _selected_documents(cache, split)
    return _validate_selected(labels, documents, predictions)


def reconstruct_predictions(cache, split, rows):
    """Build source-ordered predictions from ``(doc_id, sentence_index, label_id)``.

    Input rows may arrive in any batch order, but every selected sentence must
    occur exactly once. Boolean, floating-point and out-of-range IDs are rejected.
    """
    labels, documents = _selected_documents(cache, split)
    result = {document["id"]: [None] * len(document["sentences"]) for document in documents}
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) != 3:
            raise ValueError("Prediction rows must be (document ID, sentence index, label ID)")
        doc_id, sentence_index, label_id = row
        if not isinstance(doc_id, str) or doc_id not in result:
            raise ValueError("Prediction row has an unknown original document ID")
        sentence_index = _integer(sentence_index, "Prediction sentence index")
        label_id = _integer(label_id, "Prediction label ID")
        if not 0 <= sentence_index < len(result[doc_id]):
            raise ValueError(f"{doc_id}: prediction sentence index is outside the document")
        if not 0 <= label_id < len(labels):
            raise ValueError(f"{doc_id}: prediction label ID is outside the vocabulary")
        if result[doc_id][sentence_index] is not None:
            raise ValueError(f"{doc_id}: duplicate prediction for sentence {sentence_index}")
        result[doc_id][sentence_index] = labels[label_id]
    if any(label is None for values in result.values() for label in values):
        raise ValueError("Prediction rows do not cover every selected sentence")
    return result


def _scores(labels, documents, predictions):
    true_count, predicted_count, true_positive = Counter(), Counter(), Counter()
    sentences = 0
    for document in documents:
        for sentence, prediction in zip(document["sentences"], predictions[document["id"]]):
            if sentence["label"] is None:
                raise ValueError("Metrics require public ground-truth labels for every sentence")
            gold = labels[sentence["label"]]
            true_count[gold] += 1
            predicted_count[prediction] += 1
            if gold == prediction:
                true_positive[gold] += 1
            sentences += 1
    per_label = {}
    for label in labels:
        tp = true_positive[label]
        fp, fn = predicted_count[label] - tp, true_count[label] - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_label[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "p": p,
            "r": r,
            "f1": f1,
            "support": true_count[label],
        }
    return {
        "macro_f1": sum(row["f1"] for row in per_label.values()) / len(labels),
        "macro_precision": sum(row["p"] for row in per_label.values()) / len(labels),
        "macro_recall": sum(row["r"] for row in per_label.values()) / len(labels),
        "per_label": per_label,
        "sentences": sentences,
        "documents": len(documents),
    }


def metrics_from_predictions(cache, predictions, split="dev"):
    """Return combined official-style scores and separate CL/IT diagnostics."""
    labels, documents = _selected_documents(cache, split)
    _validate_selected(labels, documents, predictions)
    return {
        "combined": _scores(labels, documents, predictions),
        "domains": {
            domain: _scores(
                labels,
                [document for document in documents if document["domain"] == domain],
                predictions,
            )
            for domain in DOMAINS
        },
    }
