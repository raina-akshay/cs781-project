"""Phase 2 JSON/span checks and exact per-label macro scoring; no ML dependencies."""

import json
from pathlib import Path


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-standard JSON numeric value: {value}")


def load_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_unique_pairs,
        parse_constant=_reject_constant,
    )


def load_labels(path):
    labels = [
        x.strip()
        for x in Path(path).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if len(labels) != 12 or len(set(labels)) != 12:
        raise ValueError("Exactly 12 distinct official labels are required")
    return labels


def validate_texts(texts):
    if not isinstance(texts, dict) or not texts:
        raise ValueError("Texts must be a nonempty document-ID mapping")
    if any(
        not isinstance(k, str) or not k or not isinstance(v, str)
        for k, v in texts.items()
    ):
        raise ValueError(
            "Document IDs must be nonempty strings; text values must be strings"
        )
    # An empty text is a legitimate released test record. Preserve its ID.


def validate_spans(texts, annotations, labels):
    """Stricter than the scorer: also bounds-check against the original text.

    Optional entity_text must agree with text[start:end]. Never trim, repair,
    deduplicate, normalize, or silently drop an invalid prediction.
    """
    if len(labels) != 12 or len(set(labels)) != 12:
        raise ValueError("Exactly 12 distinct labels are required")
    if not isinstance(annotations, dict) or set(annotations) != set(texts):
        raise ValueError("Annotation IDs must exactly match document IDs")
    allowed = set(labels)
    for did, text in texts.items():
        rows = annotations[did]
        if not isinstance(rows, list):
            raise ValueError(f"{did}: entities must be a list (use [] for none)")
        seen = set()
        for i, row in enumerate(rows):
            if not isinstance(row, list) or len(row) not in (3, 4):
                raise ValueError(f"{did}:{i}: expected [start,end,label,(text)]")
            start, end, label = row[:3]
            if type(start) is not int or type(end) is not int:
                raise ValueError(f"{did}:{i}: offsets must be integers, not booleans")
            if not 0 <= start < end <= len(text):
                raise ValueError(f"{did}:{i}: offsets outside the original text")
            if not isinstance(label, str) or label not in allowed:
                raise ValueError(f"{did}:{i}: unknown label")
            if len(row) == 4 and (
                not isinstance(row[3], str) or text[start:end] != row[3]
            ):
                raise ValueError(f"{did}:{i}: entity text disagrees with offsets")
            span = (start, end, label)
            if span in seen:
                raise ValueError(f"{did}:{i}: duplicate span")
            seen.add(span)


def score(texts, truth, predictions, labels):
    """Score validated spans over 12 labels, including absent classes."""
    stats = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    for did in texts:
        gold = {tuple(row[:3]) for row in truth[did]}
        pred = {tuple(row[:3]) for row in predictions[did]}
        for spans, kind in (
            (gold & pred, "tp"),
            (pred - gold, "fp"),
            (gold - pred, "fn"),
        ):
            for _, _, label in spans:
                stats[label][kind] += 1
    for row in stats.values():
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        row.update(
            precision=precision,
            recall=recall,
            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        )
    return {
        "examples": len(texts),
        "per_label": stats,
        **{
            f"strict_macro_{metric}": sum(x[metric] for x in stats.values()) / 12
            for metric in ("f1", "precision", "recall")
        },
    }
