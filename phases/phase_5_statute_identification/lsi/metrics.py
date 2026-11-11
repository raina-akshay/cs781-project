"""Metrics and exact threshold calibration for the 100 statute labels."""

import numpy as np

LABELS = 100


def targets(split_cache, indices=None):
    if split_cache.split == "test":
        raise ValueError("TEST has no targets")
    chosen = list(range(len(split_cache))) if indices is None else list(indices)
    y = np.zeros((len(chosen), LABELS), dtype=np.uint8)
    for n, index in enumerate(chosen):
        labels = split_cache.records[index]["labels"]
        if (
            not isinstance(labels, list)
            or labels != sorted(set(labels))
            or any(type(x) is not int or not 0 <= x < LABELS for x in labels)
        ):
            raise ValueError("Invalid canonical statute set")
        y[n, labels] = 1
    return y


def logit_array(split_cache, rows):
    if not isinstance(rows, list) or len(rows) != len(split_cache):
        raise ValueError("Logits must cover the complete original split")
    for identifier, row in zip(split_cache.ids, rows, strict=True):
        if not isinstance(row, dict) or set(row) != {"id", "logits"} or row["id"] != identifier:
            raise ValueError("Logit IDs/schema/order differ")
        if (
            not isinstance(row["logits"], list)
            or len(row["logits"]) != LABELS
            or any(type(x) not in (float, int) for x in row["logits"])
        ):
            raise ValueError("Exactly100 numeric logits required")
    a = np.asarray([row["logits"] for row in rows], dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite logits")
    return a


def validate_thresholds(values):
    if (
        not isinstance(values, list)
        or len(values) != LABELS
        or any(type(x) not in (float, int) for x in values)
    ):
        raise ValueError("Exactly100 finite real logit thresholds required")
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite thresholds")
    return a


def decode(ids, values, thresholds):
    a = np.asarray(values, dtype=np.float64)
    threshold = validate_thresholds(thresholds)
    if a.shape != (len(ids), LABELS) or not np.isfinite(a).all():
        raise ValueError("Invalid decoding matrix")
    return [
        {"id": identifier, "pred": np.flatnonzero(row >= threshold).tolist()}
        for identifier, row in zip(ids, a, strict=True)
    ]


def validate_predictions(split_cache, predictions, require_order=True):
    if not isinstance(predictions, list) or len(predictions) != len(split_cache):
        raise ValueError("Predictions must cover every original case")
    seen = []
    for row in predictions:
        if (
            not isinstance(row, dict)
            or set(row) != {"id", "pred"}
            or not isinstance(row["id"], str)
            or not row["id"]
        ):
            raise ValueError("Invalid prediction schema")
        labels = row["pred"]
        if (
            not isinstance(labels, list)
            or any(type(x) is not int or not 0 <= x < LABELS for x in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("Predicted statutes must be unique integer IDs0..99")
        seen.append(row["id"])
    if (
        len(set(seen)) != len(seen)
        or set(seen) != set(split_cache.ids)
        or (require_order and seen != split_cache.ids)
    ):
        raise ValueError("Missing, extra, repeated or reordered case IDs")
    return {
        "documents": len(seen),
        "unique_ids": len(set(seen)),
        "original_order": seen == split_cache.ids,
        "strict_integer_statute_sets": True,
    }


def score_arrays(y, predicted):
    y, predicted = np.asarray(y), np.asarray(predicted)
    actual, guessed = y.astype(bool), predicted.astype(bool)
    tp = (actual & guessed).sum(0).astype(np.int64)
    fp = (~actual & guessed).sum(0).astype(np.int64)
    fn = (actual & ~guessed).sum(0).astype(np.int64)
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros(LABELS), where=denominator != 0)
    micro_den = int(denominator.sum())
    return {
        "macro_f1": float(f1.mean()),
        "micro_f1": float(2 * tp.sum() / micro_den) if micro_den else 0.0,
        "subset_accuracy": float(np.all(actual == guessed, axis=1).mean()),
        "documents": len(y),
        "num_labels": LABELS,
        "mean_predicted_cardinality": float(guessed.sum(1).mean()),
        "per_label": {
            str(i): {
                "tp": int(tp[i]),
                "fp": int(fp[i]),
                "fn": int(fn[i]),
                "support": int(actual[:, i].sum()),
                "f1": float(f1[i]),
            }
            for i in range(LABELS)
        },
    }


def metrics_from_rows(split_cache, predictions, indices=None):
    validate_predictions(split_cache, predictions, require_order=False)
    by_id = {row["id"]: row["pred"] for row in predictions}
    chosen = list(range(len(split_cache))) if indices is None else list(indices)
    guessed = np.zeros((len(chosen), LABELS), dtype=np.uint8)
    for n, index in enumerate(chosen):
        guessed[n, by_id[split_cache.ids[index]]] = 1
    return score_arrays(targets(split_cache, chosen), guessed)


def _best_binary_threshold(scores, truth):
    """Exact F1 sweep, ties in scores grouped; prefer the higher cutoff on F1 ties."""
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], truth[order]
    end = np.flatnonzero(np.r_[s[1:] != s[:-1], True])
    tp = np.cumsum(y, dtype=np.int64)[end]
    den = end + 1 + int(truth.sum())
    f1 = np.divide(2 * tp, den, out=np.zeros(len(end)), where=den != 0)
    best = int(np.argmax(f1))
    return float(s[end[best]])


def fit_thresholds(split_cache, logit_rows, minimum_positives=20):
    """Only the immutable DEV calibration panel is consulted; selection is untouched."""
    if split_cache.split != "dev" or minimum_positives != 20:
        raise ValueError("Only the declared DEV-calibration policy is supported")
    values = logit_array(split_cache, logit_rows)
    indices = [
        i for i, row in enumerate(split_cache.records) if row["dev_partition"] == "calibration"
    ]
    if not indices:
        raise ValueError("Empty calibration panel")
    x, y = values[indices], targets(split_cache, indices)
    # Global fallback uses a fixed logit grid, chosen by macro-F1 on calibration only.
    grid = np.linspace(-8.0, 4.0, 241)
    scores = [score_arrays(y, x >= t)["macro_f1"] for t in grid]
    maximum = max(scores)
    global_t = float(max(t for t, score in zip(grid, scores) if score >= maximum - 1e-12))
    cutoffs, supports = [], y.sum(0)
    for label in range(LABELS):
        n = int(supports[label])
        cutoffs.append(
            _best_binary_threshold(x[:, label], y[:, label])
            if n >= minimum_positives and len(y) - n >= minimum_positives
            else global_t
        )
    return {
        "format": "cs781-lsi-thresholds-v1",
        "thresholds": cutoffs,
        "comparison": "logit >= threshold",
        "global_fallback_logit": global_t,
        "minimum_positive_and_negative_support": minimum_positives,
        "calibration_cases": len(indices),
        "calibration_positive_supports": supports.tolist(),
        "selection_cases_used_for_threshold_fitting": 0,
        "test_used": False,
    }


def assess_dev(split_cache, logit_rows, threshold_state=None, train_text_hashes=None):
    state = fit_thresholds(split_cache, logit_rows) if threshold_state is None else threshold_state
    predictions = decode(split_cache.ids, logit_array(split_cache, logit_rows), state["thresholds"])
    result = {"threshold_state": state, "full_dev": metrics_from_rows(split_cache, predictions)}
    for panel in ("calibration", "selection"):
        indices = [i for i, row in enumerate(split_cache.records) if row["dev_partition"] == panel]
        result[panel] = metrics_from_rows(split_cache, predictions, indices)
    if train_text_hashes is not None:
        indices = [
            i
            for i, row in enumerate(split_cache.records)
            if row["text_sha256"] not in train_text_hashes
        ]
        result["train_text_nonoverlap"] = metrics_from_rows(split_cache, predictions, indices)
    return predictions, result
