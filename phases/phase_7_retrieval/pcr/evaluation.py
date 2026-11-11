"""Strict PCR ranking, exact public-evaluator scoring, and named diagnostics."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def official_evaluator(path: Path | None = None):
    if path is None:
        path = Path(__file__).resolve().parents[1] / "evaluation/evaluate_phase7.py"
    spec = importlib.util.spec_from_file_location("official_pcr_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rank_scores(
    scores: np.ndarray,
    query_ids: list[str],
    candidate_ids: list[str],
    k: int | None = None,
    exclude_self: bool = True,
) -> np.ndarray:
    scores = np.asarray(scores)
    if scores.shape != (len(query_ids), len(candidate_ids)):
        raise ValueError("Score dimensions do not match exact ID sequences")
    if len(set(query_ids)) != len(query_ids) or len(set(candidate_ids)) != len(
        candidate_ids
    ):
        raise ValueError("Duplicate IDs")
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("Invalid nonfinite ranking score")
    usable = scores.copy()
    positions = {c: i for i, c in enumerate(candidate_ids)}
    if exclude_self:
        for i, q in enumerate(query_ids):
            if q in positions:
                usable[i, positions[q]] = -np.inf
    required = len(candidate_ids) - int(exclude_self) if k is None else k
    if required < 1 or np.any(np.isfinite(usable).sum(axis=1) < required):
        raise ValueError("Insufficient finite nonself candidates")
    # Stable tie breaks by the preserved official candidate-file order.
    return np.argsort(-usable, axis=1, kind="stable")[:, :required].astype(np.int32)


def predictions_from_ranking(
    ranking: np.ndarray, query_ids: list[str], candidate_ids: list[str], k: int = 7
) -> dict[str, list[str]]:
    ranking = np.asarray(ranking)
    if ranking.ndim != 2 or ranking.shape[0] != len(query_ids) or ranking.shape[1] < k:
        raise ValueError("Ranking does not cover all queries and requested depth")
    if (
        ranking.dtype.kind not in "iu"
        or np.any(ranking < 0)
        or np.any(ranking >= len(candidate_ids))
    ):
        raise ValueError("Invalid candidate indices")
    return {
        q: [candidate_ids[int(j)] for j in ranking[i, :k]]
        for i, q in enumerate(query_ids)
    }


def evaluate_ranking(
    ranking: np.ndarray,
    query_ids: list[str],
    candidate_ids: list[str],
    labels: dict[str, list[str]],
    evaluator_path: Path | None = None,
) -> dict:
    if set(query_ids) != set(labels):
        raise ValueError("Gold/query key mismatch")
    official = official_evaluator(evaluator_path)
    pred = predictions_from_ranking(ranking, query_ids, candidate_ids)
    errors = official.validate(pred, labels, set(candidate_ids))
    if errors:
        raise ValueError("; ".join(errors))
    if any(q in ids for q, ids in pred.items()):
        raise ValueError("Local protocol prohibits same-ID self-retrieval")
    result = official.score(labels, pred)
    result.update(
        {
            "micro_f1_at_7": result["primary_metric_1"],
            "micro_precision_at_7": result["secondary_metric_1"],
            "micro_recall_at_7": result["secondary_metric_2"],
            "queries": len(query_ids),
            "gold_links": sum(map(len, labels.values())),
        }
    )
    for k in (50, 100):
        if ranking.shape[1] < k:
            continue
        selections = predictions_from_ranking(ranking, query_ids, candidate_ids, k)
        hits = [len(set(labels[q]) & set(selections[q])) for q in query_ids]
        total = result["gold_links"]
        result[f"micro_recall_at_{k}"] = sum(hits) / total if total else 0.0
        per_query = [h / len(labels[q]) for q, h in zip(query_ids, hits) if labels[q]]
        result[f"macro_recall_at_{k}_nonempty_gold"] = (
            float(np.mean(per_query)) if per_query else 0.0
        )
    return result
