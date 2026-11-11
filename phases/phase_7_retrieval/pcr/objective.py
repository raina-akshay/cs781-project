"""Query-balanced supervision with complete positive masks and mixed negatives."""
import hashlib
import math
import numpy as np
import torch
from torch.nn import functional as F


def query_seed(seed, qid):
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{qid}".encode()).digest()[:8], "little"
    )


def sampled_candidates(
    raw, query_indices, epoch, training, dense_rankings, lexical_rankings
):
    if raw.split != "train" or raw.labels is None:
        raise ValueError("Candidate sampling requires TRAIN relevance labels")
    positions = {c: i for i, c in enumerate(raw.candidate_ids)}
    candidates, links = set(), []
    for qi in query_indices:
        q = raw.query_ids[int(qi)]
        positives = raw.labels[q]
        if not positives:
            raise ValueError("Zero-positive query cannot enter positive loss")
        seed = training["seed"]
        permutation = np.random.default_rng(query_seed(seed, q)).permutation(
            len(positives)
        )
        count = min(
            len(positives),
            max(2, math.ceil(len(positives) / training["positive_cycle_epochs"])),
        )
        chosen = [
            positives[int(permutation[((epoch - 1) * count + j) % len(permutation)])]
            for j in range(count)
        ]
        links.extend((q, c) for c in chosen)
        candidates.update(positions[c] for c in chosen)
        excluded = {positions[c] for c in positives} | {positions[q]}
        rng = np.random.default_rng(query_seed(seed + epoch * 100003, q))
        for rankings, number in [
            (dense_rankings, training["dense_negatives"]),
            (lexical_rankings, training["lexical_negatives"]),
        ]:
            pool = [int(c) for c in rankings[int(qi)] if int(c) not in excluded]
            pool = pool[: training["mining_pool_depth"]]
            if len(pool) < number:
                raise ValueError("Insufficient eligible hard negatives")
            selected = rng.choice(pool, size=number, replace=False).tolist()
            candidates.update(selected)
            excluded.update(selected)
        pool = np.array([i for i in range(len(raw.candidate_ids)) if i not in excluded])
        selected = rng.choice(
            pool, size=training["random_negatives"], replace=False
        ).tolist()
        candidates.update(selected)
    return sorted(candidates), links


def retrieval_loss_scores(scores, query_ids, candidate_ids, labels, temperature=0.05):
    """SUM per-query means of positive-versus-valid-negative InfoNCE.

    Each positive competes only with negatives, never with another known
    positive. Extra positives introduced by another query's candidates remain
    positive targets. The trainer divides once by the actual global query count.
    """
    if scores.shape != (len(query_ids), len(candidate_ids)) or temperature <= 0:
        raise ValueError("Invalid loss shape/temperature")
    logits = scores.float() / temperature
    result = logits.new_zeros(())
    for i, q in enumerate(query_ids):
        gold = set(labels[q])
        positive = torch.tensor(
            [c in gold for c in candidate_ids], device=logits.device
        )
        negative = torch.tensor(
            [c not in gold and c != q for c in candidate_ids], device=logits.device
        )
        if not bool(positive.any()) or not bool(negative.any()):
            raise ValueError("Every query requires positive and valid negative support")
        result = (
            result
            + F.softplus(
                torch.logsumexp(logits[i, negative], 0) - logits[i, positive]
            ).mean()
        )
    return result
