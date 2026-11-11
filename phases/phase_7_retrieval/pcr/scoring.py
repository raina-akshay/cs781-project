"""Directed chunk late interaction with a global semantic score.

The query-mean MaxSim principle follows the pinned M3 author implementation.
Our raw legal chunks, original own foundation and equal global/local mixture
are a task adaptation, not a reproduction of M3 or KELLER.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def document_vectors(vectors, owners, n_documents, max_chunks):
    """Pack contiguous normalized chunk vectors without detaching gradients."""
    owners = torch.as_tensor(owners, device=vectors.device, dtype=torch.long)
    counts = torch.bincount(owners, minlength=n_documents)
    if (
        len(owners) != len(vectors)
        or len(counts) != n_documents
        or bool((counts < 1).any())
        or bool((counts > max_chunks).any())
    ):
        raise ValueError("Invalid document/chunk ownership")
    expected = torch.repeat_interleave(
        torch.arange(n_documents, device=vectors.device), counts
    )
    if not torch.equal(expected, owners):
        raise ValueError("Chunks must be contiguous in document order")
    pieces = vectors.split(counts.cpu().tolist())
    packed = torch.stack([F.pad(x, (0, 0, 0, max_chunks - len(x))) for x in pieces])
    mask = torch.arange(max_chunks, device=vectors.device)[None] < counts[:, None]
    return packed, mask


def score_components(query, query_mask, candidate, candidate_mask, local_weight=0.5):
    """Return [queries,candidates] scores; candidates may have fewer chunks.

    Invalid candidate slots are -inf before max, never zero. Only actual query
    chunks enter the mean. Neither length padding nor chunk count changes the
    global mean denominator. Candidate-max opportunity bias still exists for
    documents with genuinely more chunks; the blend does not mathematically
    eliminate that effect.
    """
    if (
        query.ndim != 3
        or candidate.ndim != 3
        or query.shape[-1] != candidate.shape[-1]
        or query_mask.shape != query.shape[:2]
        or candidate_mask.shape != candidate.shape[:2]
        or query_mask.dtype != torch.bool
        or candidate_mask.dtype != torch.bool
        or not 0 <= local_weight <= 1
        or bool((query_mask.sum(1) == 0).any())
        or bool((candidate_mask.sum(1) == 0).any())
    ):
        raise ValueError("Invalid late-interaction representations or masks")
    q, c = query.float(), candidate.float()
    qcount, ccount = query_mask.sum(1), candidate_mask.sum(1)
    qglobal = F.normalize(
        (q * query_mask[..., None]).sum(1) / qcount[:, None], dim=-1, eps=1e-8
    )
    cglobal = F.normalize(
        (c * candidate_mask[..., None]).sum(1) / ccount[:, None], dim=-1, eps=1e-8
    )
    global_score = qglobal @ cglobal.T
    dots = q.reshape(-1, q.shape[-1]) @ c.reshape(-1, c.shape[-1]).T
    dots = dots.reshape(len(q), q.shape[1], len(c), c.shape[1]).permute(0, 2, 1, 3)
    dots = dots.masked_fill(~candidate_mask[None, :, None, :], -torch.inf)
    best = dots.max(-1).values
    local_score = (best * query_mask[:, None, :]).sum(-1) / qcount[:, None]
    return {
        "global": global_score,
        "local": local_score,
        "joint": (1 - local_weight) * global_score + local_weight * local_score,
    }
