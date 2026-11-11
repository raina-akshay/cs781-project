"""Boundary ranking with an independent comparison denominator.

Ranking compares the same-channel
positive and nearby negative boundary logits using stable FP32 softplus. The
trainer divides its sum by the global valid comparison count across ranks and
accumulation microbatches, independently of BIO characters and role entities.
"""

import torch
from torch.nn import functional as F

from .boundary_loss import (
    prepare_loss as prepare_boundary_loss,
    summed_loss,
    summed_role_loss,
)
from .s4_data import RANK_OFFSETS, validate_offsets


def prepare_loss(config):
    strength = config["ranking_loss"]["lambda"]
    offsets = validate_offsets(config["ranking_loss"]["offsets"])
    weights, metadata = prepare_boundary_loss(config)
    return weights, {
        **metadata,
        "lambda_ranking": float(strength),
        "ranking_auxiliary_enabled": bool(strength),
        "ranking_auxiliary_objective": "sum_softplus_same_channel_negative_minus_gold_boundary_logit",
        "ranking_normalization": "independent_global_valid_comparison_count",
        "ranking_offsets": list(offsets),
        "ranking_assignment": "complete_containing_window_maximum_minimum_character_context",
        "ranking_negative_exclusion": "all_true_same_channel_boundaries_in_full_gold_document",
        "ranking_end_coordinate": "exclusive_annotation_end_minus_one",
        "ranking_logits_used_for_decoding": False,
    }


def _valid_comparisons(comparisons, mask):
    if (
        comparisons.dtype != torch.long
        or comparisons.ndim != 2
        or comparisons.shape[1] != 4
    ):
        raise ValueError(
            "Ranking comparisons must be long [N, batch/channel/positive/negative]"
        )
    if mask is None:
        mask = torch.ones(
            comparisons.shape[0], dtype=torch.bool, device=comparisons.device
        )
    if mask.dtype != torch.bool or mask.shape != (comparisons.shape[0],):
        raise ValueError("Ranking comparison mask must be boolean [N]")
    return comparisons[mask]


def summed_ranking_loss(boundary_logits, rank_comparisons, rank_comparison_mask=None):
    """Return stable FP32 summed pairwise ranking loss and valid pair count.

    Gradient descent raises gold logits relative to their same-channel negatives.
    Empty or wholly masked batches retain a differentiable zero connected to the
    boundary head, so distributed zero-contribution ranks participate normally.
    """
    if (
        boundary_logits.ndim != 3
        or boundary_logits.shape[-1] != 2
        or not boundary_logits.is_floating_point()
    ):
        raise ValueError(
            "Ranking requires floating boundary logits [batch, characters, 2]"
        )
    comparisons = _valid_comparisons(rank_comparisons, rank_comparison_mask).to(
        boundary_logits.device
    )
    count = len(comparisons)
    scores = boundary_logits.float()
    if not count:
        return scores.sum() * 0.0, 0
    row, channel, positive, negative = comparisons.unbind(dim=1)
    if (
        bool(((row < 0) | (row >= scores.shape[0])).any())
        or bool(((channel < 0) | (channel >= 2)).any())
        or bool(((positive < 0) | (positive >= scores.shape[1])).any())
        or bool(((negative < 0) | (negative >= scores.shape[1])).any())
        or bool((positive == negative).any())
    ):
        raise ValueError(
            "Ranking comparison coordinates exceed boundary logits or repeat the anchor"
        )
    difference = scores[row, negative, channel] - scores[row, positive, channel]
    return F.softplus(difference).sum(), count
