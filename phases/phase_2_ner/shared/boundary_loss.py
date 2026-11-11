"""Unweighted BIO CE plus a modest, separately declared boundary objective.

The capped positive weights affect only two class-agnostic auxiliary BCEs.
The denominator remains the global count of owned characters, not the sum of
positive weights. No entity-type weighting is applied to the main BIO loss.
"""

import torch
from torch.nn import functional as F

from .boundary_data import load_cache


def prepare_loss(config):
    strength = config["boundary_loss"]["lambda"]
    maximum = config["boundary_loss"]["max_pos_weight"]
    role_strength = config["role_loss"]["lambda"]
    data = (
        load_cache(config["cache_path"], "train", role_auxiliary=True)
        if role_strength
        else load_cache(config["cache_path"], "train")
    )
    positives = torch.zeros(2, dtype=torch.int64)
    characters = 0
    for window in data.windows:
        targets = torch.as_tensor(
            window.boundary_targets[window.owned_mask], dtype=torch.int64
        )
        positives += targets.sum(0)
        characters += len(targets)
    expected = sum(map(len, data.texts.values()))
    if characters != expected:
        raise ValueError(
            "Frequency fit must count every training character exactly once"
        )
    negatives = characters - positives
    if bool((positives == 0).any()):
        raise ValueError("Each auxiliary channel requires positive training boundaries")
    weights = (
        (negatives.double() / positives.double())
        .sqrt()
        .clamp(1.0, float(maximum))
        .float()
    )
    return weights, {
        "main_objective": "unweighted_BIO_cross_entropy",
        "auxiliary_objective": "mean_of_start_end_BCEWithLogitsLoss",
        "lambda": float(strength),
        "lambda_boundary": float(strength),
        "lambda_role": float(role_strength),
        "role_auxiliary_enabled": bool(role_strength),
        "role_auxiliary_objective": "entity_mean_B_start_I_interior_role_cross_entropy",
        "role_normalization": "independent_global_complete_entity_count",
        "role_class_weights": None,
        "role_training_entity_count": sum(len(rows) for rows in data.gold.values()),
        "role_assignment": "complete_containing_window_maximum_minimum_character_context",
        "role_logits_used_for_decoding": False,
        "max_pos_weight": float(maximum),
        "fit_partition": "train",
        "training_document_ids": sorted(data.texts),
        "refit_reuses_training_partition_weights": True,
        "characters": characters,
        "positive_counts": positives.tolist(),
        "negative_counts": negatives.tolist(),
        "pos_weights": weights.tolist(),
        "positive_weight_formula": "clip(sqrt(negative_count/positive_count),1,max_pos_weight)",
        "normalization": "global_owned_character_count",
        "boundary_channels": ["start_character", "last_entity_character"],
        "boundary_logits_used_for_decoding": False,
    }


def summed_loss(outputs, labels, mask, weights, boundary_targets, lambda_boundary=0.1):
    """Return total sum, character denominator/count, raw BIO sum and aux sum.

    Aux sum is the mean of the two per-channel BCE sums before multiplying by
    lambda. Lambda-zero and ignored batches still retain both head graph paths,
    so DDP can use the same model without unused-parameter special cases.
    """
    logits, boundary_logits = outputs["logits"], outputs["boundary_logits"]
    if logits.shape[:2] != labels.shape or mask.shape != labels.shape:
        raise ValueError("BIO logits, labels and ownership mask disagree")
    if (
        boundary_logits.shape != (*labels.shape, 2)
        or boundary_targets.shape != boundary_logits.shape
    ):
        raise ValueError("Boundary logits/targets must be [batch, characters, 2]")
    valid = mask & labels.ne(-100)
    count = int(valid.sum())
    precision = torch.float64 if logits.dtype == torch.float64 else torch.float32
    pos_weights = weights.to(device=logits.device, dtype=precision)
    if (
        pos_weights.shape != (2,)
        or not bool(torch.isfinite(pos_weights).all())
        or bool((pos_weights < 1).any())
    ):
        raise ValueError(
            "Boundary positive weights must be two finite values at least one"
        )
    if not count:
        raw = logits.sum() * 0.0
        auxiliary = boundary_logits.sum() * 0.0
    else:
        targets = boundary_targets[valid].to(device=logits.device, dtype=precision)
        if not bool(torch.isfinite(targets).all()) or not bool(
            ((targets == 0) | (targets == 1)).all()
        ):
            raise ValueError(
                "Owned labelled characters require binary boundary targets"
            )
        raw = F.cross_entropy(
            logits[valid].to(precision), labels[valid], reduction="sum"
        )
        auxiliary = 0.5 * F.binary_cross_entropy_with_logits(
            boundary_logits[valid].to(precision),
            targets,
            pos_weight=pos_weights,
            reduction="sum",
        )
    total = raw + float(lambda_boundary) * auxiliary
    normalizer = logits.new_tensor(count, dtype=torch.float64)
    return total, normalizer, count, raw, auxiliary


def summed_role_loss(logits, role_entities, role_entity_mask=None):
    """Return summed role CE and entity count for an independent global mean.

    For a complete span [s,e), candidate type k has score
    ``(BIO[s,B-k] + sum(BIO[s+1:e,I-k])) / (e-s)``. There is one
    ordinary, unweighted type CE per entity, with no outside candidate. This is
    an auxiliary on existing BIO logits; BIO CE still calibrates entity versus
    outside decisions and the inference decoder is unchanged.

    The caller must mask synthetic DDP filler rows and normalize across all
    ranks and accumulation microbatches by the global entity count, separately
    from the owned-character objective. Empty local batches keep a zero graph.
    """
    if logits.ndim != 3 or logits.shape[-1] < 3 or logits.shape[-1] % 2 != 1:
        raise ValueError("Role loss requires BIO logits [batch, characters, 1+2*types]")
    if (
        role_entities.dtype != torch.long
        or role_entities.ndim != 2
        or role_entities.shape[1] != 4
    ):
        raise ValueError("Role entities must be long [entities, batch/start/end/type]")
    if role_entity_mask is None:
        role_entity_mask = torch.ones(
            role_entities.shape[0], dtype=torch.bool, device=role_entities.device
        )
    if role_entity_mask.dtype != torch.bool or role_entity_mask.shape != (
        role_entities.shape[0],
    ):
        raise ValueError("Role entity mask must be boolean [entities]")
    entities = role_entities[role_entity_mask].to(device=logits.device)
    count = entities.shape[0]
    precision = torch.float64 if logits.dtype == torch.float64 else torch.float32
    scores = logits.to(precision)
    if not count:
        return scores.sum() * 0.0, 0
    batch, start, end, target = entities.unbind(dim=1)
    types = (logits.shape[-1] - 1) // 2
    if (
        bool(((batch < 0) | (batch >= logits.shape[0])).any())
        or bool(((start < 0) | (end <= start) | (end > logits.shape[1])).any())
        or bool(((target < 0) | (target >= types)).any())
    ):
        raise ValueError("Role entity coordinates or type exceed BIO logits")
    # Prefix sums retain gradients through every complete entity character.
    inside_prefix = F.pad(scores[..., 2::2].cumsum(dim=1), (0, 0, 1, 0))
    interior = inside_prefix[batch, end] - inside_prefix[batch, start + 1]
    role_scores = (scores[batch, start, 1::2] + interior) / (end - start)[:, None]
    return F.cross_entropy(role_scores, target, reduction="sum"), count
