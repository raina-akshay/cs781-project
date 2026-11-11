"""Solution 7 objectives; every auxiliary has an explicit normalization unit.

BIO CE/boundary BCE, complete-entity role CE and boundary ranking are unchanged.
Direct span ranking supervises the BIO emissions used by the existing decoder.
"""

from .s4_loss import (
    prepare_loss as prepare_previous_loss,
    summed_loss,
    summed_role_loss,
    summed_ranking_loss,
)
from .s7_span import SPAN_OFFSETS, summed_span_loss, validate_offsets


def prepare_loss(config):
    weights, metadata = prepare_previous_loss(config)
    strength = config["span_loss"]["lambda"]
    validate_offsets(config["span_loss"]["offsets"])
    return weights, {
        **metadata,
        "lambda_span": float(strength),
        "span_offsets": list(SPAN_OFFSETS),
        "span_objective": "softplus_corrupted_minus_gold_BIO_path_score_over_same_union_hull",
        "span_normalization": "independent_global_valid_span_comparison_count",
        "span_negative_exclusion": "intersection_with_any_other_full_document_gold_entity",
        "span_assignment": "one_maximum_context_complete_entity_window",
        "span_logits_used_for_decoding": True,
    }
