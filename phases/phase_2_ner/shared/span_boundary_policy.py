"""Isolated, deterministic boundary policies for already-decoded NER spans.

These are inference transformations, not tokenizer changes or new BIO decoding.
The sole candidate advances a span's start over Unicode ``str.isspace``
characters. Its end is never changed: training annotations contain legitimate
trailing whitespace. The original text and input predictions remain untouched.
"""
import copy

from .ner_contract import validate_spans

POLICIES = ("raw", "leading_whitespace_only")


def apply_policy(texts, predictions, labels, policy):
    """Trim leading whitespace and return the changed span counts.

    An entirely whitespace prediction is dropped only under the leading-only
    policy because advancing its start would leave an invalid empty span.
    Optional entity-text fields are retained and updated to the new exact slice.
    Duplicate exact spans created by a transformation cause an explicit error;
    silently deduplicating would change model predictions and hide a defect.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown boundary policy: {policy!r}")
    output = copy.deepcopy(predictions)
    counts = {
        "policy": policy,
        "input_entities": 0,
        "output_entities": 0,
        "changed_retained_spans": 0,
        "dropped_all_whitespace_spans": 0,
        "leading_characters_removed": 0,
        "retained_span_ends_preserved": True,
    }
    for did, rows in predictions.items():
        revised = []
        text = texts[did]
        for row in rows:
            counts["input_entities"] += 1
            start, end, label = row[:3]
            revised_start = start
            if policy == "leading_whitespace_only":
                while revised_start < end and text[revised_start].isspace():
                    revised_start += 1
            counts["leading_characters_removed"] += revised_start - start
            if revised_start == end:
                counts["dropped_all_whitespace_spans"] += 1
                continue
            revised_row = [revised_start, end, label]
            if len(row) == 4:
                revised_row.append(text[revised_start:end])
            revised.append(revised_row)
            counts["changed_retained_spans"] += int(revised_start != start)
        output[did] = revised
        counts["output_entities"] += len(revised)
    validate_spans(texts, output, labels)
    return output, counts
