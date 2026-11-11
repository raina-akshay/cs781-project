"""Gold-versus-nearby BIO span ranking on exact original character positions.

This objective acts on the classifier used by the unchanged BIO decoder.  It
compares complete local label paths on the same interval: their shared outside
labels cancel, leaving the entity-versus-O span score.  No length averaging,
external predictions or inference-time boundary repair is involved.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .boundary_data import assign_complete_role_entities
from .s4_data import (
    collate_windows as ranking_collate,
    load_cache as load_ranking_cache,
)


SPAN_OFFSETS = (-2, -1, 0, 1, 2)
SPAN_VERSION = 1
SPAN_COLUMNS = (
    "row",
    "gold_start",
    "gold_end",
    "negative_start",
    "negative_end",
    "type_index",
)


def validate_offsets(offsets):
    if (
        not isinstance(offsets, (list, tuple))
        or any(type(value) is not int for value in offsets)
        or tuple(offsets) != SPAN_OFFSETS
    ):
        raise ValueError("Span offsets must be exactly [-2, -1, 0, 1, 2]")
    return SPAN_OFFSETS


def assign_span_comparisons(
    windows, texts, gold, labels, *, offsets=SPAN_OFFSETS, role_entities=None
):
    """Return [gold_start, gold_end, negative_start, negative_end, type].

    Every anchor uses the existing unique maximum-context containing window.
    Negatives which exceed that window are omitted, never reassigned.  They
    must not intersect any *other* gold entity anywhere in the original
    document, including an entity not assigned to or complete in this window.
    All ends are exclusive and all coordinates are original Unicode indices.
    """
    offsets = validate_offsets(offsets)
    # The official contract validates coordinates but allows overlap; local
    # replacement paths require flat gold, just like the character BIO task.
    document_entities = {}
    for document_id, spans in gold.items():
        ordered = sorted((row[0], row[1], labels.index(row[2])) for row in spans)
        previous_end = 0
        for start, end, _ in ordered:
            if start < previous_end:
                raise ValueError("Span ranking cannot encode overlapping gold entities")
            previous_end = end
        document_entities[document_id] = ordered
    if role_entities is None:
        role_entities = assign_complete_role_entities(windows, texts, gold, labels)
    if len(role_entities) != len(windows):
        raise ValueError("Span entity assignment and window count disagree")
    result = []
    for window, entities in zip(windows, role_entities):
        size = window.char_end - window.char_start
        if (
            window.document_id not in texts
            or not 0
            <= window.char_start
            < window.char_end
            <= len(texts[window.document_id])
        ):
            raise ValueError("Span window exceeds its original document")
        rows = []
        full = document_entities[window.document_id]
        for raw_start, raw_end, raw_type in entities:
            start, end, kind = int(raw_start), int(raw_end), int(raw_type)
            anchor = (window.char_start + start, window.char_start + end, kind)
            if not 0 <= start < end <= size or anchor not in full:
                raise ValueError("Span anchor must be a complete original gold entity")
            others = [(s, e) for s, e, k in full if (s, e, k) != anchor]
            for start_offset in offsets:
                for end_offset in offsets:
                    if start_offset == end_offset == 0:
                        continue
                    negative_start, negative_end = (
                        start + start_offset,
                        end + end_offset,
                    )
                    if not 0 <= negative_start < negative_end <= size:
                        continue
                    absolute_start = window.char_start + negative_start
                    absolute_end = window.char_start + negative_end
                    if any(
                        absolute_start < other_end and other_start < absolute_end
                        for other_start, other_end in others
                    ):
                        continue
                    rows.append((start, end, negative_start, negative_end, kind))
        result.append(np.asarray(rows, dtype=np.int64).reshape(-1, 5))
    return result


@dataclass(frozen=True)
class SpanWindow:
    window: object
    span_comparisons: object

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "window"), name)


class SpanDataset(Dataset):
    """Add direct-span comparisons to boundary-ranking windows."""

    def __init__(self, base, *, offsets=SPAN_OFFSETS):
        self._base = base
        validate_offsets(offsets)
        if base.gold is None:
            comparisons = [np.empty((0, 5), dtype=np.int64) for _ in base.windows]
            anchor_count = 0
        else:
            assigned = [
                getattr(window, "role_entities", None) for window in base.windows
            ]
            if any(rows is None for rows in assigned):
                assigned = assign_complete_role_entities(
                    base.windows, base.texts, base.gold, base.labels
                )
            comparisons = assign_span_comparisons(
                base.windows,
                base.texts,
                base.gold,
                base.labels,
                offsets=offsets,
                role_entities=assigned,
            )
            anchor_count = sum(len(rows) for rows in assigned)
        self.windows = [
            SpanWindow(window, rows) for window, rows in zip(base.windows, comparisons)
        ]
        self.span_metadata = {
            "span_version": SPAN_VERSION,
            "span_offsets": list(offsets),
            "span_offset_grid": "cartesian_start_end_excluding_0_0",
            "span_anchor_assignment": "unique_complete_role_maximum_minimum_context",
            "span_negative_exclusion": "intersection_any_other_full_document_gold_entity",
            "span_window_edge_policy": "filter_noncontaining_negative_without_reassignment",
            "span_coordinates": "original_unicode_end_exclusive",
            "span_comparison_columns": list(SPAN_COLUMNS),
            "span_anchor_count": anchor_count,
            "span_comparison_count": sum(len(rows) for rows in comparisons),
        }

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_base"), name)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index]


def load_cache(path, partition, *, role_auxiliary=True, offsets=SPAN_OFFSETS):
    return SpanDataset(
        load_ranking_cache(path, partition, role_auxiliary=role_auxiliary),
        offsets=offsets,
    )


def collate_windows(windows):
    """Preserve all existing masks and expose [N,6] BIO span comparisons."""
    batch = ranking_collate(windows)
    comparisons = []
    for row, window in enumerate(windows):
        entries = getattr(window, "span_comparisons", None)
        if entries is None:
            continue
        if (
            entries.ndim != 2
            or entries.shape[1] != 5
            or not np.issubdtype(entries.dtype, np.integer)
        ):
            raise ValueError("Span comparisons must be integer [N,5]")
        if entries.size:
            gs, ge, ns, ne, kind = entries.T
            size = len(window.char_ids)
            if (
                bool(
                    (
                        (gs < 0)
                        | (ge <= gs)
                        | (ge > size)
                        | (ns < 0)
                        | (ne <= ns)
                        | (ne > size)
                        | (kind < 0)
                    ).any()
                )
                or bool(((gs == ns) & (ge == ne)).any())
                or bool((np.abs(ns - gs) > 2).any())
                or bool((np.abs(ne - ge) > 2).any())
            ):
                raise ValueError("Span comparison exceeds actual window or offset grid")
        comparisons.extend((row, *map(int, entry)) for entry in entries)
    batch["span_comparisons"] = torch.tensor(comparisons, dtype=torch.long).reshape(
        -1, 6
    )
    batch["span_comparison_mask"] = torch.ones(len(comparisons), dtype=torch.bool)
    return batch


def _valid_comparisons(comparisons, mask):
    if (
        comparisons.dtype != torch.long
        or comparisons.ndim != 2
        or comparisons.shape[1] != 6
    ):
        raise ValueError("Span comparisons must be long [N,6]")
    if mask is None:
        mask = torch.ones(
            comparisons.shape[0], dtype=torch.bool, device=comparisons.device
        )
    if mask.dtype != torch.bool or mask.shape != (comparisons.shape[0],):
        raise ValueError("Span comparison mask must be boolean [N]")
    if mask.device != comparisons.device:
        raise ValueError("Span comparisons and mask must share a device")
    return comparisons[mask]


def summed_span_loss(logits, comparisons, mask=None):
    """Return summed softplus(q_negative-q_gold), and valid comparison count.

    q(s,e,k) = (z[s,B-k]-z[s,O]) + sum(z[s+1:e,I-k]-z[s+1:e,O]).
    Equal-position log-softmax normalizers and common outside labels cancel.
    No character-length normalization appears.  FP16/BF16 inputs accumulate in
    FP32; FP64 inputs retain precision for independent numerical gradient tests.
    """
    if (
        logits.ndim != 3
        or not logits.is_floating_point()
        or logits.shape[-1] < 3
        or logits.shape[-1] % 2 != 1
    ):
        raise ValueError(
            "Span loss requires floating BIO logits [batch,characters,1+2K]"
        )
    selected = _valid_comparisons(comparisons, mask).to(logits.device)
    scores = logits if logits.dtype == torch.float64 else logits.float()
    if not len(selected):
        return scores.sum() * 0.0, 0
    row, gs, ge, ns, ne, kind = selected.unbind(dim=1)
    if (
        bool(
            (
                (row < 0)
                | (row >= scores.shape[0])
                | (gs < 0)
                | (ge <= gs)
                | (ge > scores.shape[1])
                | (ns < 0)
                | (ne <= ns)
                | (ne > scores.shape[1])
                | (kind < 0)
                | (kind >= (scores.shape[-1] - 1) // 2)
            ).any()
        )
        or bool(((gs == ns) & (ge == ne)).any())
        or bool(((ns - gs).abs() > 2).any())
        or bool(((ne - ge).abs() > 2).any())
    ):
        raise ValueError("Span comparison coordinates exceed BIO logits or offset grid")
    begin, inside = 1 + 2 * kind, 2 + 2 * kind
    # Cancel shared interiors analytically.  Each endpoint moves by at most two
    # characters, so only changed positions are summed.  This is algebraically
    # q_negative-q_gold and avoids subtracting long, nearly equal prefix sums.
    difference = (
        scores[row, ns, begin]
        - scores[row, ns, 0]
        - scores[row, gs, begin]
        + scores[row, gs, 0]
    )
    for step in (0, 1):
        # Difference of upper prefix endpoints: P(ne) - P(ge).
        tail_position = torch.minimum(ne, ge) + step
        tail_valid = step < (ne - ge).abs()
        tail_position = tail_position.clamp(max=scores.shape[1] - 1)
        tail = scores[row, tail_position, inside] - scores[row, tail_position, 0]
        difference = difference + tail * (ne - ge).sign() * tail_valid
        # Difference of lower endpoints: P(gs+1) - P(ns+1).
        head_position = torch.minimum(gs, ns) + 1 + step
        head_valid = step < (ns - gs).abs()
        head_position = head_position.clamp(max=scores.shape[1] - 1)
        head = scores[row, head_position, inside] - scores[row, head_position, 0]
        difference = difference - head * (ns - gs).sign() * head_valid
    return F.softplus(difference).sum(), len(selected)
