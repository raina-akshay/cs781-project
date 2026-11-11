"""Near-boundary comparisons on original character positions.

The supervised auxiliaries assign each complete entity to the same maximum-
context window. Negatives are nearby real characters, excluding every true
same-channel boundary in the complete gold document, not merely assigned spans.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .boundary_data import (
    assign_complete_role_entities,
    collate_windows as boundary_collate,
    document_boundary_targets,
    load_cache as load_boundary_cache,
)


RANK_OFFSETS = (-2, -1, 1, 2)


def validate_offsets(offsets):
    if (
        not isinstance(offsets, (list, tuple))
        or any(type(value) is not int for value in offsets)
        or tuple(offsets) != RANK_OFFSETS
    ):
        raise ValueError("Ranking offsets must be exactly [-2, -1, 1, 2]")
    return RANK_OFFSETS


def assign_boundary_comparisons(
    windows, texts, gold, labels, *, offsets=RANK_OFFSETS, role_entities=None
):
    """Return [channel, positive_character, negative_character] per window.

    Positions are relative original-Unicode character indices. Channels 0 and
    1 denote the first and last included entity character. Annotation ``end``
    is exclusive, so channel 1's positive is ``end - 1``. Cross-window ownership
    does not clip this complete-entity auxiliary or duplicate entity anchors.
    """
    offsets = validate_offsets(offsets)
    targets = document_boundary_targets(texts, gold, labels)
    if role_entities is None:
        role_entities = assign_complete_role_entities(windows, texts, gold, labels)
    if len(role_entities) != len(windows):
        raise ValueError("Complete entity assignment and window count disagree")
    result = []
    for window, entities in zip(windows, role_entities):
        size = window.char_end - window.char_start
        full_targets = targets[window.document_id]
        rows = []
        for start, end, _ in entities:
            start, end = int(start), int(end)
            if not 0 <= start < end <= size:
                raise ValueError("Ranking anchors require a complete window entity")
            for channel, positive in ((0, start), (1, end - 1)):
                if full_targets[window.char_start + positive, channel] != 1:
                    raise ValueError("Ranking positive is not a true document boundary")
                for offset in offsets:
                    negative = positive + offset
                    if (
                        0 <= negative < size
                        and full_targets[window.char_start + negative, channel] == 0
                    ):
                        rows.append((channel, positive, negative))
        result.append(np.asarray(rows, dtype=np.int64).reshape(-1, 3))
    return result


@dataclass(frozen=True)
class RankingWindow:
    window: object
    ranking_comparisons: object

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "window"), name)


class RankingDataset(Dataset):
    """Add near-boundary comparisons to character windows."""

    def __init__(self, base, *, offsets=RANK_OFFSETS):
        self._base = base
        validate_offsets(offsets)
        if base.gold is None:
            comparisons = [np.empty((0, 3), dtype=np.int64) for _ in base.windows]
        else:
            assigned = [
                getattr(window, "role_entities", None) for window in base.windows
            ]
            # Ranking remains well-defined when the independent role CE is off.
            if any(rows is None for rows in assigned):
                assigned = assign_complete_role_entities(
                    base.windows, base.texts, base.gold, base.labels
                )
            comparisons = assign_boundary_comparisons(
                base.windows,
                base.texts,
                base.gold,
                base.labels,
                offsets=offsets,
                role_entities=assigned,
            )
        self.windows = [
            RankingWindow(window, rows)
            for window, rows in zip(base.windows, comparisons)
        ]

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_base"), name)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index]


def load_cache(path, partition, *, role_auxiliary=True, offsets=RANK_OFFSETS):
    return RankingDataset(
        load_boundary_cache(path, partition, role_auxiliary=role_auxiliary),
        offsets=offsets,
    )


def collate_windows(windows):
    """Always expose ranking tensors, including controls and unlabelled data.

    Synthetic distributed filler rows are marked by the trainer after collation,
    through ``rank_comparison_mask`` independently of owned-character masks.
    """
    batch = boundary_collate(windows)
    comparisons = []
    for row, window in enumerate(windows):
        entries = getattr(window, "ranking_comparisons", None)
        if entries is None:
            continue
        if (
            entries.ndim != 2
            or entries.shape[1] != 3
            or not np.issubdtype(entries.dtype, np.integer)
        ):
            raise ValueError(
                "Ranking comparisons must be integer [N, channel/positive/negative]"
            )
        if entries.size:
            channels, positives, negatives = entries.T
            if (
                bool(((channels < 0) | (channels > 1)).any())
                or bool(((positives < 0) | (positives >= len(window.char_ids))).any())
                or bool(((negatives < 0) | (negatives >= len(window.char_ids))).any())
                or bool((positives == negatives).any())
            ):
                raise ValueError("Ranking comparison exceeds actual window text")
        comparisons.extend(
            (row, int(channel), int(positive), int(negative))
            for channel, positive, negative in entries
        )
    batch["rank_comparisons"] = torch.tensor(comparisons, dtype=torch.long).reshape(
        -1, 4
    )
    batch["rank_comparison_mask"] = torch.ones(len(comparisons), dtype=torch.bool)
    return batch
