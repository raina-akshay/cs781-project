"""Exact-document start/end targets for the isolated boundary auxiliary task.

The original cache, windows and character ownership are unchanged. In
particular, a window cutting through an entity must not invent an entity end.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .ner_data import load_cache as load_base_cache, collate_windows as base_collate


def document_boundary_targets(texts, gold, labels):
    """Build binary [characters, start/end] targets before cropping any window.

    An exclusive gold end ``e`` marks the last entity character ``e-1``. This
    retains leading/trailing whitespace and adjacent entities without guessing
    boundaries from the truncated BIO labels of an individual window.
    """
    result = {}
    for document_id, text in texts.items():
        targets = np.zeros((len(text), 2), dtype=np.uint8)
        for row in gold[document_id]:
            start, end, _ = row[:3]
            targets[start, 0] = 1
            targets[end - 1, 1] = 1
        result[document_id] = targets
    return result


@dataclass(frozen=True)
class BoundaryWindow:
    """A read-only wrapper; every original window attribute remains available."""

    window: object
    boundary_targets: object
    role_entities: object = None

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "window"), name)


def assign_complete_role_entities(windows, texts, gold, labels):
    """Assign every full original-Unicode span once, without ownership clipping.

    The character BIO and boundary tasks still supervise their uniquely owned
    characters. The separate entity task needs complete spans, so it chooses a
    containing window with the largest minimum left/right character context.
    Ties prefer the earlier original window index. Neither labels nor lengths
    determine an entity's weight in this auxiliary objective.
    """
    by_document = {}
    assigned = [[] for _ in windows]
    lookup = {label: index for index, label in enumerate(labels)}
    for index, window in enumerate(windows):
        by_document.setdefault(window.document_id, []).append((index, window))
    for document_id, rows in gold.items():
        candidates = by_document.get(document_id, [])
        for row in rows:
            start, end, label = row[:3]
            containing = [
                (min(start - w.char_start, w.char_end - end), -index, index, w)
                for index, w in candidates
                if w.char_start <= start < end <= w.char_end
            ]
            if not containing:
                raise ValueError(
                    f"No complete role window for {document_id}: [{start}, {end})"
                )
            _, _, index, window = max(containing, key=lambda item: item[:2])
            assigned[index].append(
                (start - window.char_start, end - window.char_start, lookup[label])
            )
    return [np.asarray(rows, dtype=np.int64).reshape(-1, 3) for rows in assigned]


class BoundaryDataset(Dataset):
    def __init__(self, base, *, role_auxiliary=False):
        self._base = base
        document_targets = (
            document_boundary_targets(base.texts, base.gold, base.labels)
            if base.gold is not None
            else None
        )
        role_entities = (
            assign_complete_role_entities(
                base.windows, base.texts, base.gold, base.labels
            )
            if role_auxiliary and base.gold is not None
            else None
        )
        self.windows = []
        for index, window in enumerate(base.windows):
            targets = (
                document_targets[window.document_id][
                    window.char_start : window.char_end
                ]
                if document_targets is not None
                else None
            )
            if targets is not None and targets.shape != (len(window.char_ids), 2):
                raise ValueError("Boundary crop and original character window disagree")
            roles = role_entities[index] if role_entities is not None else None
            self.windows.append(BoundaryWindow(window, targets, roles))

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_base"), name)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index]


def load_cache(path, partition, *, role_auxiliary=False):
    """Load the existing verified cache, adding targets only to labelled splits."""
    return BoundaryDataset(
        load_base_cache(path, partition), role_auxiliary=role_auxiliary
    )


def collate_windows(windows):
    batch = base_collate(windows)
    batch["boundary_targets"] = torch.full(
        (*batch["labels"].shape, 2), -100.0, dtype=torch.float32
    )
    role_entities = []
    for row, window in enumerate(windows):
        targets = getattr(window, "boundary_targets", None)
        if targets is not None:
            if targets.shape != (len(window.char_ids), 2):
                raise ValueError(
                    "Boundary targets must follow original window characters"
                )
            batch["boundary_targets"][row, : len(window.char_ids)] = torch.from_numpy(
                np.asarray(targets, dtype=np.float32)
            )
        roles = getattr(window, "role_entities", None)
        if roles is not None:
            if roles.ndim != 2 or roles.shape[1] != 3:
                raise ValueError("Role entities must be [entities, start/end/type]")
            if roles.size and (
                bool((roles[:, 0] < 0).any())
                or bool((roles[:, 1] <= roles[:, 0]).any())
                or bool((roles[:, 1] > len(window.char_ids)).any())
            ):
                raise ValueError("Role entities must be complete original window spans")
            role_entities.extend(
                (row, int(start), int(end), int(label)) for start, end, label in roles
            )
    batch["role_entities"] = torch.tensor(role_entities, dtype=torch.long).reshape(
        -1, 4
    )
    batch["role_entity_mask"] = torch.ones(len(role_entities), dtype=torch.bool)
    return batch
