"""Core/halo windows with one supervised owner for every original sentence."""

import torch
from torch.utils.data import Dataset


def build_windows(cache, split, core_sentences=32, halo_sentences=16):
    windows = []
    for document in cache["documents"]:
        if document["split"] != split:
            continue
        doc_id, domain, sentences = document["id"], document["domain"], document["sentences"]
        for core_start in range(0, len(sentences), core_sentences):
            core_end = min(len(sentences), core_start + core_sentences)
            start = max(0, core_start - halo_sentences)
            end = min(len(sentences), core_end + halo_sentences)
            indices = list(range(start, end))
            mask = [core_start <= index < core_end for index in indices]
            labels = [
                int(sentences[index]["label"]) if mask[slot] and split != "test" else -100
                for slot, index in enumerate(indices)
            ]
            windows.append(
                {
                    "doc_id": doc_id,
                    "domain": domain,
                    "split": split,
                    "window_index": len(windows),
                    "sentence_indices": indices,
                    "tokens": [list(sentences[index]["tokens"]) for index in indices],
                    "labels": labels,
                    "target_mask": mask,
                }
            )
    return windows


class WindowDataset(Dataset):
    def __init__(self, windows, max_length=256):
        self.windows = list(windows)
        self.max_length = int(max_length)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        capacity = self.max_length - 2
        inputs = []
        for tokens in window["tokens"]:
            if len(tokens) > capacity:
                head, tail = (capacity + 1) // 2, capacity // 2
                content = tokens[:head] + tokens[-tail:] if tail else tokens[:head]
            else:
                content = list(tokens)
            inputs.append([1, *content, 2])
        return {
            key: (list(value) if isinstance(value, list) else value)
            for key, value in window.items()
            if key != "tokens"
        } | {"input_ids": inputs}


def collate_windows(rows):
    window_lengths = [len(row["input_ids"]) for row in rows]
    batch, width = len(rows), max(window_lengths)
    labels = torch.full((batch, width), -100, dtype=torch.long)
    target_mask = torch.zeros((batch, width), dtype=torch.bool)
    indices = torch.full((batch, width), -1, dtype=torch.long)
    sentence_map, active = [], []
    for window_slot, row in enumerate(rows):
        length = window_lengths[window_slot]
        original = row["sentence_indices"]
        for sentence_slot, ids in enumerate(row["input_ids"]):
            active.append(ids)
            sentence_map.append((window_slot, sentence_slot))
        labels[window_slot, :length] = torch.tensor(row["labels"], dtype=torch.long)
        target_mask[window_slot, :length] = torch.tensor(row["target_mask"], dtype=torch.bool)
        indices[window_slot, :length] = torch.tensor(original, dtype=torch.long)
    token_lengths = torch.tensor([len(ids) for ids in active], dtype=torch.long)
    token_width = ((int(token_lengths.max()) + 7) // 8) * 8
    input_ids = torch.zeros((len(active), token_width), dtype=torch.long)
    for index, ids in enumerate(active):
        input_ids[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "lengths": token_lengths,
        "sentence_map": torch.tensor(sentence_map, dtype=torch.long),
        "sentence_lengths": torch.tensor(window_lengths, dtype=torch.long),
        "labels": labels,
        "target_mask": target_mask,
        "sentence_indices": indices,
        "doc_ids": [row["doc_id"] for row in rows],
        "domains": [row["domain"] for row in rows],
        "window_indices": [row["window_index"] for row in rows],
    }
