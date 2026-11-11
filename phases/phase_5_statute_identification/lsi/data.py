"""Verified, lossless case caches and exact-coverage distributed batching.

The cache contains ordinary content tokens only. Chunking adds BOS/EOS to
every consecutive piece; it never selects, overlaps, truncates or discards
content. TEST targets do not exist in the operational data interface.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SPLITS = ("train", "dev", "test")


def dev_partition(text_sha256):
    digest = hashlib.sha256(("cs781-lsi-dev-v1|42|" + text_sha256).encode()).digest()
    return "calibration" if digest[0] % 2 == 0 else "selection"


TOKENIZER_SHA256 = "25cb054b95fb14d53f12f265c4e09c897ba6dfc5d84b8298ab93004184648b1d"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_counts(lengths, chunk_length):
    """Count nonoverlapping content chunks, including one for empty text."""
    lengths = np.asarray(lengths, dtype=np.int64)
    capacity = chunk_length - 2
    return np.maximum(1, lengths // capacity + (lengths % capacity != 0))


class SplitCache:
    """One immutable-by-convention split backed by read-only NumPy mmaps."""

    def __init__(self, split, tokens, offsets, metadata):
        if (
            split not in SPLITS
            or metadata.get("format") != "cs781-p5-case-metadata-v1"
            or metadata.get("split") != split
        ):
            raise ValueError("Unexpected case metadata format or split")
        if tokens.dtype != np.dtype("<u4") or tokens.ndim != 1:
            raise ValueError("Content tokens must be a one-dimensional little-endian uint32 array")
        if offsets.dtype != np.dtype("<u8") or offsets.ndim != 1 or len(offsets) < 2:
            raise ValueError("Offsets must be a one-dimensional little-endian uint64 array")
        records = metadata.get("records")
        if not isinstance(records, list) or not records or len(offsets) != len(records) + 1:
            raise ValueError("Metadata and offset case counts differ")
        if (
            int(offsets[0]) != 0
            or int(offsets[-1]) != len(tokens)
            or bool((offsets[1:] < offsets[:-1]).any())
        ):
            raise ValueError("Offsets must monotonically cover the complete flat token array")
        # Check in bounded slices: even an unusually large corpus needs no
        # corpus-sized temporary boolean allocation.
        for start in range(0, len(tokens), 2**20):
            block = tokens[start : start + 2**20]
            if bool(((block <= 2) | (block >= 32768)).any()):
                raise ValueError("Content cache contains a special or out-of-vocabulary token")
        lengths = np.diff(offsets).astype(np.int64)
        lengths.flags.writeable = False
        seen = set()
        for record in records:
            case_id = record["id"]
            if not isinstance(case_id, str) or not case_id or case_id in seen:
                raise ValueError("Case IDs must be nonempty, unique strings within each split")
            seen.add(case_id)
            digest = record["text_sha256"]
            if split == "test":
                if record["labels"] is not None:
                    raise ValueError("TEST operational metadata must never retain a target")
            elif (
                not isinstance(record["labels"], list)
                or not record["labels"]
                or any(type(x) is not int or not 0 <= x < 100 for x in record["labels"])
                or record["labels"] != sorted(set(record["labels"]))
            ):
                raise ValueError("TRAIN/DEV targets must be sorted unique statute IDs 0..99")
            expected_partition = dev_partition(digest) if split == "dev" else split
            if record["dev_partition"] != expected_partition:
                raise ValueError(
                    "DEV calibration/selection must be fixed by text hash, never labels"
                )
        self.split, self.records = split, records
        self._tokens, self.offsets, self.lengths = tokens, offsets, lengths
        self.ids = [record["id"] for record in records]
        self.labels = None if split == "test" else [record["labels"] for record in records]
        self.raw_identity = metadata.get("raw_identity")

    def __len__(self):
        return len(self.records)

    def tokens(self, index):
        view = self._tokens[int(self.offsets[index]) : int(self.offsets[index + 1])]
        view.flags.writeable = False
        return view


class CacheBundle:
    """Load the three full-document token caches and their metadata."""

    def __init__(self, manifest_path):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest["tokenizer_sha256"] != TOKENIZER_SHA256:
            raise ValueError("Prepared data uses a different tokenizer")
        self.splits = {}
        for split in SPLITS:
            artifacts = self.manifest["splits"][split]["artifacts"]
            metadata = json.loads(Path(artifacts["metadata"]["path"]).read_text())
            tokens = np.load(artifacts["tokens"]["path"], mmap_mode="r", allow_pickle=False)
            offsets = np.load(artifacts["offsets"]["path"], mmap_mode="r", allow_pickle=False)
            self.splits[split] = SplitCache(split, tokens, offsets, metadata)
        self.identity = {
            "tokenizer_sha256": TOKENIZER_SHA256,
            "splits": {name: len(split) for name, split in self.splits.items()},
        }


def collate_cases(split_cache, indices, chunk_length):
    indices = list(indices)
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(index >= len(split_cache) for index in indices)
    ):
        raise ValueError("A batch must contain valid distinct original cases")
    lengths = split_cache.lengths[indices]
    counts = chunk_counts(lengths, chunk_length)
    chunk_length = int(chunk_length)
    capacity = chunk_length - 2
    chunks = []
    for index in indices:
        content = split_cache.tokens(index)
        for start in range(0, len(content), capacity) if len(content) else (0,):
            part = content[start : start + capacity]
            # Copy the read-only cache slice before constructing a Tensor.
            row = torch.empty(len(part) + 2, dtype=torch.long)
            row[0], row[-1] = 1, 2
            if len(part):
                row[1:-1] = torch.from_numpy(np.array(part, dtype=np.int64, copy=True))
            chunks.append(row)
    width = min(chunk_length, ((max(row.numel() for row in chunks) + 7) // 8) * 8)
    input_ids = torch.zeros((len(chunks), width), dtype=torch.long)
    attention_mask = torch.zeros((len(chunks), width), dtype=torch.bool)
    for i, row in enumerate(chunks):
        input_ids[i, : len(row)] = row
        attention_mask[i, : len(row)] = True
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "case_offsets": torch.from_numpy(offsets),
        "ids": [split_cache.ids[index] for index in indices],
        "content_tokens": int(lengths.sum()),
        "chunks": len(chunks),
        "case_indices": indices,
    }
    if split_cache.split in ("train", "dev"):
        labels = [split_cache.records[index]["labels"] for index in indices]
        targets = torch.zeros((len(labels), 100), dtype=torch.float32)
        for i, value in enumerate(labels):
            targets[i, value] = 1.0
        batch["labels"] = targets
    return batch


def global_case_batches(
    lengths, chunk_length, global_batch_cases, world_size, seed, epoch, shuffle=True
):
    """One deterministic exact-coverage epoch, with no distributed padding.

    Randomized pools are sorted by chunk count to reduce within-batch skew.
    An undersized final batch is merged into its predecessor before shuffling
    batches, so every global batch can supply one real case per rank.
    """
    counts = chunk_counts(lengths, chunk_length)
    batch_size = global_batch_cases
    world = world_size
    if type(shuffle) is not bool or len(counts) < world or batch_size < world:
        raise ValueError("Need at least one real case per rank and a global batch >= world_size")
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    order = rng.permutation(len(counts)).tolist() if shuffle else list(range(len(counts)))
    if shuffle:
        pool = 50 * batch_size
        order = [
            index
            for start in range(0, len(order), pool)
            for index in sorted(order[start : start + pool], key=lambda i: -int(counts[i]))
        ]
    batches = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
    if len(batches) > 1 and len(batches[-1]) < world:
        batches[-2].extend(batches.pop())
    if shuffle:
        rng.shuffle(batches)
    return batches


def rank_case_indices(global_indices, lengths, chunk_length, world_size, rank):
    """Greedily balance complete cases by chunk count, keeping local order."""
    counts = chunk_counts(lengths, chunk_length)
    world = world_size
    indices = list(global_indices)
    if (
        rank >= world
        or len(indices) < world
        or len(indices) != len(set(indices))
        or any(i >= len(counts) for i in indices)
    ):
        raise ValueError("A global batch must supply distinct valid cases to every rank")
    assigned, totals = [[] for _ in range(world)], [0] * world
    position = {index: offset for offset, index in enumerate(indices)}
    for index in sorted(indices, key=lambda i: (-int(counts[i]), position[i])):
        owner = min(range(world), key=lambda r: (totals[r], len(assigned[r]), r))
        assigned[owner].append(index)
        totals[owner] += int(counts[index])
    return sorted(assigned[rank], key=position.__getitem__)
