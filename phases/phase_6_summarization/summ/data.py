"""Complete-target windows and original-case normalization; TEST has no targets."""
from dataclasses import dataclass
import importlib
from pathlib import Path

import numpy as np
import torch

from .common import read_json
from .foundation import load_source_types, read_manifest


def own_tokenizer(manifest_path):
    manifest = read_manifest(manifest_path)
    cls, _ = load_source_types(manifest)
    package = cls.__module__.rsplit(".", 1)[0]
    module = importlib.import_module(package + ".tokenization")
    tokenizer = module.load_tokenizer(manifest["tokenizer_path"])
    if tokenizer.special_ids != (0, 1, 2) or tokenizer.vocab_size != 32768:
        raise ValueError("Original vocabulary contract changed")
    return tokenizer


class Split:
    def __init__(self, root, name):
        self.name = name
        self.meta = read_json(root / (name + ".metadata.json"))
        self.source = np.load(root / (name + ".source.npy"), mmap_mode="r")
        self.source_offsets = np.load(root / (name + ".source-offsets.npy"), mmap_mode="r")
        self.target = (
            None if name == "test" else np.load(root / (name + ".target.npy"), mmap_mode="r")
        )
        self.target_offsets = (
            None
            if name == "test"
            else np.load(root / (name + ".target-offsets.npy"), mmap_mode="r")
        )
        self.ids = [row["id"] for row in self.meta]
        if len(set(self.ids)) != len(self.ids) or len(self.source_offsets) != len(self.ids) + 1:
            raise ValueError("Cache IDs/offsets differ")
        if (
            self.source_offsets[0] != 0
            or self.source_offsets[-1] != len(self.source)
            or np.any(np.diff(self.source_offsets) < 1)
        ):
            raise ValueError("Invalid source offsets")
        if self.target is not None:
            if (
                len(self.target_offsets) != len(self.ids) + 1
                or self.target_offsets[0] != 0
                or self.target_offsets[-1] != len(self.target)
                or np.any(np.diff(self.target_offsets) < 1)
            ):
                raise ValueError("Invalid target offsets")

    def __len__(self):
        return len(self.ids)

    def source_ids(self, i):
        return (
            self.source[self.source_offsets[i] : self.source_offsets[i + 1]]
            .astype(np.int64)
            .tolist()
        )

    def target_ids(self, i):
        if self.target is None:
            raise ValueError("TEST has no targets")
        return (
            self.target[self.target_offsets[i] : self.target_offsets[i + 1]]
            .astype(np.int64)
            .tolist()
        )

    def token_count(self, i):
        return int(self.target_offsets[i + 1] - self.target_offsets[i]) + 1


class Cache:
    def __init__(self, path):
        self.manifest_path = Path(path).resolve()
        self.manifest = read_json(self.manifest_path)
        root = self.manifest_path.parent
        if self.manifest["format"] != "cs781-phase6-s1-prepared-v1":
            raise ValueError("Wrong prepared data")
        self.splits = {name: Split(root, name) for name in ("train", "val", "test")}
        if [len(self.splits[n]) for n in ("train", "val", "test")] != [6328, 700, 100]:
            raise ValueError("Need all declared original TRAIN/VAL/TEST cases")


class Formatter:
    def __init__(self, tokenizer, sequence):
        self.tokenizer, self.s = tokenizer, sequence
        self.intro = tokenizer.encode(
            "Write a factual legal summary preserving the important facts, reasoning and outcome. Do not add unsupported facts.\nDocument:\n"
        )
        self.finish = tokenizer.encode("\nContinue summary:\n")

    def prefix(self, source, history, offset):
        mode = "START" if offset == 0 else "CONTINUE"
        middle = self.tokenizer.encode(
            "\n\nSummary task: "
            + mode
            + ". Tokens already written: "
            + str(offset)
            + ".\nRecent summary:\n"
        )
        result = (
            [self.tokenizer.bos_id]
            + self.intro
            + list(source)
            + middle
            + list(history)
            + self.finish
        )
        if len(result) + self.s["block_tokens"] > self.s["context_tokens"]:
            raise ValueError("Physical native context would overflow")
        return result


@dataclass(frozen=True)
class Segment:
    case: int
    start: int
    stop: int
    total: int


def segments_for(split, indices, block_tokens):
    result = []
    for i in indices:
        total = split.token_count(i)
        result.extend(
            Segment(int(i), start, min(start + block_tokens, total), total)
            for start in range(0, total, block_tokens)
        )
    return result


def global_batches(split, epoch, seed, batch_cases):
    order = np.random.default_rng(seed + epoch).permutation(len(split)).tolist()
    return [order[start : start + batch_cases] for start in range(0, len(order), batch_cases)]


def rank_segments(split, indices, block_tokens, world, rank):
    """Partition complete cases' segments, without changing case loss weights."""
    all_segments = segments_for(split, indices, block_tokens)
    jobs = sorted(
        all_segments, key=lambda x: (-(x.stop - x.start + min(x.start, 384)), x.case, x.start)
    )
    loads = [0] * world
    assigned = [[] for _ in range(world)]
    for seg in jobs:
        r = min(range(world), key=lambda k: (loads[k], k))
        assigned[r].append(seg)
        loads[r] += (
            seg.stop - seg.start + min(seg.start, 384) + len(split.source_ids(seg.case)) + 96
        )
    # Stable length ordering saves right-padding work; case/window membership is unchanged.
    return sorted(
        assigned[rank],
        key=lambda x: (
            -(x.stop - x.start + min(x.start, 384) + len(split.source_ids(x.case))),
            x.case,
            x.start,
        ),
    )


def collate(split, segments, formatter, device="cpu", dummy=False):
    inputs = []
    labels = []
    weights = []
    for seg in segments:
        reference = split.target_ids(seg.case) + [formatter.tokenizer.eos_id]
        history = reference[max(0, seg.start - formatter.s["history_tokens"]) : seg.start]
        prefix = formatter.prefix(split.source_ids(seg.case), history, seg.start)
        target = reference[seg.start : seg.stop]
        # Last prefix state predicts first current target; final target need not be fed.
        row = prefix + target[:-1]
        lab = [-100] * (len(prefix) - 1) + target
        inputs.append(row)
        labels.append(lab)
        weights.append(0.0 if dummy else 1.0 / seg.total)
    width = max(map(len, inputs))
    n = len(inputs)
    ids = torch.full((n, width), formatter.tokenizer.pad_id, dtype=torch.long)
    labs = torch.full((n, width), -100, dtype=torch.long)
    for i, (row, lab) in enumerate(zip(inputs, labels)):
        ids[i, : len(row)] = torch.tensor(row)
        labs[i, : len(lab)] = torch.tensor(lab)
    return {
        "input_ids": ids.to(device),
        "labels": labs.to(device),
        "case_weights": torch.tensor(weights, dtype=torch.float32, device=device),
        "lengths": torch.tensor([len(x) for x in inputs], dtype=torch.long, device=device),
    }
