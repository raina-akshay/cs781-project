"""Exact-character NER data, byte-BPE alignment and complete-document windows.

The original text is never normalized.  Training character identities are fitted
on the fixed 56-document training partition only.  Labels and offsets are the
official ones.  All windows are independent foundation-model sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .ner_contract import load_json, load_labels, validate_spans, validate_texts


ROOT = Path(__file__).resolve().parents[1]
CACHE_VERSION = 2
PAD_ID, UNK_ID, BOS_ID = 0, 1, 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _labels_check(labels: Sequence[str]) -> None:
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Labels must be a nonempty ordered list of distinct names")


def spans_to_bio(
    text: str | int, spans: Sequence[Sequence], labels: Sequence[str]
) -> np.ndarray:
    """Encode flat half-open exact character spans; reject overlapping spans."""
    _labels_check(labels)
    length = len(text) if isinstance(text, str) else text
    if type(length) is not int or length < 0:
        raise ValueError("Text length must be a nonnegative integer")
    result = np.zeros(length, dtype=np.int16)
    lookup = {label: i for i, label in enumerate(labels)}
    previous_end = 0
    for row in sorted(spans, key=lambda x: (x[0], x[1], x[2])):
        start, end, label = row[:3]
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= length
        ):
            raise ValueError("Span is outside the original text")
        if label not in lookup:
            raise ValueError(f"Unknown entity label: {label}")
        if start < previous_end:
            raise ValueError(
                "Character BIO cannot encode overlapping or duplicate entities"
            )
        result[start] = 1 + 2 * lookup[label]
        result[start + 1 : end] = 2 + 2 * lookup[label]
        previous_end = end
    return result


def bio_to_spans(ids: Sequence[int], labels: Sequence[str]) -> list[list]:
    """Decode a legal BIO path into exact original-character intervals.

    Invalid I transitions are rejected, rather than silently changing the path.
    Use decode_bio for constrained predictions from unconstrained model scores.
    """
    _labels_check(labels)
    path = np.asarray(ids)
    if path.ndim != 1 or not np.issubdtype(path.dtype, np.integer):
        raise ValueError("BIO labels must be a one-dimensional integer sequence")
    if path.size and (path.min() < 0 or path.max() >= 1 + 2 * len(labels)):
        raise ValueError("BIO state out of range")
    spans, start, active = [], None, None
    for position, raw in enumerate(path):
        state = int(raw)
        if state == 0 or state % 2 == 1:
            if start is not None:
                spans.append([start, position, labels[active]])
                start, active = None, None
            if state:
                start, active = position, (state - 1) // 2
        else:
            kind = (state - 2) // 2
            if start is None or active != kind:
                raise ValueError(f"Illegal BIO I transition at character {position}")
    if start is not None:
        spans.append([start, len(path), labels[active]])
    return spans


def decode_bio(logits, labels: Sequence[str]) -> np.ndarray:
    """Global maximum-emission BIO Viterbi with fixed legal transitions.

    All allowed transitions have zero cost.  I-X can only follow B-X or I-X,
    including across window-ownership boundaries.  Ties choose the lowest state
    index deterministically; no transition weights are estimated on any split.
    Raw logits and log-softmax scores produce the same maximizing sequence.
    """
    _labels_check(labels)
    if torch.is_tensor(logits):
        logits = logits.detach().float().cpu().numpy()
    emissions = np.asarray(logits, dtype=np.float32)
    states = 1 + 2 * len(labels)
    if emissions.ndim != 2 or emissions.shape[1] != states:
        raise ValueError(f"Expected [characters,{states}] logits")
    if not np.isfinite(emissions).all():
        raise ValueError("NER logits must be finite")
    length = len(emissions)
    if length == 0:
        return np.empty(0, dtype=np.int16)
    back = np.empty((length, states), dtype=np.uint8 if states < 256 else np.uint16)
    back[0] = 0
    score = emissions[0].astype(np.float64)
    inside = np.arange(2, states, 2)
    beginning = inside - 1
    score[inside] = -np.inf
    for position in range(1, length):
        best = int(np.argmax(score))
        back[position] = best
        new_score = emissions[position].astype(np.float64) + score[best]
        predecessors = np.where(score[beginning] >= score[inside], beginning, inside)
        back[position, inside] = predecessors
        new_score[inside] = emissions[position, inside] + score[predecessors]
        # Subtract a common constant to bound accumulation on long documents.
        new_score -= new_score.max()
        score = new_score
    path = np.empty(length, dtype=np.int16)
    path[-1] = int(np.argmax(score))
    for position in range(length - 1, 0, -1):
        path[position - 1] = back[position, path[position]]
    return path


@dataclass
class Window:
    document_id: str
    token_start: int
    token_end: int
    token_ids: np.ndarray
    char_start: int
    char_end: int
    char_ids: np.ndarray
    edge_token_index: np.ndarray
    edge_char_index: np.ndarray
    labels: np.ndarray
    owned_mask: np.ndarray


def make_windows(
    document_id: str,
    text: str,
    tokenizer,
    char_vocab: dict[str, int],
    labels: Sequence[str],
    gold: Sequence[Sequence] | None = None,
    window_tokens: int = 1024,
    overlap_tokens: int = 256,
) -> list[Window]:
    """Tokenize one untouched document and assign every character to one window.

    Byte-level BPE may have multiple tokens covering one Unicode character.  A
    boundary is safe only between non-overlapping offset groups; window starts
    and ends always fall on safe boundaries.  For ownership, the character's
    minimum left/right context in characters is maximized, with earlier windows
    winning a tie.  This keeps complete coverage while counting each supervised
    character exactly once.
    """
    if type(window_tokens) is not int or window_tokens < 2:
        raise ValueError("A window must allow BOS and at least one content token")
    capacity = window_tokens - 1
    if type(overlap_tokens) is not int or not 0 <= overlap_tokens < capacity:
        raise ValueError("Overlap must be in [0,window_tokens-1)")
    if not text:
        if gold:
            raise ValueError("An empty document cannot contain an entity")
        return []
    encoding = tokenizer.encode(text, add_special_tokens=False)
    ids = np.asarray(encoding.ids, dtype=np.int32)
    offsets = np.asarray(encoding.offsets, dtype=np.int32)
    if not len(ids) or offsets.shape != (len(ids), 2):
        raise ValueError("Nonempty document has invalid tokenization")
    if tokenizer.decode(ids.tolist(), skip_special_tokens=False) != text:
        raise ValueError("The frozen tokenizer changed original text")
    if (
        (offsets[:, 0] < 0).any()
        or (offsets[:, 1] > len(text)).any()
        or (offsets[:, 0] >= offsets[:, 1]).any()
    ):
        raise ValueError(
            "All original-text tokens need nonempty valid character offsets"
        )
    if (np.diff(offsets[:, 0]) < 0).any():
        raise ValueError("Token offsets must be ordered")
    if offsets[0, 0] != 0 or offsets[-1, 1] != len(text):
        raise ValueError("Tokenizer offsets do not reach both document boundaries")
    prefix_end = np.maximum.accumulate(offsets[:, 1])
    if (offsets[1:, 0] > prefix_end[:-1]).any():
        raise ValueError("Tokenizer offset coverage contains a character gap")
    safe = np.ones(len(ids) + 1, dtype=bool)
    safe[1:-1] = offsets[1:, 0] >= prefix_end[:-1]
    safe_positions = np.flatnonzero(safe)
    # Unknown test annotations are not an all-O target.  Explicit empty gold
    # lists remain supervised all-O documents; None means genuinely unlabeled.
    char_labels = (
        spans_to_bio(text, gold, labels)
        if gold is not None
        else np.full(len(text), -100, dtype=np.int16)
    )
    character_ids = np.fromiter(
        (char_vocab.get(c, UNK_ID) for c in text), dtype=np.int32, count=len(text)
    )
    windows = []
    start = 0
    while start < len(ids):
        position = (
            np.searchsorted(
                safe_positions, min(start + capacity, len(ids)), side="right"
            )
            - 1
        )
        end = int(safe_positions[position])
        if end <= start:
            raise ValueError(
                "One Unicode character needs more tokens than the window capacity"
            )
        char_start, char_end = int(offsets[start, 0]), int(prefix_end[end - 1])
        edge_tokens, edge_chars = [], []
        for local, (left, right) in enumerate(offsets[start:end], start=1):
            edge_tokens.extend([local] * int(right - left))
            edge_chars.extend(range(int(left - char_start), int(right - char_start)))
        local_counts = np.bincount(edge_chars, minlength=char_end - char_start)
        if len(local_counts) != char_end - char_start or (local_counts == 0).any():
            raise ValueError("Window does not cover each contained character")
        windows.append(
            Window(
                document_id=document_id,
                token_start=start,
                token_end=end,
                token_ids=np.concatenate(
                    (np.array([BOS_ID], dtype=np.int32), ids[start:end])
                ),
                char_start=char_start,
                char_end=char_end,
                char_ids=character_ids[char_start:char_end].copy(),
                edge_token_index=np.asarray(edge_tokens, dtype=np.int32),
                edge_char_index=np.asarray(edge_chars, dtype=np.int32),
                labels=char_labels[char_start:char_end].copy(),
                owned_mask=np.zeros(char_end - char_start, dtype=bool),
            )
        )
        if end == len(ids):
            break
        # Round the next start down to a safe boundary, increasing overlap by at
        # most the token group.  If that would stall, advance one complete group.
        desired = end - overlap_tokens
        next_position = np.searchsorted(safe_positions, desired, side="right") - 1
        next_start = int(safe_positions[next_position])
        if next_start <= start:
            next_position = np.searchsorted(safe_positions, start, side="right")
            next_start = int(safe_positions[next_position])
        if not start < next_start <= end:
            raise AssertionError("Window construction did not progress safely")
        start = next_start
    owner = np.full(len(text), -1, dtype=np.int32)
    best_context = np.full(len(text), -1, dtype=np.int32)
    for index, window in enumerate(windows):
        size = window.char_end - window.char_start
        context = np.minimum(np.arange(size), np.arange(size - 1, -1, -1))
        positions = slice(window.char_start, window.char_end)
        improve = context > best_context[positions]
        best_context[positions][improve] = context[improve]
        owner[positions][improve] = index
    if (owner < 0).any():
        raise AssertionError("Window ownership omitted document characters")
    for index, window in enumerate(windows):
        window.owned_mask[:] = owner[window.char_start : window.char_end] == index
    return windows


class NERDataset(Dataset):
    def __init__(self, payload: dict, partition: str):
        if partition not in payload["partitions"]:
            raise ValueError(f"Unknown partition {partition!r}")
        self.partition = partition
        document_ids = payload["partitions"][partition]
        wanted = set(document_ids)
        self.windows = [w for w in payload["windows"] if w.document_id in wanted]
        self.texts = {did: payload["texts"][did] for did in document_ids}
        self.gold = (
            {did: payload["gold"][did] for did in document_ids}
            if partition != "test"
            else None
        )
        self.labels = payload["labels"]
        self.char_vocab = payload["char_vocab"]
        self.manifest = payload["manifest"]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        return self.windows[index]


def collate_windows(windows: Sequence[Window]) -> dict:
    if not windows:
        raise ValueError("Cannot collate an empty batch")
    batch, tokens, chars = (
        len(windows),
        max(len(w.token_ids) for w in windows),
        max(len(w.char_ids) for w in windows),
    )
    input_ids = torch.zeros((batch, tokens), dtype=torch.long)
    attention_mask = torch.zeros((batch, tokens), dtype=torch.bool)
    char_ids = torch.zeros((batch, chars), dtype=torch.long)
    labels = torch.full((batch, chars), -100, dtype=torch.long)
    loss_mask = torch.zeros((batch, chars), dtype=torch.bool)
    counts = torch.ones((batch, chars), dtype=torch.float32)
    edge_token, edge_char = [], []
    for row, window in enumerate(windows):
        nt, nc = len(window.token_ids), len(window.char_ids)
        if not nc:
            raise ValueError("Empty documents must be represented without windows")
        input_ids[row, :nt] = torch.from_numpy(window.token_ids.astype(np.int64))
        attention_mask[row, :nt] = True
        char_ids[row, :nc] = torch.from_numpy(window.char_ids.astype(np.int64))
        labels[row, :nc] = torch.from_numpy(window.labels.astype(np.int64))
        loss_mask[row, :nc] = torch.from_numpy(window.owned_mask)
        edge_token.append(
            torch.from_numpy(window.edge_token_index.astype(np.int64)) + row * tokens
        )
        edge_char.append(
            torch.from_numpy(window.edge_char_index.astype(np.int64)) + row * chars
        )
        counts[row, :nc] = torch.from_numpy(
            np.bincount(window.edge_char_index, minlength=nc).astype(np.float32)
        )
    if not torch.all(counts > 0):
        raise AssertionError("Every real character requires at least one token feature")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_lengths": torch.tensor(
            [len(w.token_ids) for w in windows], dtype=torch.long
        ),
        "char_ids": char_ids,
        "char_lengths": torch.tensor(
            [len(w.char_ids) for w in windows], dtype=torch.long
        ),
        "edge_token_index": torch.cat(edge_token),
        "edge_char_index": torch.cat(edge_char),
        "edge_counts": counts,
        "labels": labels,
        "loss_mask": loss_mask,
        "windows": list(windows),
    }


def prepare_cache(
    output: str | Path,
    window_tokens: int = 1024,
    overlap_tokens: int = 256,
    *,
    source_paths: dict[str, Path],
) -> Path:
    """Prepare character windows and store their data/configuration fingerprint."""
    from tokenizers import Tokenizer

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "cache.pkl"
    paths = source_paths
    fingerprint = {
        "version": CACHE_VERSION,
        "window_tokens": window_tokens,
        "overlap_tokens": overlap_tokens,
        "source_sha256": {name: sha256(path) for name, path in paths.items()},
    }
    if cache_path.exists():
        manifest = load_json(output / "manifest.json")
        if (
            manifest["fingerprint"] != fingerprint
            or sha256(cache_path) != manifest["cache_sha256"]
        ):
            raise ValueError(
                "Existing cache differs from requested data/configuration; use a new cache directory"
            )
        return cache_path
    split = load_json(paths["split"])
    labels = load_labels(paths["labels"])
    train_text, train_gold = load_json(paths["train_text"]), load_json(
        paths["train_gold"]
    )
    validation_text, validation_gold = load_json(paths["validation_text"]), load_json(
        paths["validation_gold"]
    )
    test_text = load_json(paths["test_text"])
    for texts, gold in ((train_text, train_gold), (validation_text, validation_gold)):
        validate_texts(texts)
        validate_spans(texts, gold, labels)
    validate_texts(test_text)
    if (
        set(train_text) != set(split["train_ids"])
        or set(validation_text) != set(split["validation_ids"])
        or set(test_text) != set(split["test_ids"])
    ):
        raise ValueError("Prepared partition IDs differ from the frozen split")
    if set(train_text) & set(validation_text) or (
        set(train_text) | set(validation_text)
    ) & set(test_text):
        raise ValueError("Partitions contain document-ID overlap")
    vocab = {
        character: index + 2
        for index, character in enumerate(sorted(set("".join(train_text.values()))))
    }
    tokenizer = Tokenizer.from_file(str(paths["tokenizer"]))
    tokenizer.encode_special_tokens = True
    tokenizer_config = json.loads(tokenizer.to_str())
    if (
        tokenizer_config.get("normalizer") is not None
        or tokenizer_config.get("post_processor") is not None
    ):
        raise ValueError(
            "Frozen tokenizer must preserve original text without a postprocessor"
        )
    if (
        tokenizer.token_to_id("<|bos|>") != BOS_ID
        or tokenizer.token_to_id("<|pad|>") != PAD_ID
    ):
        raise ValueError("Frozen tokenizer special IDs changed")
    texts = {**train_text, **validation_text, **test_text}
    gold = {**train_gold, **validation_gold}
    windows = []
    summaries = {}
    partitions = {
        "train": sorted(train_text),
        "validation": sorted(validation_text),
        "all_train": sorted(gold),
        "test": sorted(test_text),
    }
    for did in sorted(texts):
        generated = make_windows(
            did,
            texts[did],
            tokenizer,
            vocab,
            labels,
            gold.get(did),
            window_tokens,
            overlap_tokens,
        )
        windows.extend(generated)
    for name, ids in partitions.items():
        selected = [w for w in windows if w.document_id in set(ids)]
        summaries[name] = {
            "documents": len(ids),
            "windows": len(selected),
            "characters": sum(len(texts[did]) for did in ids),
            "owned_characters": sum(int(w.owned_mask.sum()) for w in selected),
            "window_tokens_including_overlap": sum(len(w.token_ids) for w in selected),
            "max_window_characters": max(
                (len(w.char_ids) for w in selected), default=0
            ),
        }
        if summaries[name]["owned_characters"] != summaries[name]["characters"]:
            raise AssertionError(
                "Partition ownership does not cover exactly every character"
            )
    manifest = {
        "fingerprint": fingerprint,
        "labels": labels,
        "bio_order": ["O"]
        + [prefix + label for label in labels for prefix in ("B-", "I-")],
        "character_vocabulary_size_including_pad_unk": len(vocab) + 2,
        "character_vocabulary_fitted_on": "holdout-v1 train only (56 documents)",
        "character_pad_id": PAD_ID,
        "character_unknown_id": UNK_ID,
        "unicode_offsets": "Python Unicode codepoints; zero-based, end-exclusive",
        "ownership": "max minimum left/right character context; earlier window tie-break",
        "partitions": summaries,
    }
    payload = {
        "manifest": manifest,
        "texts": texts,
        "gold": gold,
        "labels": labels,
        "char_vocab": vocab,
        "windows": windows,
        "partitions": partitions,
    }
    temporary = output / f".cache-{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as stream:
            pickle.dump(payload, stream, protocol=5)
        os.replace(temporary, cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest["cache_sha256"] = sha256(cache_path)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "character-vocabulary.json").write_text(
        json.dumps(
            {"pad_id": PAD_ID, "unk_id": UNK_ID, "characters": vocab},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return cache_path


def load_cache(cache_path: str | Path, partition: str) -> NERDataset:
    """Read only our local checksum-verified cache, never an external pickle."""
    path = Path(cache_path)
    if path.is_dir():
        path = path / "cache.pkl"
    manifest = load_json(path.parent / "manifest.json")
    if (
        manifest["fingerprint"]["version"] != CACHE_VERSION
        or sha256(path) != manifest["cache_sha256"]
    ):
        raise ValueError("Cache version or checksum mismatch")
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if payload["manifest"]["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Cache and manifest fingerprints differ")
    return NERDataset(payload, partition)
