"""Training-only indexes for exact document deduplication.

The original corpus/token streams are opened read-only. SQLite bounds the
canonical-document index by disk space; only one raw document is read at a time.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import (
    WINDOW_DTYPE,
    _connect_readonly,
    _new_directory,
    _progress,
    _write_json,
    file_sha256,
    read_prepared_manifest,
    verify_token_artifacts,
)
from .tokenization import load_tokenizer

SCHEMA_VERSION = 1
INDEX_DTYPE = np.dtype([("window_index", "<u8"), ("targets", "<u4"), ("source_id", "<u2")])


def _fingerprint(manifest: dict) -> str:
    value = {k: v for k, v in manifest.items() if k != "fingerprint"}
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def _mmap(path: Path, dtype) -> np.ndarray:
    size = path.stat().st_size
    if size % np.dtype(dtype).itemsize:
        raise ValueError(f"Invalid binary artifact size: {path.name}")
    return np.memmap(path, mode="r", dtype=dtype) if size else np.empty(0, dtype=dtype)


def prepare_stage3(token_dir: str | Path, output_dir: str | Path, *, seed: int = 42) -> dict:
    """Index ordinary training windows, retaining the first occurrence of each text."""
    token_dir = Path(token_dir).resolve()
    manifest_path = token_dir / "manifest.json"
    _progress("Stage 3 prepare: verifying original train/dev tokens and corpus provenance")
    parent = verify_token_artifacts(token_dir, splits=("train", "dev"))
    parent_sha = file_sha256(manifest_path)
    prepared_dir = Path(parent["prepared_dir"])
    prepared = read_prepared_manifest(prepared_dir)
    tokenizer = load_tokenizer(parent["tokenizer_path"])
    sources = parent["sources"]
    if not sources or len(sources) > 63 or len(set(sources)) != len(sources):
        raise ValueError("Expected one to 63 distinct corpus sources")
    if [item["name"] for item in prepared["sources"]] != sources:
        raise ValueError("Prepared/token source order differs")
    if parent["token_dtype"] != "<u2" or tokenizer.vocab_size != parent["vocab_size"]:
        raise ValueError("Unsupported token dtype or vocabulary mismatch")
    context = parent["context_length"]
    info = parent["splits"]["train"]
    token_path = token_dir / info["tokens_file"]
    window_path = token_dir / info["windows_file"]
    docs_path = token_dir / info["documents_file"]
    tokens = _mmap(token_path, "<u2")
    windows = _mmap(window_path, WINDOW_DTYPE)
    if len(tokens) != info["num_tokens"] or len(windows) != info["num_windows"]:
        raise ValueError("Parent binary size differs from token manifest")
    output = _new_directory(output_dir)
    _progress(
        "Stage 3 prepare: indexing canonical training occurrences without modifying original streams"
    )
    raw_path, dedup_path = output / "ordinary.raw.bin", output / "ordinary.dedup.bin"
    index_path = output / "canonical.sqlite3"
    database = _connect_readonly(prepared_dir / prepared["database"])
    index = sqlite3.connect(index_path)
    index.execute("PRAGMA temp_store=FILE")
    index.execute("PRAGMA cache_size=-32768")
    index.execute(
        """CREATE TABLE canonical(doc_hash TEXT PRIMARY KEY, document_id INTEGER NOT NULL,
        source_id INTEGER NOT NULL, first_window INTEGER NOT NULL, num_windows INTEGER NOT NULL,
        source_mask INTEGER NOT NULL, occurrences INTEGER NOT NULL, sample_order TEXT NOT NULL)"""
    )
    raw_totals, dedup_totals = [0] * len(sources), [0] * len(sources)
    n_documents = n_unique = n_raw = n_dedup = token_cursor = 0
    try:
        # Match each index row to the original training occurrence.
        expected = database.execute(
            """SELECT o.source_id,d.doc_hash FROM occurrences o
            CROSS JOIN documents d ON d.doc_hash=o.doc_hash WHERE d.split='train'
            ORDER BY o.source_id,o.row_number"""
        )
        with docs_path.open(encoding="utf-8") as doc_file, raw_path.open(
            "wb"
        ) as raw_out, dedup_path.open("wb") as dedup_out:
            for line in doc_file:
                doc = json.loads(line)
                truth = expected.fetchone()
                source = doc["source_id"]
                if (
                    truth is None
                    or tuple(truth) != (source, doc["doc_hash"])
                    or doc["document_id"] != n_documents
                    or doc["token_offset"] != token_cursor
                    or doc["first_window"] != n_raw
                ):
                    raise ValueError(
                        "Training document metadata disagrees with authoritative occurrence order"
                    )
                count, num_tokens = doc["num_windows"], doc["num_tokens"]
                entries = windows[n_raw : n_raw + count]
                starts = np.arange(count, dtype=np.uint64) * context
                expected_targets = np.minimum(context, num_tokens - 1 - starts)
                if (
                    len(entries) != count
                    or np.any(entries["document_id"] != n_documents)
                    or np.any(entries["source_id"] != source)
                    or np.any(entries["offset"] != token_cursor + starts)
                    or np.any(entries["targets"] != expected_targets)
                ):
                    raise ValueError("Training windows disagree with document metadata")
                if (
                    token_cursor + num_tokens > len(tokens)
                    or int(tokens[token_cursor]) != tokenizer.bos_id
                    or int(tokens[token_cursor + num_tokens - 1]) != tokenizer.eos_id
                ):
                    raise ValueError("Original document BOS/EOS or bounds invalid")
                records = np.empty(count, dtype=INDEX_DTYPE)
                records["window_index"] = np.arange(n_raw, n_raw + count)
                records["targets"] = entries["targets"]
                records["source_id"] = source
                raw_out.write(records.tobytes())
                raw_totals[source] += num_tokens - 1
                order = hashlib.sha256(
                    f"{seed}:stage3:document:{doc['doc_hash']}".encode()
                ).hexdigest()
                inserted = index.execute(
                    "INSERT OR IGNORE INTO canonical VALUES(?,?,?,?,?,?,?,?)",
                    (doc["doc_hash"], n_documents, source, n_raw, count, 1 << source, 1, order),
                ).rowcount
                if inserted:
                    dedup_out.write(records.tobytes())
                    dedup_totals[source] += num_tokens - 1
                    n_unique += 1
                    n_dedup += count
                else:
                    index.execute(
                        "UPDATE canonical SET source_mask=source_mask | ?,occurrences=occurrences+1 WHERE doc_hash=?",
                        (1 << source, doc["doc_hash"]),
                    )
                n_documents += 1
                n_raw += count
                token_cursor += num_tokens
                if n_documents % 10000 == 0:
                    index.commit()
                if n_documents % 100000 == 0:
                    _progress(
                        f"Stage 3 prepare: {n_documents:,} occurrences; {n_unique:,} unique texts"
                    )
            if (
                expected.fetchone() is not None
                or n_documents != info["num_documents"]
                or n_raw != len(windows)
                or token_cursor != len(tokens)
            ):
                raise ValueError("Training metadata is incomplete")
        if (
            sum(raw_totals) != info["target_tokens"]
            or dict(zip(sources, raw_totals)) != info["source_target_tokens"]
        ):
            raise ValueError("Original source target totals disagree")
        index.execute("CREATE INDEX canonical_queue ON canonical(source_id,sample_order,doc_hash)")
        index.commit()
        cross_source_groups = index.execute(
            "SELECT COUNT(*) FROM canonical WHERE (source_mask & (source_mask-1)) != 0"
        ).fetchone()[0]
    finally:
        database.close()
        index.close()
    files = {
        p.name: {"sha256": file_sha256(p), "size_bytes": p.stat().st_size}
        for p in sorted(output.iterdir())
        if p.is_file()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "stage3_training_view",
        "diagnostic": False,
        "token_dir": str(token_dir),
        "token_manifest_sha256": parent_sha,
        "token_manifest": parent,
        "sources": sources,
        "context_length": context,
        "pad_id": tokenizer.pad_id,
        "source_target_tokens": info["source_target_tokens"],
        "seed": seed,
        "prepared_database_sha256": prepared["database_sha256"],
        "train_documents_sha256": file_sha256(docs_path),
        "files": files,
        "canonical": {
            "file": "canonical.sqlite3",
            "documents": n_unique,
            "raw_documents": n_documents,
            "removed_occurrences": n_documents - n_unique,
            "cross_source_groups": cross_source_groups,
            "representative_policy": "lowest source_id, then original training occurrence order",
        },
        "ordinary": {
            "raw": {
                "file": raw_path.name,
                "num_examples": n_raw,
                "target_tokens": sum(raw_totals),
                "source_target_tokens": dict(zip(sources, raw_totals)),
            },
            "dedup": {
                "file": dedup_path.name,
                "num_examples": n_dedup,
                "target_tokens": sum(dedup_totals),
                "source_target_tokens": dict(zip(sources, dedup_totals)),
            },
        },
    }
    manifest["fingerprint"] = _fingerprint(manifest)
    _write_json(output / "manifest.json", manifest)
    return manifest


def verify_stage3_artifacts(view_dir: str | Path) -> dict:
    """Read a prepared training view and check the foundation tokenizer."""
    directory = Path(view_dir).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    verify_token_artifacts(manifest["token_dir"], splits=("train", "dev"))
    return manifest


class Stage3Dataset(Dataset):
    """Read-only padded tensors plus physical-target metadata for quota queues."""

    def __init__(
        self,
        view_dir: str | Path,
        component: str = "ordinary",
        *,
        deduplicate: bool = True,
        verify: bool = True,
    ):
        if component != "ordinary":
            raise ValueError("The selected model trains on ordinary windows")
        self.view_dir = Path(view_dir).resolve()
        self.manifest = (
            verify_stage3_artifacts(self.view_dir)
            if verify
            else json.loads((self.view_dir / "manifest.json").read_text(encoding="utf-8"))
        )
        self.component = component
        self.deduplicate = deduplicate
        self.token_dir = Path(self.manifest["token_dir"])
        self.token_manifest = self.manifest["token_manifest"]
        self.context_length = self.manifest["context_length"]
        self.sources = self.source_names = self.manifest["sources"]
        self.pad_id = self.manifest["pad_id"]
        info = self.manifest["ordinary"]["dedup" if deduplicate else "raw"]
        self.entries = _mmap(self.view_dir / info["file"], INDEX_DTYPE)
        parent = self.token_manifest["splits"]["train"]
        self.tokens = _mmap(self.token_dir / parent["tokens_file"], "<u2")
        self.windows = _mmap(self.token_dir / parent["windows_file"], WINDOW_DTYPE)
        self.raw_window_indices = self.entries["window_index"]
        if len(self.entries) != info["num_examples"]:
            raise ValueError("View index length differs from manifest")
        self.valid_counts = self.target_counts = self.entries["targets"]
        self.source_ids = self.entries["source_id"]
        if (
            np.any(self.valid_counts == 0)
            or np.any(self.valid_counts > self.context_length)
            or np.any(self.source_ids >= len(self.sources))
        ):
            raise ValueError("Invalid physical target counts or source IDs")
        self.indices_by_source = [
            np.flatnonzero(self.source_ids == i) for i in range(len(self.sources))
        ]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        record = self.entries[index]
        count, source = int(record["targets"]), int(record["source_id"])
        window = self.windows[int(record["window_index"])]
        offset = int(window["offset"])
        sequence = np.array(self.tokens[offset : offset + count + 1], dtype=np.int64)
        if len(sequence) != count + 1:
            raise ValueError("View token bounds invalid")
        x = torch.full((self.context_length,), self.pad_id, dtype=torch.long)
        y = torch.full_like(x, -100)
        weights = torch.zeros(self.context_length, dtype=torch.float32)
        continuation_mask = torch.zeros(self.context_length, dtype=torch.bool)
        x[:count] = torch.from_numpy(sequence[:-1])
        y[:count] = torch.from_numpy(sequence[1:])
        weights[:count] = 1.0
        return {
            "input_ids": x,
            "labels": y,
            "weights": weights,
            "continuation_mask": continuation_mask,
            "source_id": torch.tensor(source, dtype=torch.long),
        }
