"""Immutable, split-preserving PCR data and bounded full-input chunk sampling.

No model/tokenizer is fitted here. All official IDs and complete relevance lists
are retained. Only TRAIN is eligible for supervised fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np

SPLITS = {"train": "Train", "dev": "Dev", "test": "Test"}
CACHE_VERSION = 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-standard JSON constant: {value}")


def read_json(path: Path):
    return json.loads(
        Path(path).read_text(encoding="utf8"),
        object_pairs_hook=_unique_pairs,
        parse_constant=_reject_constant,
    )


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf8", dir=path.parent, prefix=path.name + ".", delete=False
    ) as f:
        f.write(payload)
        temporary = Path(f.name)
    os.replace(temporary, path)


@dataclass
class RawSplit:
    split: str
    query_ids: list[str]
    candidate_ids: list[str]
    query_texts: list[str]
    candidate_texts: list[str]
    labels: dict[str, list[str]] | None

    @property
    def supervised_query_indices(self) -> list[int]:
        if self.split != "train" or self.labels is None:
            raise ValueError("Supervised sampling is restricted to TRAIN")
        return [i for i, q in enumerate(self.query_ids) if self.labels[q]]


def load_split(raw_root: Path, split: str, verify_hashes: bool = True) -> RawSplit:
    raw_root = Path(raw_root)
    manifest = read_json(raw_root / "source-manifest.json") if verify_hashes else None
    expected = (
        {Path(x["repository_path"]).name: x["sha256"] for x in manifest["files"]}
        if manifest
        else {}
    )
    records = []
    for suffix in ("", "_candidates"):
        path = raw_root / SPLITS[split] / f"{split}{suffix}.json"
        if verify_hashes and sha256(path) != expected[path.name]:
            raise ValueError(f"Raw file hash mismatch: {path}")
        value = read_json(path)
        if not isinstance(value, dict) or not value:
            raise ValueError(f"Expected nonempty record mapping: {path}")
        for case_id, record in value.items():
            if (
                not isinstance(case_id, str)
                or len(case_id) != 10
                or not case_id.isdecimal()
            ):
                raise ValueError(f"Invalid unchanged case ID: {case_id!r}")
            if not isinstance(record, dict) or not isinstance(record.get("text"), list):
                raise ValueError(f"Invalid text record: {case_id}")
            if any(not isinstance(s, str) for s in record["text"]):
                raise ValueError(f"Non-string sentence: {case_id}")
            if not "\n".join(record["text"]).strip():
                raise ValueError(f"Empty document: {case_id}")
            if (split == "test" or suffix) and "relevant_candidates" in record:
                raise ValueError(f"Unexpected relevance label field in {path.name}")
        records.append(value)
    queries, candidates = records
    labels = None
    if split != "test":
        labels = {}
        for q, rec in queries.items():
            gold = rec.get("relevant_candidates")
            if not isinstance(gold, list) or any(not isinstance(c, str) for c in gold):
                raise ValueError(f"Invalid gold list: {q}")
            if len(set(gold)) != len(gold) or any(c not in candidates for c in gold):
                raise ValueError(f"Duplicate/out-of-pool gold ID: {q}")
            if q in gold:
                raise ValueError(
                    f"Self-positive requires revisiting self-exclusion: {q}"
                )
            labels[q] = list(gold)
    return RawSplit(
        split,
        list(queries),
        list(candidates),
        ["\n".join(v["text"]) for v in queries.values()],
        ["\n".join(v["text"]) for v in candidates.values()],
        labels,
    )


def window_indices(token_count: int, chunk_length: int, max_chunks: int) -> list[int]:
    """Select whole contiguous windows evenly across the document, including ends.

    With more than max_chunks windows this deliberately samples, not covers, the
    document. Integer arithmetic avoids platform-dependent float rounding.
    """
    if token_count <= 0 or chunk_length <= 0 or max_chunks <= 0:
        raise ValueError("Counts and chunk limits must be positive")
    n = (token_count + chunk_length - 1) // chunk_length
    if n <= max_chunks:
        return list(range(n))
    if max_chunks == 1:
        return [0]
    return [i * (n - 1) // (max_chunks - 1) for i in range(max_chunks)]


def _summary(values: Iterable[float]) -> dict:
    x = np.asarray(list(values), dtype=np.float64)
    return {
        "count": int(x.size),
        "min": float(x.min()),
        "median": float(np.median(x)),
        "mean": float(x.mean()),
        "max": float(x.max()),
    }


def _write_view(
    path: Path,
    ids: list[str],
    texts: list[str],
    tokenizer,
    chunk_length: int,
    max_chunks: int,
    bos_id: int,
) -> dict:
    path.mkdir(parents=True)
    doc_offsets, chunk_offsets, original_counts, starts = [0], [0], [], []
    selected_counts, chunks_per_doc, coverage = [], [], []
    # Streaming avoids retaining entire raw tokenizations in RAM.
    with (path / "tokens.bin").open("wb") as stream:
        for text in texts:
            tok = tokenizer.encode(text, add_special_tokens=False).ids
            if not tok:
                raise ValueError("Tokenizer produced an empty nonempty document")
            original_counts.append(len(tok))
            content_length = chunk_length - 1
            indices = window_indices(len(tok), content_length, max_chunks)
            used = 0
            for i in indices:
                start = i * content_length
                chunk = np.asarray(
                    [bos_id] + tok[start : start + content_length], dtype="<u4"
                )
                stream.write(chunk.tobytes())
                chunk_offsets.append(chunk_offsets[-1] + len(chunk))
                starts.append(start)
                used += len(chunk) - 1
            doc_offsets.append(doc_offsets[-1] + len(indices))
            chunks_per_doc.append(len(indices))
            selected_counts.append(used)
            coverage.append(used / len(tok))
    write_json(path / "ids.json", ids)
    for name, array in (
        ("doc_offsets", doc_offsets),
        ("chunk_offsets", chunk_offsets),
        ("token_counts", original_counts),
        ("chunk_starts", starts),
    ):
        np.save(
            path / f"{name}.npy", np.asarray(array, dtype=np.int64), allow_pickle=False
        )
    audit = {
        "documents": len(ids),
        "chunks": doc_offsets[-1],
        "original_tokens": sum(original_counts),
        "selected_tokens": sum(selected_counts),
        "aggregate_token_coverage": sum(selected_counts) / sum(original_counts),
        "document_token_coverage": _summary(coverage),
        "chunks_per_document": _summary(chunks_per_doc),
        "fully_covered_documents": sum(
            a == b for a, b in zip(original_counts, selected_counts)
        ),
        "special_tokens_added": {"bos_id": bos_id, "eos": False},
        "content_tokens_per_window": chunk_length - 1,
        "coverage_excludes_special_tokens": True,
        "token_dtype": "little-endian uint32",
    }
    write_json(path / "audit.json", audit)
    return audit


def prepare_cache(
    raw_root: Path,
    tokenizer_path: Path,
    output_root: Path,
    chunk_length: int = 512,
    max_chunks: int = 8,
    bos_id: int = 1,
) -> Path:
    """Build a content-addressed cache; completed caches are verified, not rewritten."""
    from tokenizers import Tokenizer

    raw_root, tokenizer_path, output_root = map(
        Path, (raw_root, tokenizer_path, output_root)
    )
    if chunk_length < 2 or max_chunks < 1 or bos_id < 0:
        raise ValueError("Invalid chunk recipe")
    source = read_json(raw_root / "source-manifest.json")
    recipe = {
        "cache_version": CACHE_VERSION,
        "tokenizer_sha256": sha256(tokenizer_path),
        "chunk_length": chunk_length,
        "max_chunks": max_chunks,
        "window_rule": "nonoverlap_ceil_integer_even_endpoints",
        "text_join": "newline",
        "special_tokens": {"bos_id": bos_id, "eos": False},
        "sources": {x["repository_path"]: x["sha256"] for x in source["files"]},
    }
    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    output = output_root / key[:20]
    if (output / "manifest.json").exists():
        verify_cache(output)
        return output
    output_root.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(
            f"Incomplete cache requires explicit inspection: {output}"
        )
    temporary = Path(tempfile.mkdtemp(prefix="building-", dir=output_root))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    audits = {}
    for split in SPLITS:
        data = load_split(raw_root, split)
        for role in ("query", "candidate"):
            print(f"tokenizing {split}/{role}", flush=True)
            audits[f"{split}/{role}"] = _write_view(
                temporary / split / role,
                getattr(data, role + "_ids"),
                getattr(data, role + "_texts"),
                tokenizer,
                chunk_length,
                max_chunks,
                bos_id,
            )
        write_json(temporary / split / "labels.json", data.labels)
    files = {
        str(p.relative_to(temporary)): {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(temporary.rglob("*"))
        if p.is_file()
    }
    write_json(
        temporary / "manifest.json",
        {"recipe": recipe, "cache_key": key, "files": files, "audits": audits},
    )
    os.rename(temporary, output)
    verify_cache(output)
    return output


def verify_cache(path: Path) -> dict:
    manifest = read_json(Path(path) / "manifest.json")
    recipe_key = hashlib.sha256(
        json.dumps(manifest["recipe"], sort_keys=True).encode()
    ).hexdigest()
    if recipe_key != manifest["cache_key"]:
        raise ValueError("Cache recipe identity mismatch")
    for relative, info in manifest["files"].items():
        p = Path(path) / relative
        if p.stat().st_size != info["bytes"] or sha256(p) != info["sha256"]:
            raise ValueError(f"Cache content mismatch: {relative}")
    return manifest


class ChunkStore:
    """Memory-mapped document chunk store. get_document returns unpadded arrays."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.ids = read_json(self.path / "ids.json")
        self.id_to_index = {key: i for i, key in enumerate(self.ids)}
        self.tokens = np.memmap(self.path / "tokens.bin", dtype="<u4", mode="r")
        for name in ("doc_offsets", "chunk_offsets", "token_counts", "chunk_starts"):
            setattr(
                self,
                name,
                np.load(self.path / f"{name}.npy", mmap_mode="r", allow_pickle=False),
            )
        if len(self.doc_offsets) != len(self.ids) + 1 or self.doc_offsets[
            -1
        ] + 1 != len(self.chunk_offsets):
            raise ValueError("Corrupt document/chunk offsets")
        if self.chunk_offsets[-1] != len(self.tokens):
            raise ValueError("Corrupt token offsets")

    def __len__(self) -> int:
        return len(self.ids)

    def get_document(self, index: int) -> list[np.ndarray]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        a, b = self.doc_offsets[index : index + 2]
        return [
            self.tokens[self.chunk_offsets[i] : self.chunk_offsets[i + 1]]
            for i in range(int(a), int(b))
        ]
