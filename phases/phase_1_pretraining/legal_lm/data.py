"""Document-disjoint preprocessing and indexed, single-document LM windows.

No raw-text normalization is performed. SQLite keeps text/grouping work bounded
by a document, rather than loading the complete corpus into Python memory.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


SCHEMA_VERSION = 1
WINDOW_DTYPE = np.dtype(
    [
        ("offset", "<u8"),
        ("targets", "<u4"),
        ("source_id", "<u2"),
        ("document_id", "<u8"),
    ]
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _new_directory(path: str | Path) -> Path:
    path = Path(path).resolve()
    path.mkdir(parents=True)
    return path


def iter_json_records(
    path: str | Path, *, chunk_chars: int = 1024 * 1024, max_record_chars: int = 128 * 1024**2
) -> Iterator[dict]:
    """Strictly stream a JSON array, retaining at most one record plus a chunk.

    JSONL and top-level dictionaries are deliberately rejected: their meaning
    must not be guessed when the official files use arrays of records.
    """
    decoder = json.JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as handle:
        buffer, position, eof = "", 0, False

        def refill() -> bool:
            nonlocal buffer, position, eof
            buffer = buffer[position:]
            position = 0
            block = handle.read(chunk_chars)
            eof = not block
            buffer += block
            return bool(block)

        def whitespace() -> None:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position] in " \t\r\n":
                    position += 1
                if position < len(buffer) or eof:
                    return
                refill()

        whitespace()
        if position == len(buffer) or buffer[position] != "[":
            raise ValueError(f"Expected a top-level JSON array: {path}")
        position += 1
        whitespace()
        if position < len(buffer) and buffer[position] == "]":
            position += 1
        else:
            row = 0
            while True:
                while True:
                    try:
                        record, end = decoder.raw_decode(buffer, position)
                        break
                    except json.JSONDecodeError as error:
                        if eof:
                            raise ValueError(
                                f"Malformed/incomplete JSON record {row} in {path}"
                            ) from error
                        if len(buffer) - position > max_record_chars:
                            raise ValueError(
                                f"Record {row} exceeds max_record_chars or is malformed in {path}; inspect before increasing the guard"
                            ) from error
                        refill()
                if not isinstance(record, dict):
                    raise ValueError(f"Record {row} must be a JSON object in {path}")
                if end - position > max_record_chars:
                    raise ValueError(
                        f"Record {row} exceeds max_record_chars in {path}; inspect before increasing the guard"
                    )
                position = end
                yield record
                row += 1
                whitespace()
                if position == len(buffer):
                    raise ValueError(f"Missing closing array in {path}")
                delimiter = buffer[position]
                position += 1
                if delimiter == "]":
                    break
                if delimiter != ",":
                    raise ValueError(f"Expected comma after record {row - 1} in {path}")
                whitespace()
                if position < len(buffer) and buffer[position] == "]":
                    raise ValueError(f"Trailing comma in {path}")
        whitespace()
        if position < len(buffer):
            raise ValueError(f"Unexpected content after JSON array in {path}")


def _connect_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-32768")
    return connection


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _sampled_source_documents(
    connection: sqlite3.Connection, split: str, source_id: int
) -> sqlite3.Cursor:
    """Visit unique texts in frozen hash order without sorting corpus text.

    EXISTS preserves the original per-source deduplication even when a source
    contains repeated occurrences. The split/sample and occurrence indexes let
    callers stop after obtaining their bounded sample or continuation panel.
    """
    return connection.execute(
        """SELECT d.doc_hash,d.text,d.parent
        FROM documents d WHERE d.split=? AND EXISTS (
            SELECT 1 FROM occurrences o
            WHERE o.doc_hash=d.doc_hash AND o.source_id=?)
        ORDER BY d.sample_order,d.doc_hash""",
        (split, source_id),
    )


def _root(connection: sqlite3.Connection, key: str) -> str:
    path = []
    while True:
        parent = connection.execute(
            "SELECT parent FROM documents WHERE doc_hash=?", (key,)
        ).fetchone()[0]
        if parent == key:
            break
        path.append(key)
        key = parent
    for child in path:
        connection.execute("UPDATE documents SET parent=? WHERE doc_hash=?", (key, child))
    return key


def _union(connection: sqlite3.Connection, first: str, second: str) -> None:
    a, b = _root(connection, first), _root(connection, second)
    if a != b:
        low, high = sorted((a, b))
        connection.execute("UPDATE documents SET parent=? WHERE doc_hash=?", (low, high))


def _split(group_hash: str, seed: int, dev_fraction: float, audit_fraction: float) -> str:
    value = (
        int.from_bytes(hashlib.sha256(f"{seed}:split:{group_hash}".encode()).digest()[:8], "big")
        / 2**64
    )
    if value < audit_fraction:
        return "audit"
    if value < audit_fraction + dev_fraction:
        return "dev"
    return "train"


def continuation_spans(
    text: str, *, seed: int, key: str, prompt_words: int = 512, continuation_words: int = 128
) -> tuple[str, str, int] | None:
    """Return contiguous original-text spans without rewriting whitespace.

    The prompt ends at its last word; the reference starts at the next word.
    Both use Python's Unicode-whitespace word convention, matching str.split.
    """
    if prompt_words < 1 or continuation_words < 1:
        raise ValueError("Word counts must be positive")
    words = list(re.finditer(r"\S+", text))
    required = prompt_words + continuation_words
    if len(words) < required:
        return None
    offset = int.from_bytes(hashlib.sha256(f"{seed}:offset:{key}".encode()).digest()[:8], "big") % (
        len(words) - required + 1
    )
    prompt = text[words[offset].start() : words[offset + prompt_words - 1].end()]
    reference = text[words[offset + prompt_words].start() : words[offset + required - 1].end()]
    return prompt, reference, offset


def _make_panels(
    connection: sqlite3.Connection,
    output: Path,
    *,
    seed: int,
    per_source: int,
    prompt_words: int,
    continuation_words: int,
) -> dict:
    panels = {}
    # The audit texts are only materialized here, never used by the fitting APIs.
    for split in ("dev", "audit"):
        prompts, references, details = {}, {}, []
        selected_hashes: set[str] = set()
        counts = {}
        for source_id, source in connection.execute(
            "SELECT source_id,name FROM sources ORDER BY source_id"
        ):
            count = 0
            rows = _sampled_source_documents(connection, split, source_id)
            for doc_hash, text, group_hash in rows:
                if count >= per_source:
                    break
                if doc_hash in selected_hashes:
                    continue
                spans = continuation_spans(
                    text,
                    seed=seed,
                    key=doc_hash,
                    prompt_words=prompt_words,
                    continuation_words=continuation_words,
                )
                if spans is None:
                    continue
                prompt, reference, offset = spans
                example_id = f"{split}_{source_id}_{doc_hash}"
                prompts[example_id] = {"text": prompt}
                references[example_id] = {"text": reference}
                details.append(
                    {
                        "id": example_id,
                        "source": source,
                        "source_id": source_id,
                        "doc_hash": doc_hash,
                        "group_hash": group_hash,
                        "word_offset": offset,
                    }
                )
                selected_hashes.add(doc_hash)
                count += 1
            counts[source] = count
            _progress(f"Prepare: {split} continuation panel for {source}: {count:,} examples")
        _write_json(output / f"{split}_prompts.json", prompts)
        _write_json(output / f"{split}_references.json", references)
        _write_json(output / f"{split}_examples.json", details)
        panels[split] = {
            "prompts": f"{split}_prompts.json",
            "references": f"{split}_references.json",
            "examples": f"{split}_examples.json",
            "num_examples": len(prompts),
            "examples_per_source": counts,
            "requested_per_source": per_source,
            "prompt_words": prompt_words,
            "continuation_words": continuation_words,
        }
        for artifact in ("prompts", "references", "examples"):
            panels[split][f"{artifact}_sha256"] = file_sha256(output / panels[split][artifact])
    return panels


def prepare_corpus(
    source_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    seed: int = 42,
    dev_fraction: float = 0.01,
    audit_fraction: float = 0.01,
    generation_examples_per_source: int = 128,
    prompt_words: int = 512,
    continuation_words: int = 128,
    id_fields: Sequence[str] = (),
    max_record_chars: int = 128 * 1024**2,
) -> dict:
    """Create a deterministic, globally grouped split of official training text.

    id_fields is empty by default. Only explicitly verified globally meaningful
    case identifiers should be supplied; source-local row IDs must not be used.
    Duplicated records retain original multiplicity for source-mixture fidelity.
    """
    if not (
        0 <= dev_fraction < 1 and 0 <= audit_fraction < 1 and dev_fraction + audit_fraction < 1
    ):
        raise ValueError("Invalid split fractions")
    paths = {name: Path(path).resolve() for name, path in source_paths.items()}
    output = _new_directory(output_dir)
    connection = sqlite3.connect(output / "documents.sqlite3")
    try:
        connection.executescript(
            """
            PRAGMA temp_store=FILE;
            CREATE TABLE sources(source_id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL);
            CREATE TABLE documents(doc_hash TEXT PRIMARY KEY, text TEXT NOT NULL,
                parent TEXT NOT NULL, split TEXT, sample_order TEXT NOT NULL);
            CREATE TABLE occurrences(source_id INTEGER NOT NULL, row_number INTEGER NOT NULL,
                doc_hash TEXT NOT NULL, PRIMARY KEY(source_id,row_number));
            CREATE TABLE case_keys(field TEXT NOT NULL, value TEXT NOT NULL, doc_hash TEXT NOT NULL,
                PRIMARY KEY(field,value));
        """
        )
        sources = []
        for source_id, (name, path) in enumerate(sorted(paths.items())):
            _progress(
                f"Prepare: verifying and reading {name} ({path.stat().st_size / 1024**3:.2f} GiB)"
            )
            stat = path.stat()
            info = {
                "name": name,
                "source_id": source_id,
                "path": str(path),
                "bytes": stat.st_size,
                "sha256": file_sha256(path),
                "records": 0,
                "empty_records": 0,
                "nonempty_records": 0,
            }
            connection.execute("INSERT INTO sources VALUES(?,?)", (source_id, name))
            for row_number, record in enumerate(
                iter_json_records(path, max_record_chars=max_record_chars)
            ):
                info["records"] += 1
                if info["records"] % 10_000 == 0:
                    _progress(f"Prepare: {name}: reading record {info['records']:,}")
                text = record.get("text")
                if not isinstance(text, str):
                    raise ValueError(f"{name} record {row_number}: text must be a string")
                if not text.strip():
                    info["empty_records"] += 1
                    continue
                doc_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                sample_order = hashlib.sha256(f"{seed}:sample:{doc_hash}".encode()).hexdigest()
                connection.execute(
                    "INSERT OR IGNORE INTO documents VALUES(?,?,?,?,?)",
                    (doc_hash, text, doc_hash, None, sample_order),
                )
                connection.execute(
                    "INSERT INTO occurrences VALUES(?,?,?)", (source_id, row_number, doc_hash)
                )
                for field in id_fields:
                    identifier = record.get(field)
                    if identifier is None or identifier == "":
                        continue
                    if isinstance(identifier, bool) or not isinstance(identifier, (str, int)):
                        raise ValueError(f"Case identifier {field!r} must be a string or integer")
                    encoded = json.dumps(identifier, ensure_ascii=False)
                    match = connection.execute(
                        "SELECT doc_hash FROM case_keys WHERE field=? AND value=?", (field, encoded)
                    ).fetchone()
                    if match:
                        _union(connection, doc_hash, match[0])
                    else:
                        connection.execute(
                            "INSERT INTO case_keys VALUES(?,?,?)", (field, encoded, doc_hash)
                        )
                info["nonempty_records"] += 1
                if row_number % 1000 == 0:
                    connection.commit()
            if not info["nonempty_records"]:
                raise ValueError(f"Source contains no nonempty documents: {name}")
            sources.append(info)
            connection.commit()
            _progress(
                f"Prepare: {name} complete: {info['records']:,} records, {info['empty_records']:,} empty"
            )
        _progress("Prepare: indexing exact document groups and assigning train/dev/audit splits")
        connection.executescript(
            """
            CREATE INDEX occurrences_doc ON occurrences(doc_hash,source_id);
            CREATE INDEX documents_sample ON documents(sample_order);
        """
        )
        for (doc_hash,) in connection.execute("SELECT doc_hash FROM documents ORDER BY doc_hash"):
            group_hash = _root(connection, doc_hash)
            connection.execute(
                "UPDATE documents SET parent=?,split=? WHERE doc_hash=?",
                (group_hash, _split(group_hash, seed, dev_fraction, audit_fraction), doc_hash),
            )
        connection.execute(
            "CREATE INDEX documents_split_sample ON documents(split,sample_order,doc_hash)"
        )
        connection.commit()
        splits = {}
        for split in ("train", "dev", "audit"):
            splits[split] = {
                "unique_documents": connection.execute(
                    "SELECT COUNT(*) FROM documents WHERE split=?", (split,)
                ).fetchone()[0],
                "document_groups": connection.execute(
                    "SELECT COUNT(DISTINCT parent) FROM documents WHERE split=?", (split,)
                ).fetchone()[0],
                "records_per_source": {
                    name: count
                    for name, count in connection.execute(
                        """
                    SELECT s.name,COUNT(d.doc_hash) FROM sources s
                    LEFT JOIN occurrences o ON o.source_id=s.source_id
                    LEFT JOIN documents d ON d.doc_hash=o.doc_hash AND d.split=?
                    GROUP BY s.source_id ORDER BY s.source_id""",
                        (split,),
                    )
                },
            }
        if not splits["train"]["unique_documents"]:
            raise ValueError("The deterministic split produced no training documents")
        panels = _make_panels(
            connection,
            output,
            seed=seed,
            per_source=generation_examples_per_source,
            prompt_words=prompt_words,
            continuation_words=continuation_words,
        )
        unique_docs = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "prepared_corpus",
            "seed": seed,
            "database": "documents.sqlite3",
            "sources": sources,
            "splits": splits,
            "split_fractions": {
                "train": 1 - dev_fraction - audit_fraction,
                "dev": dev_fraction,
                "audit": audit_fraction,
            },
            "case_id_fields": list(id_fields),
            "text_normalization": "none",
            "max_record_chars": max_record_chars,
            "duplicate_policy": "Global exact-text/case grouping; preserve record multiplicity in model data",
            "unique_documents": unique_docs,
            "duplicate_records": sum(s["nonempty_records"] for s in sources) - unique_docs,
            "near_duplicate_policy": "Not removed; exact hashes do not establish absence of near-duplicate leakage",
            "panels": panels,
        }
        connection.commit()
    finally:
        connection.close()
    _progress("Prepare: verifying completed database checksum")
    manifest["database_sha256"] = file_sha256(output / "documents.sqlite3")
    _write_json(output / "manifest.json", manifest)
    _progress(f"Prepare: complete: {manifest['unique_documents']:,} unique documents")
    return manifest


def read_prepared_manifest(prepared_dir: str | Path, *, verify_database: bool = False) -> dict:
    directory = Path(prepared_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("kind") != "prepared_corpus"
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("Unsupported prepared-corpus manifest")
    if (
        verify_database
        and file_sha256(directory / manifest["database"]) != manifest["database_sha256"]
    ):
        raise ValueError("Prepared database checksum mismatch")
    return manifest


def iter_prepared_documents(
    prepared_dir: str | Path,
    split: str,
    *,
    source_id: int | None = None,
    unique: bool = False,
    sample_order: bool = False,
) -> Iterator[tuple[int, str, str]]:
    """Yield (source_id, document_hash, original_text); unique is per source."""
    if split not in {"train", "dev", "audit"}:
        raise ValueError("Unknown split")
    manifest = read_prepared_manifest(prepared_dir)
    connection = _connect_readonly(Path(prepared_dir) / manifest["database"])
    try:
        if unique and sample_order and source_id is not None:
            for doc_hash, text, _ in _sampled_source_documents(connection, split, source_id):
                yield source_id, doc_hash, text
            return
        # Keep occurrences as the outer loop for full tokenization. A reorderable
        # JOIN otherwise encourages SQLite to start with the split index and
        # sort the complete selected text corpus before yielding any records.
        join = "CROSS JOIN" if not unique and not sample_order else "JOIN"
        query = f"""SELECT o.source_id,d.doc_hash,d.text FROM occurrences o
                    {join} documents d ON d.doc_hash=o.doc_hash WHERE d.split=?"""
        parameters: list = [split]
        if source_id is not None:
            query += " AND o.source_id=?"
            parameters.append(source_id)
        if unique:
            query += " GROUP BY o.source_id,d.doc_hash"
        query += (
            " ORDER BY d.sample_order,o.source_id,d.doc_hash"
            if sample_order
            else " ORDER BY o.source_id,o.row_number"
        )
        yield from connection.execute(query, parameters)
    finally:
        connection.close()


def _document_batches(
    documents: Iterable[tuple[int, str, str]], *, batch_size: int, max_batch_bytes: int
) -> Iterator[list[tuple[int, str, str]]]:
    """Preserve document order while bounding normal batches by UTF-8 bytes.

    A document larger than the byte budget is emitted alone and never split or
    truncated. The upstream iterator may hold one lookahead document while the
    preceding batch is encoded.
    """
    batch, size = [], 0
    for document in documents:
        document_bytes = len(document[2].encode("utf-8"))
        if batch and (len(batch) >= batch_size or size + document_bytes > max_batch_bytes):
            yield batch
            batch, size = [], 0
        batch.append(document)
        size += document_bytes
        if len(batch) >= batch_size or size >= max_batch_bytes:
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


def _encoded_documents(
    documents: Iterable[tuple[int, str, str]], tokenizer, *, batch_size: int, max_batch_bytes: int
) -> Iterator[tuple[int, str, list[int]]]:
    for batch in _document_batches(
        documents, batch_size=batch_size, max_batch_bytes=max_batch_bytes
    ):
        if batch_size == 1:
            encoded = [tokenizer.encode(batch[0][2])]
        else:
            # The wrapper has disabled special-marker recognition on this same
            # backend. Encode each full document independently, in native code.
            encoded = [
                item.ids
                for item in tokenizer.tokenizer.encode_batch(
                    [row[2] for row in batch], add_special_tokens=False
                )
            ]
        for (source_id, doc_hash, _), text_ids in zip(batch, encoded):
            yield source_id, doc_hash, text_ids
        del batch, encoded, text_ids


def tokenize_corpus(
    prepared_dir: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    *,
    context_length: int = 2048,
    encode_batch_size: int = 64,
    max_batch_bytes: int = 8 * 1024**2,
) -> dict:
    """Freeze full-document token streams and nonoverlapping target windows.

    Each adjacent window shares one input/target boundary token, but no target
    position is counted twice. EOS is only present at the real document end.
    Audit language-model windows are prepared here, but no trainer reads them.
    """
    from .tokenization import load_tokenizer

    if context_length < 1 or context_length >= 2**32:
        raise ValueError("context_length must fit a positive uint32")
    if encode_batch_size < 1 or max_batch_bytes < 1:
        raise ValueError("Encoding batch size and byte budget must be positive")
    prepared_dir = Path(prepared_dir).resolve()
    prepared = read_prepared_manifest(prepared_dir)
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.vocab_size > 65536:
        raise ValueError("Vocabulary cannot be stored in uint16")
    tokenizer_path = tokenizer.path
    tokenizer_manifest = json.loads(
        (tokenizer_path.parent / "manifest.json").read_text(encoding="utf-8")
    )
    prepared_sha = file_sha256(prepared_dir / "manifest.json")
    if (
        tokenizer_manifest.get("kind") != "legal_byte_bpe"
        or tokenizer_manifest.get("sampling_split") != "train"
    ):
        raise ValueError("Missing training-only tokenizer provenance")
    if tokenizer_manifest.get("prepared_manifest_sha256") != prepared_sha:
        raise ValueError("Tokenizer was fitted against a different prepared corpus/split")
    if tokenizer_manifest.get("tokenizer_sha256") != file_sha256(tokenizer_path):
        raise ValueError("Tokenizer checksum mismatch")
    output = _new_directory(output_dir)
    sources = [source["name"] for source in prepared["sources"]]
    splits = {}
    for split in ("train", "dev", "audit"):
        _progress(f"Tokenize: starting {split}")
        token_count, window_count, target_count, document_count = 0, 0, 0, 0
        per_source = {name: 0 for name in sources}
        current_source, source_documents = None, 0
        with (output / f"{split}.tokens.bin").open("wb") as tokens_out, (
            output / f"{split}.windows.bin"
        ).open("wb") as windows_out, (output / f"{split}.documents.jsonl").open(
            "w", encoding="utf-8"
        ) as docs_out:
            documents = iter_prepared_documents(prepared_dir, split)
            for source_id, doc_hash, text_ids in _encoded_documents(
                documents, tokenizer, batch_size=encode_batch_size, max_batch_bytes=max_batch_bytes
            ):
                if source_id != current_source:
                    if current_source is not None:
                        _progress(
                            f"Tokenize: {split}/{sources[current_source]} complete: {source_documents:,} documents"
                        )
                    current_source, source_documents = source_id, 0
                    _progress(f"Tokenize: starting {split}/{sources[source_id]}")
                ids = np.asarray([tokenizer.bos_id, *text_ids, tokenizer.eos_id], dtype="<u2")
                targets = len(ids) - 1
                count = math.ceil(targets / context_length)
                entries = np.empty(count, dtype=WINDOW_DTYPE)
                starts = np.arange(count, dtype=np.uint64) * context_length
                entries["offset"] = token_count + starts
                entries["targets"] = np.minimum(context_length, targets - starts)
                entries["source_id"] = source_id
                entries["document_id"] = document_count
                tokens_out.write(ids.tobytes())
                windows_out.write(entries.tobytes())
                docs_out.write(
                    json.dumps(
                        {
                            "document_id": document_count,
                            "doc_hash": doc_hash,
                            "source_id": source_id,
                            "token_offset": token_count,
                            "num_tokens": len(ids),
                            "first_window": window_count,
                            "num_windows": count,
                        }
                    )
                    + "\n"
                )
                token_count += len(ids)
                window_count += count
                target_count += targets
                document_count += 1
                source_documents += 1
                per_source[sources[source_id]] += targets
                if source_documents % 10_000 == 0:
                    _progress(
                        f"Tokenize: {split}/{sources[source_id]}: {source_documents:,} documents; {target_count:,} split target tokens"
                    )
        if current_source is not None:
            _progress(
                f"Tokenize: {split}/{sources[current_source]} complete: {source_documents:,} documents"
            )
        _progress(
            f"Tokenize: verifying {split} artifact checksums ({document_count:,} documents, {target_count:,} target tokens)"
        )
        splits[split] = {
            "tokens_file": f"{split}.tokens.bin",
            "windows_file": f"{split}.windows.bin",
            "documents_file": f"{split}.documents.jsonl",
            "num_tokens": token_count,
            "num_windows": window_count,
            "target_tokens": target_count,
            "num_documents": document_count,
            "source_target_tokens": per_source,
            "padding_fraction": 1 - target_count / (window_count * context_length)
            if window_count
            else 0.0,
            "tokens_sha256": file_sha256(output / f"{split}.tokens.bin"),
            "windows_sha256": file_sha256(output / f"{split}.windows.bin"),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "token_windows",
        "context_length": context_length,
        "token_dtype": "<u2",
        "window_dtype": WINDOW_DTYPE.descr,
        "sources": sources,
        "vocab_size": tokenizer.vocab_size,
        "special_ids": {"pad": tokenizer.pad_id, "bos": tokenizer.bos_id, "eos": tokenizer.eos_id},
        "prepared_dir": str(prepared_dir),
        "prepared_manifest_sha256": prepared_sha,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "splits": splits,
        "panels": prepared["panels"],
        "encoding": {
            "implementation": "serial" if encode_batch_size == 1 else "native_batch",
            "batch_size": encode_batch_size,
            "max_batch_utf8_bytes": max_batch_bytes,
            "oversized_document_policy": "Encode alone without splitting or truncation",
            "add_special_tokens": False,
            "ordinary_special_marker_recognition": False,
        },
        "sampling_policy": "Uniform windows; report actual valid target counts per source (short final windows reduce padding efficiency)",
        "unique_text_tokens_note": "Record multiplicity preserved; target_tokens are corpus target positions, not deduplicated-text positions",
    }
    _write_json(output / "manifest.json", manifest)
    _progress(
        f"Tokenize: complete: {sum(info['target_tokens'] for info in splits.values()):,} corpus target tokens"
    )
    return manifest


def verify_token_artifacts(token_dir: str | Path, splits: Sequence[str] = ("train", "dev")) -> dict:
    """Read token-window metadata and check its tokenizer compatibility."""
    token_dir = Path(token_dir).resolve()
    manifest = json.loads((token_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "token_windows" or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported token-window manifest")
    tokenizer_path = Path(manifest["tokenizer_path"])
    if file_sha256(tokenizer_path) != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer checksum mismatch")
    return manifest


def verify_panel_artifacts(token_dir: str | Path, split: str = "dev") -> dict:
    """Read the continuation panel filenames for development evaluation."""
    if split not in {"dev", "audit"}:
        raise ValueError("Continuation panels are available only for dev/audit")
    token_dir = Path(token_dir).resolve()
    manifest = json.loads((token_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "token_windows" or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported token-window manifest")
    prepared_dir = Path(manifest["prepared_dir"])
    panel = manifest["panels"][split]
    return panel


class TokenWindowDataset(Dataset):
    """Read-only, compact indexed causal examples; padding labels are -100."""

    def __init__(
        self, token_dir: str | Path, split: str = "train", context_length: int | None = None
    ):
        self.token_dir = Path(token_dir).resolve()
        self.manifest = json.loads((self.token_dir / "manifest.json").read_text(encoding="utf-8"))
        if (
            self.manifest.get("kind") != "token_windows"
            or self.manifest.get("schema_version") != SCHEMA_VERSION
        ):
            raise ValueError("Unsupported token-window manifest")
        if split not in {"train", "dev", "audit"}:
            raise ValueError("Unknown split")
        self.split = split
        self.context_length = self.manifest["context_length"]
        if context_length is not None and context_length != self.context_length:
            raise ValueError("Context length differs from the frozen window manifest")
        self.source_names = self.manifest["sources"]
        self.pad_id = self.manifest["special_ids"]["pad"]
        info = self.manifest["splits"][split]
        token_path, window_path = (
            self.token_dir / info["tokens_file"],
            self.token_dir / info["windows_file"],
        )
        if (
            token_path.stat().st_size != info["num_tokens"] * 2
            or window_path.stat().st_size != info["num_windows"] * WINDOW_DTYPE.itemsize
        ):
            raise ValueError("Token/index file size differs from its manifest")
        self.tokens = (
            np.memmap(token_path, dtype="<u2", mode="r")
            if info["num_tokens"]
            else np.empty(0, dtype="<u2")
        )
        self.windows = (
            np.memmap(window_path, dtype=WINDOW_DTYPE, mode="r")
            if info["num_windows"]
            else np.empty(0, dtype=WINDOW_DTYPE)
        )
        self.target_counts = self.windows["targets"]
        self.source_ids = self.windows["source_id"]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        entry = self.windows[index]
        offset, targets = int(entry["offset"]), int(entry["targets"])
        if not 1 <= targets <= self.context_length or offset + targets >= len(self.tokens):
            raise ValueError("Invalid token-window index entry")
        input_ids = torch.full((self.context_length,), self.pad_id, dtype=torch.long)
        labels = torch.full((self.context_length,), -100, dtype=torch.long)
        input_ids[:targets] = torch.from_numpy(
            self.tokens[offset : offset + targets].astype(np.int64)
        )
        labels[:targets] = torch.from_numpy(
            self.tokens[offset + 1 : offset + targets + 1].astype(np.int64)
        )
        return {"input_ids": input_ids, "labels": labels, "source_id": int(entry["source_id"])}
