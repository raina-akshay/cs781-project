"""Pinned public Phase 3 data, full-token sentence cache.

The cache preserves every original occurrence. Only the window dataset truncates
inputs, so changing an input budget never silently changes the source cache.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer


CACHE_FORMAT = "cs781-p3-sentence-cache-v1"
REPOSITORY = "Exploration-Lab/CS-781-Capstone"
TOKENIZER_SHA256 = "25cb054b95fb14d53f12f265c4e09c897ba6dfc5d84b8298ab93004184648b1d"
VOCAB_SIZE = 32768
LABELS = (
    "Fact",
    "Issue",
    "ArgumentPetitioner",
    "ArgumentRespondent",
    "Statute",
    "Dissent",
    "PrecedentReliedUpon",
    "PrecedentNotReliedUpon",
    "PrecedentOverruled",
    "RulingByLowerCourt",
    "RatioOfTheDecision",
    "RulingByPresentCourt",
    "None",
)
SOURCES = tuple(
    f"Phase_3_RR/{'Test' if split == 'test' else 'Train'}/{domain}_{split}.json"
    for domain in ("CL", "IT")
    for split in ("train", "dev", "test")
) + ("Phase_3_RR/Train/label_vocab.json",)


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        obj[key] = value
    return obj


def strict_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalized_id(doc_id: Any) -> str:
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("Document IDs must be nonempty strings.")
    normalized = doc_id[:-4] if doc_id.endswith(".txt") else doc_id
    if not normalized.strip():
        raise ValueError("Document ID becomes empty after .txt normalization.")
    return normalized


def _check_cache_schema(cache: Any) -> None:
    if cache["format"] != CACHE_FORMAT or cache["labels"] != list(LABELS):
        raise ValueError("Wrong cache format or ordered role vocabulary")
    if cache["tokenizer_sha256"] != TOKENIZER_SHA256:
        raise ValueError("Cache uses a different foundation tokenizer")
    ids: set[str] = set()
    for doc in cache["documents"]:
        normalized = _normalized_id(doc["id"])
        if normalized in ids:
            raise ValueError(f"Duplicate document ID after .txt normalization: {normalized}")
        ids.add(normalized)
        if doc["domain"] not in ("CL", "IT") or doc["split"] not in ("train", "dev", "test"):
            raise ValueError("Unknown document domain or split.")
        if not isinstance(doc["sentences"], list) or not doc["sentences"]:
            raise ValueError("Every cached document must contain sentence records.")
        for index, sentence in enumerate(doc["sentences"]):
            if not _is_int(sentence["index"]) or sentence["index"] != index:
                raise ValueError("Cached sentence indices must preserve contiguous original order.")
            tokens = sentence["tokens"]
            if not isinstance(tokens, list) or any(
                not _is_int(token) or not 3 <= token < VOCAB_SIZE for token in tokens
            ):
                raise ValueError("Cached tokens must be unspecialized own-vocabulary IDs >=3.")
            label = sentence["label"]
            if doc["split"] == "test":
                if label is not None:
                    raise ValueError("Public test sentences must not contain training labels.")
            elif not _is_int(label) or not 0 <= label < len(LABELS):
                raise ValueError("Training/dev sentences require an official integer role ID.")


def _build_cache(provenance_path: Path, tokenizer_path: Path) -> dict:
    provenance_path, tokenizer_path = provenance_path.resolve(), tokenizer_path.resolve()
    provenance = strict_json(provenance_path)
    source_paths = {record["path"]: Path(record["local_path"]) for record in provenance["files"]}
    hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    vocab = strict_json(source_paths[SOURCES[-1]])
    expected_vocab = {label: index for index, label in enumerate(LABELS)}
    if (
        not isinstance(vocab, dict)
        or vocab != expected_vocab
        or any(not _is_int(value) for value in vocab.values())
    ):
        raise ValueError("Official label vocabulary or ID mapping differs.")
    if sha256_file(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("Tokenizer is not the pinned own S5 tokenizer.")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if (
        tokenizer.get_vocab_size() != VOCAB_SIZE
        or tokenizer.truncation is not None
        or tokenizer.padding is not None
    ):
        raise ValueError(
            "Full-token cache requires the unchanged own tokenizer without truncation/padding."
        )
    documents = []
    for domain in ("CL", "IT"):
        for split in ("train", "dev", "test"):
            name = f"Phase_3_RR/{'Test' if split == 'test' else 'Train'}/{domain}_{split}.json"
            obj = strict_json(source_paths[name])
            if split == "test":
                if not isinstance(obj, dict):
                    raise ValueError("Public test schema must map IDs directly to sentence lists.")
                rows = [{"id": doc_id, "text": texts} for doc_id, texts in obj.items()]
            else:
                if not isinstance(obj, list):
                    raise ValueError("Public train/dev schema must be a list of document objects.")
                rows = obj
            for row in rows:
                expected_fields = {"id", "text"} if split == "test" else {"id", "text", "labels"}
                if not isinstance(row, dict) or set(row) != expected_fields:
                    raise ValueError("Unexpected official raw document schema.")
                _normalized_id(row["id"])
                texts = row["text"]
                if (
                    not isinstance(texts, list)
                    or not texts
                    or any(not isinstance(text, str) for text in texts)
                ):
                    raise ValueError("Raw sentences must be a nonempty list of strings.")
                if split != "test":
                    if not isinstance(row["labels"], list) or len(row["labels"]) != len(texts):
                        raise ValueError("Raw labels must align with the original sentence list.")
                    if any(
                        not isinstance(label, str) or label not in expected_vocab
                        for label in row["labels"]
                    ):
                        raise ValueError("Unknown official raw label.")
                encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
                sentences = [
                    {
                        "tokens": encoding.ids,
                        "label": None if split == "test" else expected_vocab[row["labels"][index]],
                        "index": index,
                    }
                    for index, encoding in enumerate(encodings)
                ]
                documents.append(
                    {"id": row["id"], "domain": domain, "split": split, "sentences": sentences}
                )
    cache = {
        "format": CACHE_FORMAT,
        "labels": list(LABELS),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": TOKENIZER_SHA256,
        "provenance_path": str(provenance_path),
        "provenance_sha256": sha256_file(provenance_path),
        "raw_sha256": hashes,
        "documents": documents,
    }
    _check_cache_schema(cache)
    return cache


def prepare_cache(
    provenance_path: str | Path, tokenizer_path: str | Path, output_path: str | Path
) -> dict:
    """Tokenize the official files and write their complete sentence cache."""
    cache = _build_cache(Path(provenance_path), Path(tokenizer_path))
    data = (
        json.dumps(cache, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return cache


def load_cache(path: str | Path) -> dict:
    """Read tokenized sentences and check IDs, label alignment and tokenizer."""
    cache = strict_json(path)
    _check_cache_schema(cache)
    return cache


def flat_records(cache: dict, splits: Iterable[str] = ("train",)) -> list[dict]:
    """Flatten selected splits, retaining a stable global original row index."""
    chosen = tuple(splits)
    if not chosen or any(split not in ("train", "dev", "test") for split in chosen):
        raise ValueError("Select only official train/dev/test splits.")
    records = []
    row_index = 0
    for doc in cache["documents"]:
        for sentence in doc["sentences"]:
            if doc["split"] in chosen:
                records.append(
                    {
                        "tokens": sentence["tokens"],
                        "label": sentence["label"],
                        "doc_id": doc["id"],
                        "domain": doc["domain"],
                        "sentence_index": sentence["index"],
                        "row_index": row_index,
                    }
                )
            row_index += 1
    return records
