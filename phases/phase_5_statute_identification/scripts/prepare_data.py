#!/usr/bin/env python3
"""Lossless own-BPE cache; canonical statute sets; label-blind DEV panels."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from tokenizers import Tokenizer

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE))
from lsi.data import TOKENIZER_SHA256, dev_partition, sha256_file
from lsi.foundation import read_manifest


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    output = PHASE / "data/prepared/full-bpe-v1"
    output.mkdir(parents=True)
    raw_path = PHASE / "metadata/raw-data.json"
    raw = json.loads(raw_path.read_text())
    foundation_path = (PHASE / "../../assets/foundation/manifest.json").resolve()
    tok = Path(read_manifest(foundation_path)["tokenizer_path"])
    if sha256_file(tok) != TOKENIZER_SHA256:
        raise ValueError("Original own BPE changed")
    tokenizer = Tokenizer.from_file(str(tok))
    manifest = {
        "format": "cs781-p5-full-bpe-v1",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_manifest": str(raw_path),
        "raw_manifest_sha256": sha256_file(raw_path),
        "repository_revision": raw["revision"],
        "tokenizer_path": str(tok),
        "tokenizer_sha256": TOKENIZER_SHA256,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(__file__),
        "tokens_dtype": "<u4",
        "offsets_dtype": "<u8",
        "num_labels": 100,
        "content_policy": "all original text, no truncation, no added special tokens, no fitted transformations",
        "label_policy": "sorted unique original IDs; raw release unchanged; no TEST targets",
        "dev_panel_policy": "SHA256(cs781-lsi-dev-v1|42|text_sha256) first-byte parity; no labels used",
        "splits": {},
    }
    frequency = Counter()
    for split, expected in (("train", 42750), ("dev", 10181), ("test", 13019)):
        identity = next(row for row in raw["files"] if Path(row["repository_path"]).stem == split)
        path = Path(identity["path"])
        rows = json.loads(path.read_text())
        if len(rows) != expected or len({row["id"] for row in rows}) != expected:
            raise ValueError("Original case count/IDs differ")
        offsets, records, duplicate_labels = [0], [], []
        partial = output / (split + ".tokens.u32.partial")
        with partial.open("xb") as stream:
            for begin in range(0, len(rows), 128):
                batch = rows[begin : begin + 128]
                encoded = tokenizer.encode_batch(
                    [row["text"] for row in batch], add_special_tokens=False
                )
                for index, (row, entry) in enumerate(zip(batch, encoded, strict=True), begin):
                    wanted = {"id", "text"} if split == "test" else {"id", "text", "labels"}
                    if (
                        set(row) != wanted
                        or not isinstance(row["id"], str)
                        or not row["id"].strip()
                        or not isinstance(row["text"], str)
                    ):
                        raise ValueError("Raw schema differs")
                    ids = entry.ids
                    if any(type(x) is not int or not 3 <= x < 32768 for x in ids):
                        raise ValueError("Content contains a special/foreign token")
                    labels = None
                    if split != "test":
                        labels = row["labels"]
                        if (
                            not isinstance(labels, list)
                            or not labels
                            or any(type(x) is not int or not 0 <= x < 100 for x in labels)
                        ):
                            raise ValueError("Public target schema differs")
                        if len(set(labels)) != len(labels):
                            duplicate_labels.append(row["id"])
                        labels = sorted(set(labels))
                        if split == "train":
                            frequency.update(labels)
                    np.asarray(ids, dtype="<u4").tofile(stream)
                    offsets.append(offsets[-1] + len(ids))
                    text_sha = hashlib.sha256(row["text"].encode()).hexdigest()
                    records.append(
                        {
                            "id": row["id"],
                            "text_sha256": text_sha,
                            "characters": len(row["text"]),
                            "tokens": len(ids),
                            "source_row_indices": [index],
                            "labels": labels,
                            "dev_partition": dev_partition(text_sha) if split == "dev" else split,
                        }
                    )
        tokens_path = output / (split + ".tokens.npy")
        with tokens_path.open("xb") as stream, partial.open("rb") as source:
            np.lib.format.write_array_header_2_0(
                stream, {"descr": "<u4", "fortran_order": False, "shape": (offsets[-1],)}
            )
            shutil.copyfileobj(source, stream, length=8 * 1024 * 1024)
        partial.unlink()
        offsets_path = output / (split + ".offsets.npy")
        with offsets_path.open("xb") as stream:
            np.save(stream, np.asarray(offsets, dtype="<u8"), allow_pickle=False)
        metadata_path = output / (split + ".metadata.json")
        write_new(
            metadata_path,
            {
                "format": "cs781-p5-case-metadata-v1",
                "split": split,
                "raw_identity": identity,
                "records": records,
            },
        )
        manifest["splits"][split] = {
            "documents": len(rows),
            "content_tokens": offsets[-1],
            "duplicate_label_rows_canonicalized": duplicate_labels,
            "panels": dict(Counter(row["dev_partition"] for row in records)),
            "artifacts": {
                kind: {"path": str(file), "bytes": file.stat().st_size, "sha256": sha256_file(file)}
                for kind, file in (
                    ("tokens", tokens_path),
                    ("offsets", offsets_path),
                    ("metadata", metadata_path),
                )
            },
        }
        print(
            json.dumps(
                {
                    "prepared": split,
                    "cases": len(rows),
                    "tokens": offsets[-1],
                    "panels": manifest["splits"][split]["panels"],
                }
            ),
            flush=True,
        )
        del rows, records
    manifest["train_label_frequencies"] = [frequency[index] for index in range(100)]
    if min(manifest["train_label_frequencies"]) <= 0:
        raise ValueError("Every target must occur in TRAIN")
    write_new(PHASE / "data/prepared/manifest.json", manifest)
    print(
        json.dumps({"completed": True, "manifest": str(PHASE / "data/prepared/manifest.json")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
