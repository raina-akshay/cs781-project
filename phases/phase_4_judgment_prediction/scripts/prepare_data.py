#!/usr/bin/env python3
"""Tokenize complete cases into flat arrays with one offset range per case."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_SHA256 = "25cb054b95fb14d53f12f265c4e09c897ba6dfc5d84b8298ab93004184648b1d"
SPLITS = ("train", "dev", "test")


def load_records(path, split):
    raw_rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"{split}: expected a nonempty JSON list")
    rows, by_id = [], {}
    for index, row in enumerate(raw_rows):
        if not isinstance(row["id"], str) or not row["id"] or not isinstance(row["text"], str):
            raise ValueError(f"{split}[{index}]: expected a string case ID and text")
        clean = {"id": row["id"], "text": row["text"]}
        # Released test targets are deliberately excluded from operational data.
        if split != "test":
            if type(row["label"]) is not int or row["label"] not in (0, 1):
                raise ValueError(f"{split}[{index}]: label must be integer 0 or 1")
            clean["label"] = row["label"]
        if clean["id"] in by_id:
            prior = by_id[clean["id"]]
            if split == "test" or any(prior[key] != value for key, value in clean.items()):
                raise ValueError(f"{split}[{index}]: conflicting or duplicate case ID")
            prior["_source_indices"].append(index)
        else:
            clean["_source_indices"] = [index]
            by_id[clean["id"]] = clean
            rows.append(clean)
    return rows


def build_split(rows, split, tokenizer, output, batch_size, raw_identity):
    partial = output / f"{split}.tokens.u32.partial"
    tokens_path = output / f"{split}.tokens.npy"
    offsets_path = output / f"{split}.offsets.npy"
    metadata_path = output / f"{split}.metadata.json"
    offsets, metadata = [0], []
    with partial.open("wb") as stream:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encodings = tokenizer.encode_batch(
                [row["text"] for row in batch], add_special_tokens=False
            )
            for row, encoding in zip(batch, encodings):
                ids = encoding.ids
                np.asarray(ids, dtype="<u4").tofile(stream)
                offsets.append(offsets[-1] + len(ids))
                metadata.append(
                    {
                        "id": row["id"],
                        "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                        "characters": len(row["text"]),
                        "tokens": len(ids),
                        "source_row_indices": row["_source_indices"],
                        "label": row["label"] if split != "test" else None,
                    }
                )
    # The flat token count is known only after streaming all cases.
    with tokens_path.open("wb") as stream, partial.open("rb") as raw:
        np.lib.format.write_array_header_2_0(
            stream,
            {
                "descr": "<u4",
                "fortran_order": False,
                "shape": (offsets[-1],),
            },
        )
        shutil.copyfileobj(raw, stream, length=8 * 1024 * 1024)
    partial.unlink()
    with offsets_path.open("wb") as stream:
        np.save(stream, np.asarray(offsets, dtype="<u8"), allow_pickle=False)
    metadata_path.write_text(
        json.dumps(
            {
                "format": "cs781-p4-raw-case-metadata-v1",
                "split": split,
                "raw_identity": raw_identity,
                "records": metadata,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = {
        kind: {"path": str(path)}
        for kind, path in (
            ("tokens", tokens_path),
            ("offsets", offsets_path),
            ("metadata", metadata_path),
        )
    }
    return {"documents": len(rows), "content_tokens": offsets[-1], "artifacts": artifacts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/cache/full-bpe-v1")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_manifest = json.loads((ROOT / "data/raw/source-manifest.json").read_text())
    raw_by_split = {record["split"]: record for record in raw_manifest["files"]}
    tokenizer_path = (ROOT / "../../assets/foundation/tokenizer.json").resolve()
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Tokenizer differs from the selected foundation")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    manifest = {
        "format": "cs781-p4-full-bpe-v1",
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tokens_dtype": "<u4",
        "offsets_dtype": "<u8",
        "splits": {},
    }
    for split in SPLITS:
        source = raw_by_split[split]
        rows = load_records(Path(source["project_symlink"]), split)
        manifest["splits"][split] = build_split(
            rows, split, tokenizer, output, args.batch_size, source
        )
        print(f"{split}: {len(rows)} cases")
    payload = json.dumps(manifest, indent=2) + "\n"
    (output / "manifest.json").write_text(payload)
    prepared = ROOT / "data/prepared/manifest.json"
    prepared.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_text(payload)
    print(f"Prepared data: {prepared}")


if __name__ == "__main__":
    main()
