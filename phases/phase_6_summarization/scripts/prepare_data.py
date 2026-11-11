"""Fit lexical evidence only on the frozen TRAIN IDs; preserve every target."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from summ.common import PHASE, ROOT, atomic_json, now, read_json, sha256
from summ.data import own_tokenizer
from summ.selection import select_evidence
from summ.foundation import read_manifest


def save_npy(path, values, dtype):
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.save(stream, np.asarray(values, dtype=dtype), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    args = parser.parse_args()
    c = read_json(args.config)
    root = Path(c["data_manifest"]).parent
    root.mkdir(parents=True, exist_ok=True)
    status = ROOT / "artifacts/preparation/status.json"
    start = time.monotonic()
    train = read_json(PHASE / "data/raw/Phase_6_Summ/Train/train.json")
    test = read_json(PHASE / "data/raw/Phase_6_Summ/Test/test.json")
    if len(train) != 7030 or len(test) != 100:
        raise ValueError("Wrong release")
    by_id = {r["id"]: r for r in train}
    if len(by_id) != 7030 or any(set(r) != {"id", "document", "summary"} for r in train):
        raise ValueError("Wrong TRAIN schema")
    if any(set(r) != {"id", "document"} for r in test):
        raise ValueError("TEST references must be absent")
    split_path = PHASE / "metadata/split.json"
    split = read_json(split_path)
    tr, va = split["TRAIN_ids"], split["VAL_ids"]
    if (
        len(tr) != 6328
        or len(va) != 700
        or set(tr) & set(va)
        or set(tr) | set(va) != set(by_id) - {"299", "4799"}
        or any(not by_id[i]["summary"].strip() for i in tr + va)
    ):
        raise ValueError("Frozen group-aware split is invalid")
    atomic_json(root / "split.json", split)
    atomic_json(
        status, {"state": "fitting_train_only_tfidf", "at_utc": now(), "train_cases": len(tr)}
    )
    vectorizer = TfidfVectorizer(
        max_features=60000,
        min_df=1,
        sublinear_tf=True,
        dtype=np.float32,
        strip_accents="unicode",
        lowercase=True,
        norm="l2",
    )
    vectorizer.fit(by_id[i]["document"] for i in tr)
    joblib.dump(vectorizer, root / "selector.joblib", compress=3)
    tok = own_tokenizer(c["foundation_manifest"])
    files = {}
    totals = {}
    for name, rows in (
        ("train", [by_id[i] for i in tr]),
        ("val", [by_id[i] for i in va]),
        ("test", test),
    ):
        source = []
        source_offsets = [0]
        target = []
        target_offsets = [0]
        metadata = []
        original_tokens = selected_tokens = supervised_tokens = windows = 0
        refs = []
        for n, row in enumerate(rows):
            full = tok.encode(row["document"])
            evidence, spans = select_evidence(
                full, tok, vectorizer, budget=c["sequence"]["source_tokens"]
            )
            if any(t < 3 or t >= tok.vocab_size for t in evidence):
                raise ValueError("Source control collision")
            source.extend(evidence)
            source_offsets.append(len(source))
            kept = sum(b - a for a, b in spans)
            meta = {
                "id": row["id"],
                "original_source_tokens": len(full),
                "selected_content_tokens": kept,
                "selected_serialized_tokens": len(evidence),
                "selected_spans": spans,
            }
            original_tokens += len(full)
            selected_tokens += kept
            if name != "test":
                ref = tok.encode(row["summary"])
                if not ref or any(t < 3 or t >= tok.vocab_size for t in ref):
                    raise ValueError("Reference control collision/empty")
                target.extend(ref)
                target_offsets.append(len(target))
                meta["reference_tokens"] = len(ref)
                supervised_tokens += len(ref) + 1
                windows += (len(ref) + 1 + 511) // 512
                refs.append({"id": row["id"], "summary": row["summary"]})
            metadata.append(meta)
            if n % 100 == 0:
                atomic_json(
                    status,
                    {
                        "state": "selecting_evidence",
                        "split": name,
                        "cases_done": n,
                        "cases_total": len(rows),
                        "at_utc": now(),
                        "elapsed_seconds": time.monotonic() - start,
                    },
                )
        save_npy(root / (name + ".source.npy"), source, np.uint16)
        save_npy(root / (name + ".source-offsets.npy"), source_offsets, np.int64)
        atomic_json(root / (name + ".metadata.json"), metadata)
        if name != "test":
            save_npy(root / (name + ".target.npy"), target, np.uint16)
            save_npy(root / (name + ".target-offsets.npy"), target_offsets, np.int64)
            atomic_json(root / (name + ".references.json"), refs)
        totals[name] = {
            "cases": len(rows),
            "original_source_tokens": original_tokens,
            "selected_content_tokens": selected_tokens,
            "content_token_coverage": selected_tokens / original_tokens,
            "supervised_tokens_including_EOS": supervised_tokens,
            "complete_target_windows": windows,
        }
    for p in root.iterdir():
        if p.is_file() and p.name != "manifest.json":
            files[p.name] = sha256(p)
    manifest = {
        "format": "cs781-phase6-s1-prepared-v1",
        "created_at_utc": now(),
        "raw_provenance_sha256": sha256(PHASE / "metadata/data-provenance.json"),
        "split_proposal_sha256": sha256(split_path),
        "tokenizer_sha256": sha256(read_manifest(c["foundation_manifest"])["tokenizer_path"]),
        "selector": {
            "kind": "TRAIN_only_TFIDF_MMR_original_token_passages",
            "max_features": 60000,
            "min_df": 1,
            "passage_tokens": 192,
            "opening_tokens": 128,
            "conclusion_tokens": 192,
            "relevance_weight": 0.7,
            "source_budget": 1536,
            "fit_ids_sha256": sha256(root / "split.json"),
            "test_targets_used": False,
        },
        "excluded_empty_reference_ids": ["299", "4799"],
        "totals": totals,
        "files": files,
    }
    atomic_json(c["data_manifest"], manifest)
    atomic_json(
        status,
        {
            "state": "completed",
            "at_utc": now(),
            "elapsed_seconds": time.monotonic() - start,
            "totals": totals,
        },
    )
    print(
        json.dumps({"passed": True, "manifest": c["data_manifest"], "totals": totals}), flush=True
    )


if __name__ == "__main__":
    main()
