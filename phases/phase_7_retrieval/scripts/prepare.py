#!/usr/bin/env python3
"""Prepare deterministic PCR caches and CPU-only TRAIN/DEV BM25 diagnostics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

SOLUTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLUTION))

import numpy as np

from pcr.data import load_split, prepare_cache, read_json, sha256, write_json
from pcr.evaluation import evaluate_ranking, predictions_from_ranking, rank_scores
from pcr.lexical import TrainBM25


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-root", type=Path, default=SOLUTION)
    parser.add_argument("--output", type=Path, default=SOLUTION / "artifacts/prepared")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=SOLUTION.parents[1] / "assets/foundation/tokenizer.json",
    )
    parser.add_argument("--chunk-length", type=int, default=512)
    parser.add_argument("--max-chunks", type=int, default=16)
    parser.add_argument("--skip-chunks", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw_root = args.phase_root / "data/raw"
    cache = None
    if not args.skip_chunks:
        cache = prepare_cache(
            raw_root,
            args.tokenizer,
            args.output / "chunks",
            args.chunk_length,
            args.max_chunks,
        )
    lexical = args.output / "lexical"
    if (lexical / "manifest.json").exists():
        manifest = read_json(lexical / "manifest.json")
        for relative, digest in manifest["files"].items():
            if sha256(lexical / relative) != digest:
                raise ValueError(f"Lexical artifact hash mismatch: {relative}")
        print("Existing lexical diagnostic cache verified", flush=True)
    else:
        if lexical.exists() and any(lexical.iterdir()):
            raise FileExistsError(
                f"Incomplete lexical cache: inspect before rebuilding {lexical}"
            )
        lexical.mkdir(parents=True, exist_ok=True)
        train = load_split(raw_root, "train")
        print("Fitting BM25 on TRAIN candidate texts only", flush=True)
        bm25 = TrainBM25.fit(train.candidate_texts)
        bm25.save(lexical / "model")
        results = {}
        for split in ("train", "dev"):
            data = train if split == "train" else load_split(raw_root, split)
            print(f"Scoring complete {split} lexical pool", flush=True)
            scores = bm25.score(data.query_texts, data.candidate_texts)
            ranking = rank_scores(scores, data.query_ids, data.candidate_ids)
            np.save(lexical / f"{split}-scores.npy", scores, allow_pickle=False)
            np.save(lexical / f"{split}-ranking.npy", ranking, allow_pickle=False)
            write_json(
                lexical / f"{split}-ids.json",
                {"query_ids": data.query_ids, "candidate_ids": data.candidate_ids},
            )
            predictions = predictions_from_ranking(
                ranking, data.query_ids, data.candidate_ids
            )
            write_json(lexical / f"{split}-prediction.json", predictions)
            results[split] = evaluate_ranking(
                ranking, data.query_ids, data.candidate_ids, data.labels
            )
            print(f"{split} metrics: {results[split]}", flush=True)
        write_json(lexical / "metrics.json", results)
        files = {
            str(p.relative_to(lexical)): sha256(p)
            for p in sorted(lexical.rglob("*"))
            if p.is_file()
        }
        write_json(
            lexical / "manifest.json",
            {
                "files": files,
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "raw_manifest_sha256": sha256(raw_root / "source-manifest.json"),
                "policy": "TRAIN-fitted BM25 statistics; complete same-split inference pools; self exclusion",
                "test_predictions_created": False,
                "trained_neural_model": False,
            },
        )
    write_json(
        args.output / "prepared.json",
        {
            "chunk_cache": str(cache) if cache else None,
            "lexical_directory": str(lexical),
            "test_submission_created": False,
        },
    )
    print(f"Prepared: {args.output / 'prepared.json'}", flush=True)


if __name__ == "__main__":
    main()
