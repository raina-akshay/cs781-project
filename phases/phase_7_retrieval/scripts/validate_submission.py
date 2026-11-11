#!/usr/bin/env python3
"""Validate the required query IDs and seven-candidate output format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcr.evaluation import official_evaluator


def validate_artifact(path: Path, query_ids: set[str], candidate_ids: set[str]):
    evaluator = official_evaluator()
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            if archive.namelist() != ["prediction.json"]:
                raise ValueError(
                    "Submission ZIP must contain only prediction.json at its root"
                )
            payload = archive.read("prediction.json")
    else:
        payload = path.read_bytes()
    predictions = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=evaluator.pairs_no_duplicates,
        parse_constant=evaluator.reject_constant,
    )
    errors = evaluator.validate(predictions, {q: [] for q in query_ids}, candidate_ids)
    if errors:
        raise ValueError("\n".join(errors))
    return {
        "passed": True,
        "query_count": len(predictions),
        "candidate_pool_count": len(candidate_ids),
        "candidate_ids_per_query": 7,
        "self_match_query_ids": sorted(
            q for q, values in predictions.items() if q in values
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    folder = {"train": "Train", "dev": "Dev", "test": "Test"}[args.split]
    evaluator = official_evaluator()
    raw = ROOT / "data/raw" / folder
    queries = evaluator.load_json(raw / f"{args.split}.json")
    candidates = evaluator.load_json(raw / f"{args.split}_candidates.json")
    result = validate_artifact(args.artifact, set(queries), set(candidates))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
