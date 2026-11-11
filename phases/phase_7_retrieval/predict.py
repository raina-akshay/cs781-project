#!/usr/bin/env python3
"""Generate seven prior-case IDs per query using the selected epoch-8 model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import torch
from tokenizers import Tokenizer

from pcr.data import ChunkStore, RawSplit, _write_view, read_json, sha256
from pcr.evaluation import official_evaluator, predictions_from_ranking, rank_scores
from pcr.inference import encode_pool, score_pool, validate_identity
from pcr.model import build_model
from pcr.runtime import configure_numerics, restore_trainable

ROOT = Path(__file__).resolve().parent


def read_documents(path):
    records = read_json(path)
    if not isinstance(records, dict) or not records:
        raise ValueError("Expected a nonempty mapping of case IDs to text records")
    texts = []
    for case_id, record in records.items():
        if (
            not isinstance(case_id, str)
            or len(case_id) != 10
            or not case_id.isdecimal()
        ):
            raise ValueError("Case IDs must retain their original ten-digit strings")
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("text"), list)
            or any(not isinstance(sentence, str) for sentence in record["text"])
        ):
            raise ValueError("Each case must contain a list of text sentences")
        text = "\n".join(record["text"])
        if not text.strip():
            raise ValueError("Empty case text: " + case_id)
        texts.append(text)
    return list(records), texts


def load_selected_model(manifest_path, weights_path, device):
    metadata = read_json(weights_path.with_name("metadata.json"))
    if sha256(weights_path) != metadata["sha256"]:
        raise ValueError("Selected task weights do not match metadata.json")
    saved = torch.load(weights_path, map_location="cpu", weights_only=True)
    if (
        saved["format"] != "cs781-phase7-partial-tuning-v1"
        or saved["epoch"] != 8
        or saved["selection"]["dense_weight"] != 1.0
    ):
        raise ValueError("Expected the selected epoch-8 joint-neural checkpoint")
    manifest = read_json(manifest_path)
    original_hash = manifest.get(
        "original_checkpoint_sha256", manifest["checkpoint_sha256"]
    )
    if original_hash != saved["original_foundation_checkpoint_sha256"]:
        raise ValueError(
            "Task weights were trained from a different foundation checkpoint"
        )
    if (
        manifest["tokenizer_sha256"] != saved["tokenizer_sha256"]
        or manifest["model_config"] != saved["model_config"]
    ):
        raise ValueError("Foundation model configuration or tokenizer differs")
    config = saved["config"]
    config["foundation_manifest"] = str(manifest_path)
    model = build_model(config)
    restore_trainable(model, saved["trainable"])
    tokenizer_path = Path(manifest["tokenizer_path"])
    if not tokenizer_path.is_absolute():
        tokenizer_path = manifest_path.parent / tokenizer_path
    return model.to(device), config, tokenizer_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help='Official query JSON mapping IDs to {"text": [...]}',
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Complete candidate JSON mapping; file order is preserved",
    )
    parser.add_argument("--output", type=Path, default=Path("prediction.json"))
    parser.add_argument(
        "--foundation",
        type=Path,
        default=ROOT.parents[1] / "assets/foundation/manifest.json",
    )
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/model.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Parent directory for temporary token and embedding files",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Output already exists: " + str(args.output))
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The selected native FLA inference path requires a CUDA GPU")
    torch.cuda.set_device(device)
    configure_numerics()
    query_ids, query_texts = read_documents(args.queries)
    candidate_ids, candidate_texts = read_documents(args.candidates)
    raw = RawSplit(
        split="test",
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        query_texts=query_texts,
        candidate_texts=candidate_texts,
        labels=None,
    )
    if len(candidate_ids) < 8:
        raise ValueError(
            "At least eight candidates are required for seven nonself results"
        )
    model, config, tokenizer_path = load_selected_model(
        args.foundation.resolve(), args.weights.resolve(), device
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="phase7-predict-", dir=args.work_dir
    ) as name:
        work = Path(name)
        _write_view(
            work / "candidate",
            candidate_ids,
            candidate_texts,
            tokenizer,
            config["representation"]["chunk_length"],
            config["representation"]["max_chunks"],
            1,
        )
        store = ChunkStore(work / "candidate")
        validate_identity(raw, store)
        encode_pool(model, store, config, 0, 1, device, work / "encoded")
        scores = score_pool(raw, store, config, 0, 1, device, work / "encoded")["joint"]
        ranking = rank_scores(scores, query_ids, candidate_ids)
        predictions = predictions_from_ranking(ranking, query_ids, candidate_ids, 7)
    evaluator = official_evaluator()
    errors = evaluator.validate(
        predictions, {q: [] for q in query_ids}, set(candidate_ids)
    )
    if errors or any(q in values for q, values in predictions.items()):
        raise ValueError("Invalid prediction output: " + "; ".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2)
        handle.write("\n")
    print(
        f"Wrote {len(predictions)} queries with seven candidate IDs each to {args.output}"
    )


if __name__ == "__main__":
    main()
