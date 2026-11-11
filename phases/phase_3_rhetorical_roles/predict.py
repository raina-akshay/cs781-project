#!/usr/bin/env python3
"""Predict rhetorical roles from official test JSON without training or dev data."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
import torch
from tokenizers import Tokenizer
from rr.data import LABELS, TOKENIZER_SHA256, sha256_file, strict_json
from rr.metrics import validate_predictions
from rr_context.model import build_model
from rr_context.runtime import restore_delta, predict


def test_documents(paths, tokenizer_path, labels_path, domain=None):
    """Tokenize the public {document_id: [sentence, ...]} test contract."""
    label_map = strict_json(labels_path)
    expected = {str(i): label for i, label in enumerate(LABELS)}
    if label_map != expected:
        raise ValueError("The ordered label map differs from the selected 13-role model")
    if sha256_file(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("Tokenizer differs from the selected foundation")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if (
        tokenizer.get_vocab_size() != 32768
        or tokenizer.truncation is not None
        or tokenizer.padding is not None
    ):
        raise ValueError("Use the original tokenizer without padding or truncation")
    documents, seen = [], set()
    for path in paths:
        source_domain = domain
        if source_domain is None:
            name = path.name.upper()
            source_domain = (
                "CL" if name.startswith("CL_") else "IT" if name.startswith("IT_") else None
            )
        if source_domain not in ("CL", "IT"):
            raise ValueError(
                "Use official CL_/IT_ filenames, or specify --domain for a renamed file"
            )
        raw = strict_json(path)
        if not isinstance(raw, dict) or not raw:
            raise ValueError("Test JSON must map document IDs to nonempty sentence lists")
        for identifier, sentences in raw.items():
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError("Document IDs must be nonempty strings")
            normalized = identifier[:-4] if identifier.endswith(".txt") else identifier
            if not normalized.strip() or normalized in seen:
                raise ValueError("Duplicate or empty document ID after .txt normalization")
            seen.add(normalized)
            if (
                not isinstance(sentences, list)
                or not sentences
                or any(not isinstance(sentence, str) for sentence in sentences)
            ):
                raise ValueError("Each document must contain a nonempty list of sentence strings")
            encoded = tokenizer.encode_batch(sentences, add_special_tokens=False)
            documents.append(
                {
                    "id": identifier,
                    "domain": source_domain,
                    "split": "test",
                    "sentences": [
                        {"tokens": value.ids, "label": None, "index": index}
                        for index, value in enumerate(encoded)
                    ],
                }
            )
    return {"labels": list(LABELS), "documents": documents}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Official CL_test.json and/or IT_test.json",
    )
    parser.add_argument(
        "--domain", choices=("CL", "IT"), help="Domain for a renamed single-domain input file"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/task.pt")
    parser.add_argument(
        "--tokenizer", type=Path, default=ROOT / "../../assets/foundation/tokenizer.json"
    )
    parser.add_argument("--output", type=Path, default=Path("prediction.json"))
    args = parser.parse_args()
    args.input = [path.resolve() for path in args.input]
    for key in ("config", "checkpoint", "tokenizer", "output"):
        setattr(args, key, getattr(args, key).resolve())
    os.chdir(ROOT)
    config = strict_json(args.config)
    cache = test_documents(args.input, args.tokenizer, ROOT / "configs/label-map.json", args.domain)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("config") != config:
        raise ValueError("Checkpoint configuration differs from the configured model")
    model = build_model(config, load_foundation=True, initialize_adapters=False)
    restore_delta(model, saved)
    device = torch.device("cuda")
    model.to(device)
    result, _ = predict(model, cache, "test", config, device)
    summary = validate_predictions(cache, result, "test")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "documents": summary["documents"],
                "sentences": summary["sentences"],
            }
        )
    )


if __name__ == "__main__":
    main()
