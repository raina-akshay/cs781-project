#!/usr/bin/env python3
"""Run selected LoRMA case classification on raw, unlabeled case JSON."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from tokenizers import Tokenizer
from cjpe.data import SplitCache, TOKENIZER_SHA256
from cjpe.model import build_model
from cjpe.metrics import validate_predictions
from cjpe.runtime import configure_cuda_runtime, predict_split, restore_delta
from prepare_data import load_records


def make_test_cache(path, tokenizer_path):
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Tokenizer differs from the selected foundation")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.truncation is not None or tokenizer.padding is not None:
        raise ValueError("Inference needs the original tokenizer without truncation/padding")
    rows = load_records(path, "test")
    tokens, offsets, records = [], [0], []
    for row in rows:
        ids = tokenizer.encode(row["text"], add_special_tokens=False).ids
        tokens.extend(ids)
        offsets.append(len(tokens))
        records.append(
            {
                "id": row["id"],
                "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                "characters": len(row["text"]),
                "tokens": len(ids),
                "source_row_indices": row["_source_indices"],
                "label": None,
            }
        )
    metadata = {"format": "cs781-p4-raw-case-metadata-v1", "split": "test", "records": records}
    return SplitCache(
        "test", np.asarray(tokens, dtype="<u4"), np.asarray(offsets, dtype="<u8"), metadata
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="JSON list of {id,text} case records"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--reference", action="store_true", help="Explicit CPU reference diagnostics only"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/task.pt")
    parser.add_argument(
        "--tokenizer", type=Path, default=ROOT / "../../assets/foundation/tokenizer.json"
    )
    parser.add_argument("--output", type=Path, default=Path("prediction.json"))
    args = parser.parse_args()
    for key in ("input", "config", "checkpoint", "tokenizer", "output"):
        setattr(args, key, getattr(args, key).resolve())
    os.chdir(ROOT)
    torch.set_num_threads(4)
    config = json.loads(args.config.read_text())
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("config") != config:
        raise ValueError("Checkpoint and configured model differ")
    cache = make_test_cache(args.input, args.tokenizer)
    model = build_model(config, load_foundation=True, initialize_adapters=False)
    restore_delta(model, saved)
    device = torch.device(args.device)
    if device.type == "cuda":
        configure_cuda_runtime(device)
    elif device.type == "cpu" and args.reference:
        torch.set_num_threads(4)
        for module in model.modules():
            if hasattr(module, "backend"):
                module.backend = "reference"
    else:
        raise ValueError("CPU execution requires explicit --reference diagnostic mode")
    model.to(device)
    predictions, _, _ = predict_split(model, cache, config, device)
    validate_predictions(cache, predictions, require_order=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(predictions, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "documents": len(predictions)}))


if __name__ == "__main__":
    main()
