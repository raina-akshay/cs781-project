#!/usr/bin/env python3
"""Predict legal entity spans with the selected direct-span model."""
import argparse
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from shared.ner_contract import load_json, validate_texts
from shared.ner_data import make_windows, NERDataset
from shared.s7_model import build_model
from shared.s7_predict import evaluate_model
from shared.s7_runtime import sha256_file

ROOT = Path(__file__).resolve().parent


def load_selected_model(foundation, weights, device="cpu"):
    foundation, weights = Path(foundation).resolve(), Path(weights).resolve()
    metadata = load_json(weights.with_name("metadata.json"))
    if sha256_file(weights) != metadata["sha256"]:
        raise ValueError("Task weight checksum mismatch")
    saved = torch.load(weights, map_location="cpu", weights_only=True)
    if saved["format"] != "cs781-ner-direct-span-v1" or saved["epoch"] != 26:
        raise ValueError("Expected the selected twenty-six-epoch refit")
    manifest = load_json(foundation)
    if (
        manifest["original_checkpoint_sha256"]
        != saved["original_foundation_checkpoint_sha256"]
    ):
        raise ValueError("Task weights require the original selected GDN foundation")
    if manifest["tokenizer_sha256"] != saved["tokenizer_sha256"]:
        raise ValueError("Tokenizer identity mismatch")
    base = foundation.parent / manifest["checkpoint"]
    if sha256_file(base) != manifest["checkpoint_sha256"]:
        raise ValueError("Foundation weight checksum mismatch")
    config = saved["config"]
    config["foundation_manifest"] = str(foundation)
    model = build_model(
        config, saved["char_vocab"], saved["labels"], load_foundation=True
    )
    trained = {k for k, p in model.named_parameters() if p.requires_grad}
    buffers = {k for k, _ in model.named_buffers()}
    if trained != set(saved["trainable_names"]) or buffers != set(
        saved["buffer_names"]
    ):
        raise ValueError("Selected task parameter schema changed")
    if set(saved["trainable_state"]) != trained | buffers:
        raise ValueError("Task weights have missing or unexpected tensors")
    full = model.state_dict()
    for key, value in saved["trainable_state"].items():
        if full[key].shape != value.shape or full[key].dtype != value.dtype:
            raise ValueError("Task tensor shape or dtype mismatch: " + key)
        full[key] = value
    model.load_state_dict(full, strict=True)
    tokenizer_path = (foundation.parent / manifest["tokenizer_path"]).resolve()
    return model.to(device).eval(), saved, tokenizer_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON mapping document IDs to original text strings",
    )
    parser.add_argument("--output", type=Path, default=Path("prediction.json"))
    parser.add_argument(
        "--foundation",
        type=Path,
        default=ROOT.parents[1] / "assets/foundation/manifest.json",
    )
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/model.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Native FLA inference requires a CUDA GPU")
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected inference recipe requires BF16 support")
    torch.set_num_threads(2)
    texts = load_json(args.input)
    validate_texts(texts)
    model, saved, tokenizer_path = load_selected_model(
        args.foundation, args.weights, device
    )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.encode_special_tokens = True
    windows = []
    for did, text in texts.items():
        windows.extend(
            make_windows(
                did,
                text,
                tokenizer,
                saved["char_vocab"],
                saved["labels"],
                window_tokens=1024,
                overlap_tokens=256,
            )
        )
    payload = {
        "partitions": {"test": list(texts)},
        "texts": texts,
        "windows": windows,
        "labels": saved["labels"],
        "char_vocab": saved["char_vocab"],
        "manifest": {},
    }
    dataset = NERDataset(payload, "test")
    result, _ = evaluate_model(
        model,
        dataset,
        device,
        0,
        1,
        None,
        args.batch_size,
        prediction_policy="leading_whitespace_only",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result["prediction"], handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote entity spans for {len(texts)} documents to {args.output}")


if __name__ == "__main__":
    main()
