#!/usr/bin/env python3
"""Generate 128-word continuations with the selected Phase 1 foundation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

import torch

from legal_lm.config import ModelConfig
from legal_lm.generation import (
    generate_continuation,
    stable_seed,
    validate_records,
    validate_predictions,
)
from legal_lm.io import load_json, sha256_file
from legal_lm.model import LegalLanguageModel
from legal_lm.tokenization import load_tokenizer

ROOT = Path(__file__).resolve().parent


def resolve_asset(manifest_path, value):
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def load_model(manifest_path, device):
    manifest = load_json(manifest_path)
    checkpoint = resolve_asset(manifest_path, manifest["checkpoint"])
    if (
        checkpoint.stat().st_size != manifest["checkpoint_bytes"]
        or sha256_file(checkpoint) != manifest["checkpoint_sha256"]
    ):
        raise ValueError("Foundation checkpoint does not match its manifest")
    tokenizer_path = resolve_asset(manifest_path, manifest["tokenizer_path"])
    if sha256_file(tokenizer_path) != manifest["tokenizer_sha256"]:
        raise ValueError("Foundation tokenizer does not match its manifest")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if (
        saved["model_config"] != manifest["model_config"]
        or saved["parameter_count"] != manifest["parameter_count"]
    ):
        raise ValueError("Checkpoint model configuration differs from the manifest")
    model = LegalLanguageModel(ModelConfig(**saved["model_config"]))
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval()
    return model, load_tokenizer(tokenizer_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help='Official JSON mapping IDs to 512-word {"text": ...} records',
    )
    parser.add_argument("--output", default=Path("prediction.json"), type=Path)
    parser.add_argument(
        "--foundation", type=Path, default=ROOT.parents[1] / "assets/foundation/manifest.json"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    records = load_json(args.input)
    validate_records(records, "prompts")
    if any(len(record["text"].split()) != 512 for record in records.values()):
        raise ValueError("Every official Phase 1 prompt must contain exactly 512 words")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The selected native FLA inference path requires a CUDA GPU")
    torch.cuda.set_device(device)
    settings = load_json(ROOT / "configs/generation.json")
    model, tokenizer = load_model(args.foundation.resolve(), device)
    predictions = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for index, (key, record) in enumerate(records.items(), 1):
            text, _ = generate_continuation(
                model,
                tokenizer,
                record["text"],
                target_words=settings["target_words"],
                max_new_tokens=settings["max_new_tokens"],
                strategy=settings["strategy"],
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                seed=stable_seed(settings["seed"], key),
                repetition_penalty=settings["repetition_penalty"],
                no_repeat_ngram_size=settings["no_repeat_ngram_size"],
                max_seconds=settings["max_seconds"],
            )
            predictions[key] = {"text": text}
            print(f"Generated {index}/{len(records)}", flush=True)
    validate_predictions(
        predictions, records, target_words=128, special_tokens=tokenizer.special_tokens
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(predictions, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {len(predictions)} continuations to {args.output}")


if __name__ == "__main__":
    main()
