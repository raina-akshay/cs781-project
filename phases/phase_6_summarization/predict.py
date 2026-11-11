"""Inference with the selected LoRMA weights and fitted lexical selector."""
import argparse
import copy
import json
import os
from pathlib import Path
import zipfile

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import joblib
import torch

from summ.common import sha256, validate_predictions
from summ.data import Formatter, own_tokenizer
from summ.foundation import read_manifest
from summ.generation import generate_batch
from summ.model import build_model
from summ.runtime import restore_delta, configure_cuda, FORMAT
from summ.selection import select_evidence

ROOT = Path(__file__).resolve().parent


def load_model(
    weights=ROOT / "weights/model.pt",
    foundation=ROOT / "../../assets/foundation/manifest.json",
    device="cuda",
    reference=False,
):
    weights, foundation = Path(weights).resolve(), Path(foundation).resolve()
    metadata = json.loads((weights.parent / "metadata.json").read_text())
    if sha256(weights) != metadata["weights_sha256"]:
        raise ValueError("Selected task weights differ from their metadata")
    saved = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    if (
        saved["format"] != "cs781-portable-phase6-v1"
        or saved["epoch"] != 16
        or saved["config"] != metadata["config"]
    ):
        raise ValueError("Expected the selected Phase 6 epoch 16 export")
    manifest = read_manifest(foundation)
    if saved["original_foundation_sha256"] != manifest["original_checkpoint_sha256"]:
        raise ValueError("Task adapter and foundation do not belong to the same model")
    config = copy.deepcopy(saved["config"])
    config["foundation_manifest"] = str(foundation)
    model = build_model(config)
    # The portable format contains only learned tensors and the frozen-weight
    # digest. Reuse the original exact tensor and frozen-foundation checks.
    restore_delta(model, {**saved, "format": FORMAT})
    if torch.device(device).type == "cpu":
        if not reference:
            raise ValueError("CPU diagnostics require --reference; scored inference used CUDA/FLA")
        for module in model.modules():
            if hasattr(module, "backend"):
                module.backend = "reference"
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
        configure_cuda()
    model.to(device).eval()
    return model, config, metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="JSON list of {id, document}")
    p.add_argument("--output", type=Path, required=True, help="New prediction JSON file")
    p.add_argument("--weights", type=Path, default=ROOT / "weights/model.pt")
    p.add_argument(
        "--foundation", type=Path, default=ROOT / "../../assets/foundation/manifest.json"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--reference", action="store_true", help="Explicit slow CPU diagnostic backend")
    p.add_argument("--archive", type=Path, help="Optional new ZIP with prediction.json at its root")
    args = p.parse_args()
    if args.batch_size < 1 or args.output.exists() or (args.archive and args.archive.exists()):
        raise ValueError("Use a positive batch size and new output paths")
    rows = json.loads(args.input.read_text())
    if (
        not isinstance(rows, list)
        or not rows
        or any(
            not isinstance(r, dict)
            or set(r) != {"id", "document"}
            or not isinstance(r["id"], str)
            or not isinstance(r["document"], str)
            or not r["document"].strip()
            for r in rows
        )
    ):
        raise ValueError("Input must contain nonempty documents with string IDs")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("IDs must be unique")
    torch.set_num_threads(4)
    model, config, metadata = load_model(args.weights, args.foundation, args.device, args.reference)
    selector_path = args.weights.resolve().parent / "selector.joblib"
    if sha256(selector_path) != metadata["selector_sha256"]:
        raise ValueError("Fitted TRAIN selector differs")
    selector = joblib.load(selector_path)
    tokenizer = own_tokenizer(args.foundation.resolve())
    formatter = Formatter(tokenizer, config["sequence"])
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            sources = [
                select_evidence(
                    tokenizer.encode(r["document"]),
                    tokenizer,
                    selector,
                    budget=config["sequence"]["source_tokens"],
                )[0]
                for r in batch
            ]
            generated = generate_batch(model, sources, formatter, config["generation"], args.device)
            predictions.extend(
                {"id": r["id"], "summary": g["summary"]} for r, g in zip(batch, generated)
            )
    validate_predictions(predictions, ids)
    data = json.dumps(predictions, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(data)
    if args.archive:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.archive, "x", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("prediction.json", data)


if __name__ == "__main__":
    main()
