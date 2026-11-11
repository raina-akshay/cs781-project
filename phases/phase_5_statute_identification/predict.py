"""Full-document statute inference with the selected EVA-LoRA checkpoint."""
import argparse
import copy
import json
import os
from pathlib import Path
import zipfile

os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from lsi.data import collate_cases
from lsi.foundation import read_manifest, sha256_file, load_source_types
from lsi.metrics import decode, validate_thresholds
from lsi.model import build_model
from lsi.runtime import restore_delta, configure_cuda_runtime, FORMAT

ROOT = Path(__file__).resolve().parent


def load_model(
    weights=ROOT / "weights/model.pt",
    foundation=ROOT / "../../assets/foundation/manifest.json",
    device="cuda",
    reference=False,
):
    weights, foundation = Path(weights).resolve(), Path(foundation).resolve()
    metadata = json.loads((weights.parent / "metadata.json").read_text())
    if sha256_file(weights) != metadata["weights_sha256"]:
        raise ValueError("Selected task weights differ from their metadata")
    saved = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    if (
        saved["format"] != "cs781-portable-phase5-v1"
        or saved["epoch"] != 10
        or saved["config"] != metadata["config"]
    ):
        raise ValueError("Expected the selected Phase 5 epoch 10 export")
    manifest = read_manifest(foundation)
    if saved["original_foundation_sha256"] != manifest["original_checkpoint_sha256"]:
        raise ValueError("Task adapter and foundation do not belong to the same model")
    config = copy.deepcopy(saved["config"])
    config["foundation_manifest"] = str(foundation)
    # Every trained head tensor and EVA factor is restored below. No TRAIN data
    # or fitting-time EVA basis is required to construct that exact schema.
    model = build_model(config, initialize_adapters=False)
    restore_delta(model, {**saved, "format": FORMAT})
    thresholds = validate_thresholds(saved["thresholds"]).tolist()
    if torch.device(device).type == "cpu":
        if not reference:
            raise ValueError("CPU diagnostics require --reference; scored inference used CUDA/FLA")
        for module in model.modules():
            if hasattr(module, "backend"):
                module.backend = "reference"
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
        configure_cuda_runtime(device)
    model.to(device).eval()
    return model, config, thresholds


class InputCases:
    split = "test"

    def __init__(self, rows, tokenizer):
        self.ids = [r["id"] for r in rows]
        self.encoded = [np.asarray(tokenizer.encode(r["text"]), dtype=np.int64) for r in rows]
        self.lengths = np.asarray([len(x) for x in self.encoded], dtype=np.int64)
        self.records = [{"labels": None} for _ in rows]

    def __len__(self):
        return len(self.ids)

    def tokens(self, index):
        return self.encoded[index]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="JSON list of {id, text}")
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
            or set(r) != {"id", "text"}
            or not isinstance(r["id"], str)
            or not isinstance(r["text"], str)
            for r in rows
        )
    ):
        raise ValueError("Input must be a JSON list with string IDs and texts")
    if len(set(r["id"] for r in rows)) != len(rows):
        raise ValueError("IDs must be unique")
    torch.set_num_threads(4)
    model, config, thresholds = load_model(
        args.weights, args.foundation, args.device, args.reference
    )
    manifest = read_manifest(args.foundation)
    cls, _ = load_source_types(manifest)
    import importlib

    tokenization = importlib.import_module(cls.__module__.rsplit(".", 1)[0] + ".tokenization")
    tokenizer = tokenization.load_tokenizer(manifest["tokenizer_path"])
    cases = InputCases(rows, tokenizer)
    predictions = []
    from contextlib import nullcontext

    with torch.inference_mode():
        for start in range(0, len(cases), args.batch_size):
            batch = collate_cases(
                cases,
                list(range(start, min(start + args.batch_size, len(cases)))),
                config["input"]["chunk_length"],
            )
            moved = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            amp = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if torch.device(args.device).type == "cuda"
                else nullcontext()
            )
            with amp:
                logits = model(moved)["logits"].float().cpu().numpy()
            predictions.extend(decode(batch["ids"], logits, thresholds))
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
