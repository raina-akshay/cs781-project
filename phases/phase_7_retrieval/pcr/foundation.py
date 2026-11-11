"""Load the shared pretrained model and tokenizer used for case retrieval."""

import hashlib
import json
from pathlib import Path
import sys

import torch

PHASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PHASE_ROOT.parents[1] / "assets/foundation/manifest.json"
TOKENIZER_SHA256 = "25cb054b95fb14d53f12f265c4e09c897ba6dfc5d84b8298ab93004184648b1d"
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_own_foundation(config):
    manifest_path = Path(config.get("foundation_manifest", DEFAULT_MANIFEST))
    if not manifest_path.is_absolute():
        manifest_path = PHASE_ROOT / manifest_path
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("source_root", "checkpoint", "tokenizer_path"):
        value = Path(manifest[key])
        if not value.is_absolute():
            manifest[key] = str((manifest_path.parent / value).resolve())
    if sha256_file(manifest["tokenizer_path"]) != TOKENIZER_SHA256:
        raise ValueError("The tokenizer differs from the pretrained model tokenizer")

    sys.path.insert(0, manifest["source_root"])
    from legal_lm.config import ModelConfig
    from legal_lm.model import LegalLanguageModel

    model_config = ModelConfig(**manifest["model_config"])
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["training"]["seed"])
        backbone = LegalLanguageModel(model_config)
    checkpoint = Path(manifest["checkpoint"])
    if sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Foundation checkpoint does not match its manifest")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if saved["model_config"] != manifest["model_config"]:
        raise ValueError("Checkpoint and model configuration differ")
    backbone.load_state_dict(saved["model"], strict=True)
    backbone.gradient_checkpointing = config["training"]["block_gradient_checkpointing"]
    report = {
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "tokenizer_sha256": TOKENIZER_SHA256,
        "parameter_count": sum(
            parameter.numel() for parameter in backbone.parameters()
        ),
    }
    return backbone, model_config, report
