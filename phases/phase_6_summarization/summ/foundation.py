"""Load the shared foundation and expose its causal hidden states."""

import hashlib
import json
from pathlib import Path
import sys

import torch


PHASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PHASE_ROOT / "../../assets/foundation/manifest.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key in ("checkpoint", "tokenizer_path", "source_root"):
        value = Path(manifest[key])
        manifest[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    return manifest


def load_source_types(manifest):
    sys.path.insert(0, manifest["source_root"])
    from legal_lm.config import ModelConfig
    from legal_lm.model import LegalLanguageModel

    return ModelConfig, LegalLanguageModel


def load_own_foundation(config, load_foundation=True):
    manifest_path = Path(config.get("foundation_manifest", DEFAULT_MANIFEST))
    if not manifest_path.is_absolute():
        manifest_path = PHASE_ROOT / manifest_path
    manifest_path = manifest_path.resolve()
    manifest = read_manifest(manifest_path)
    if sha256_file(manifest["tokenizer_path"]) != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer does not match the foundation")
    ModelConfig, LegalLanguageModel = load_source_types(manifest)
    model_config = ModelConfig(**manifest["model_config"])
    seed = config.get("training", {}).get("seed", 42)
    with torch.random.fork_rng(devices=[]):
        # Keep initialization independent of each rank's CUDA dropout stream.
        torch.random.default_generator.manual_seed(seed)
        backbone = LegalLanguageModel(model_config)
    if load_foundation:
        checkpoint_path = Path(manifest["checkpoint"])
        if sha256_file(checkpoint_path) != manifest["checkpoint_sha256"]:
            raise ValueError("Foundation weights do not match the manifest")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        backbone.load_state_dict(saved["model"], strict=True)
    backbone.gradient_checkpointing = config.get("training", {}).get(
        "block_gradient_checkpointing", False
    )
    report = {
        "family": config.get("foundation_family", "s5_hybrid"),
        "manifest_path": str(manifest_path),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "original_parameter_count": sum(p.numel() for p in backbone.parameters()),
        "foundation_loaded": load_foundation,
    }
    return backbone, model_config, report
