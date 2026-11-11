"""Load the shared foundation and extract causal mean/EOS features."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_own_foundation(config, load_foundation=True):
    manifest_path = (ROOT / config["foundation_manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_root = (manifest_path.parent / manifest["source_root"]).resolve()
    sys.path.insert(0, str(source_root))
    from legal_lm.config import ModelConfig
    from legal_lm.model import LegalLanguageModel

    model_config = ModelConfig(**config.get("model_config", manifest["model_config"]))
    if load_foundation and asdict(model_config) != manifest["model_config"]:
        raise ValueError("Model configuration differs from the shared foundation")
    seed = config.get("training", {}).get("seed", 42)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        backbone = LegalLanguageModel(model_config)
    if load_foundation:
        path = manifest_path.parent / manifest["checkpoint"]
        if sha256_file(path) != manifest["checkpoint_sha256"]:
            raise ValueError("Foundation checkpoint differs from its manifest")
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        backbone.load_state_dict(saved["model"], strict=True)
    backbone.gradient_checkpointing = config.get("training", {}).get("gradient_checkpointing", True)
    report = {
        "manifest_path": str(manifest_path),
        "original_parameter_count": sum(p.numel() for p in backbone.parameters()),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
    }
    return backbone, model_config, report


def causal_features(backbone, model_config, input_ids, lengths):
    batch, tokens = input_ids.shape
    if tokens > model_config.context_length:
        raise ValueError("Input exceeds the foundation context length")
    positions = torch.arange(tokens, device=input_ids.device)[None].expand(batch, -1)
    hidden = backbone.embedding_dropout(backbone.token_embedding(input_ids))
    for block in backbone.blocks:
        if backbone.training and backbone.gradient_checkpointing:

            def apply_block(value, layer=block):
                return layer(value, positions, attention_mask=None)[0]

            hidden = checkpoint(apply_block, hidden, use_reentrant=False)
        else:
            hidden = block(hidden, positions, attention_mask=None)[0]
    return backbone.final_norm(hidden)


def pool_sentence_features(hidden, lengths):
    """FP32 content mean plus contextual EOS, excluding all trailing padding."""
    batch, tokens, _ = hidden.shape
    positions = torch.arange(tokens, device=hidden.device)[None]
    content = (positions > 0) & (positions < lengths[:, None] - 1)
    value = hidden.float()
    eos = value[torch.arange(batch, device=hidden.device), lengths - 1]
    mean = (
        torch.where(content[..., None], value, 0.0).sum(dim=1) / (lengths - 2).clamp_min(1)[:, None]
    )
    mean = torch.where((lengths > 2)[:, None], mean, eos)
    return torch.cat((mean, eos), dim=-1)
