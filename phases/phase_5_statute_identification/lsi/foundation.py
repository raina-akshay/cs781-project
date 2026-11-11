"""Load the shared foundation and expose its causal hidden states."""

import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.utils.checkpoint import checkpoint


PHASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PHASE_ROOT / "../../assets/foundation/manifest.json"
TOKENIZER_SHA256 = "25cb054b95fb14d53f12f265c4e09c897ba6dfc5d84b8298ab93004184648b1d"
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2


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


def validate_chunk_inputs(input_ids, lengths, model_config):
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 2
        or input_ids.dtype != torch.long
        or min(input_ids.shape) < 1
    ):
        raise ValueError("Chunk input_ids must be nonempty long [chunks,tokens]")
    batch, tokens = input_ids.shape
    if not 2 <= tokens <= model_config.context_length:
        raise ValueError("Chunk token width exceeds the original foundation context")
    if (
        not isinstance(lengths, torch.Tensor)
        or lengths.dtype != torch.long
        or lengths.shape != (batch,)
        or lengths.device != input_ids.device
    ):
        raise ValueError("Chunk lengths must be matching device-local long values")
    if bool(((lengths < 2) | (lengths > tokens)).any()):
        raise ValueError("Chunk lengths must include valid BOS/EOS")
    positions = torch.arange(tokens, device=input_ids.device)[None]
    content = (positions > 0) & (positions < lengths[:, None] - 1)
    padding = positions >= lengths[:, None]
    eos = input_ids.gather(1, (lengths - 1)[:, None]).squeeze(1)
    invalid = (
        (input_ids < 0)
        | (input_ids >= model_config.vocab_size)
        | (content & (input_ids <= EOS_ID))
        | (padding & (input_ids != PAD_ID))
    )
    if bool(invalid.any() | (input_ids[:, 0] != BOS_ID).any() | (eos != EOS_ID).any()):
        raise ValueError("Chunks must be [BOS, ordinary content, EOS] with trailing PAD only")


def causal_features(backbone, model_config, input_ids, lengths):
    # This guard is essential: using the native blocks directly must not bypass
    # LegalLanguageModel.forward's pretrained positional/context bound.
    validate_chunk_inputs(input_ids, lengths, model_config)
    batch, tokens = input_ids.shape
    positions = torch.arange(tokens, device=input_ids.device)[None].expand(batch, -1)
    hidden = backbone.embedding_dropout(backbone.token_embedding(input_ids))
    for block in backbone.blocks:
        if backbone.training and backbone.gradient_checkpointing:

            def apply_block(value, layer=block):
                return layer(value, positions, attention_mask=None)[0]

            hidden = checkpoint(apply_block, hidden, use_reentrant=False, preserve_rng_state=True)
        else:
            hidden = block(hidden, positions, attention_mask=None)[0]
    return backbone.final_norm(hidden)


def pool_chunk_features(hidden, lengths):
    """FP32 content mean + valid EOS; a BOS/EOS-only chunk uses EOS twice."""
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
