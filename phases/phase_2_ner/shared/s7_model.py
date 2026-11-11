"""LoRMA adaptation with a contextual character BiLSTM and BIO classifier."""
import hashlib
import json
from pathlib import Path

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .adaptation_targets import parameter_report
from .boundary_model import BoundaryCharacterNER
from .lorma_adaptation import apply_lorma
from .ner_model import DEFAULT_MANIFEST, PHASE_ROOT

TOKENIZER_SHA256 = "25cb054b95fb14d53f12f265c4e09c897ba6dfc5d84b8298ab93004184648b1d"


def _manifest(config):
    path = Path(config.get("foundation_manifest", DEFAULT_MANIFEST))
    if not path.is_absolute():
        path = PHASE_ROOT / path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    tokenizer = path.parent / manifest["tokenizer_path"]
    if hashlib.sha256(tokenizer.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Foundation tokenizer differs from the NER tokenizer")
    return path.resolve()


def _forward_with_features(model, batch):
    """Pool contextual tokens into characters, then apply the packed BiLSTM."""
    input_ids, char_ids = batch["input_ids"], batch["char_ids"]
    char_lengths = batch["char_lengths"].detach().to(device="cpu", dtype=torch.long)
    if char_ids.ndim != 2 or char_ids.dtype != torch.long:
        raise ValueError("char_ids must be long [batch, characters]")
    b, c = char_ids.shape
    if b != input_ids.shape[0] or char_lengths.shape != (b,):
        raise ValueError("Token and character batch sizes disagree")
    if c < 1 or bool((char_lengths < 1).any()) or bool((char_lengths > c).any()):
        raise ValueError("char_lengths must identify nonempty right-padded sequences")
    if "attention_mask" in batch:
        lengths = batch["token_lengths"].to(input_ids.device)
        expected = (
            torch.arange(input_ids.shape[1], device=input_ids.device)[None]
            < lengths[:, None]
        )
        if not torch.equal(batch["attention_mask"], expected):
            raise ValueError("NER backbones accept only right-padded token batches")
    hidden = model.backbone_features(input_ids, batch["token_lengths"])
    with torch.autocast(device_type=input_ids.device.type, enabled=False):
        token_features = model.projection(hidden.float()).reshape(
            -1, model.head_config.projection_width
        )
        token_index, char_index = batch["edge_token_index"], batch["edge_char_index"]
        if (
            token_index.ndim != 1
            or token_index.shape != char_index.shape
            or token_index.numel() < 1
        ):
            raise ValueError(
                "Character-token alignment must contain nonempty paired flat indices"
            )
        character_features = token_features.new_zeros(
            b * c, model.head_config.projection_width
        )
        character_features.index_add_(
            0, char_index, token_features.index_select(0, token_index)
        )
        character_features = character_features.reshape(b, c, -1)
        counts = batch["edge_counts"].to(character_features.dtype)
        if counts.shape != (b, c):
            raise ValueError("edge_counts must match the padded character batch")
        character_features = character_features / counts.clamp_min(1)[..., None]
        head_input = torch.cat(
            (character_features, model.char_embedding(char_ids)), dim=-1
        )
        packed = pack_padded_sequence(
            model.head_dropout(head_input),
            char_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = model.char_lstm(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=c)
        features = model.head_dropout(encoded)
        logits = model.classifier(features)
        boundary_logits = model.boundary_head(features)
    return {"logits": logits, "boundary_logits": boundary_logits, "features": features}


class S7LoRMACharacterNER(BoundaryCharacterNER):
    """LoRMA backbone with character BIO and boundary heads."""

    def forward(self, batch):
        return _forward_with_features(self, batch)


def build_model(config, char_vocab, labels, load_foundation=True):
    """Construct the selected LoRMA character model."""
    manifest_path = _manifest(config)
    head = dict(config.get("head", {}))
    expected = 1 + 2 * len(labels)
    if head.get("num_bio_labels", expected) != expected:
        raise ValueError("BIO head size disagrees with the official label list")
    head["num_bio_labels"] = expected
    if isinstance(char_vocab, dict):
        mapping = char_vocab.get("char_to_id", char_vocab)
        count = max(mapping.values()) + 1 if mapping else 0
    else:
        count = len(char_vocab)
    model = S7LoRMACharacterNER(
        config.get("model_config"),
        head,
        count,
        load_foundation,
        manifest_path,
        boundary_seed=config["training"]["seed"],
    )
    settings = dict(config.get("adaptation", {}))
    rank, alpha = settings.pop("rank", 16), settings.pop("alpha", 16.0)
    seed = settings.pop("seed", config["training"]["seed"])
    report = apply_lorma(
        model.backbone, rank, alpha, seed=seed, initialize=load_foundation
    )
    model.adaptation_report = {
        **report,
        **parameter_report(model),
        "auxiliary_boundary_parameters": sum(
            p.numel() for p in model.boundary_head.parameters()
        ),
        "boundary_channels": ["start_character", "last_entity_character"],
        "boundary_logits_used_for_decoding": False,
    }
    return model
