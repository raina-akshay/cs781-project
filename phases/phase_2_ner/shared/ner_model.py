"""Phase 2 contextual-character BIO model using our verified Phase 1 foundation.

The causal backbone is unchanged. A small packed bidirectional character LSTM
provides local right context without exposing labels or changing backbone masks.
No language-model vocabulary logits are materialized during NER.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


PHASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PHASE_ROOT.parents[1] / "assets/foundation/manifest.json"


def _load_foundation_types(manifest):
    sys.path.insert(0, str(Path(manifest["source_root"])))
    from legal_lm.config import ModelConfig
    from legal_lm.model import LegalLanguageModel

    return ModelConfig, LegalLanguageModel


@dataclass
class HeadConfig:
    projection_width: int = 256
    character_embedding_width: int = 32
    bidirectional_lstm_hidden_per_direction: int = 128
    num_bio_labels: int = 25
    dropout: float = 0.1
    character_padding_id: int = 0


class CharacterNER(nn.Module):
    """Shared backbone and character-head layers; restore skips foundation loading."""

    def __init__(
        self,
        model_config=None,
        head_config=None,
        vocab_count=None,
        load_foundation=True,
        foundation_manifest=None,
    ):
        super().__init__()
        path = (
            DEFAULT_MANIFEST
            if foundation_manifest is None
            else Path(foundation_manifest)
        )
        if not path.is_absolute():
            path = PHASE_ROOT / path
        self.foundation_manifest_path = str(path.resolve())
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for field in ("source_root", "checkpoint", "tokenizer_path"):
            if not Path(manifest[field]).is_absolute():
                manifest[field] = str((path.parent / manifest[field]).resolve())
        ModelConfig, LegalLanguageModel = _load_foundation_types(manifest)
        if model_config is None:
            model_config = manifest["model_config"]
        self.model_config = (
            ModelConfig(**model_config)
            if isinstance(model_config, dict)
            else model_config
        )
        if head_config is None:
            head_config = HeadConfig()
        elif isinstance(head_config, dict):
            head_config = HeadConfig(**head_config)
        self.head_config = head_config
        if (
            type(vocab_count) is not int
            or vocab_count <= head_config.character_padding_id
        ):
            raise ValueError("vocab_count must include the character padding ID")
        self.vocab_count = vocab_count
        self.backbone = LegalLanguageModel(self.model_config)
        if load_foundation:
            if asdict(self.model_config) != manifest["model_config"]:
                raise ValueError(
                    "Fresh adaptation must use the exact foundation model configuration"
                )
            checkpoint_path = Path(manifest["checkpoint"])
            saved = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True, mmap=True
            )
            if saved.get("model_config") != manifest["model_config"]:
                raise ValueError(
                    "Checkpoint and foundation manifest configurations disagree"
                )
            self.backbone.load_state_dict(saved["model"], strict=True)
            del saved
        h = head_config
        self.projection = nn.Linear(self.model_config.d_model, h.projection_width)
        self.char_embedding = nn.Embedding(
            vocab_count, h.character_embedding_width, padding_idx=h.character_padding_id
        )
        self.head_dropout = nn.Dropout(h.dropout)
        self.char_lstm = nn.LSTM(
            h.projection_width + h.character_embedding_width,
            h.bidirectional_lstm_hidden_per_direction,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(
            2 * h.bidirectional_lstm_hidden_per_direction, h.num_bio_labels
        )

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def backbone_features(self, input_ids, token_lengths):
        """Causal contextual features; trailing padding cannot affect valid slots.

        We deliberately use the fast dense FLA path. Both the recurrence and
        ordinary attention are causal, so right padding is in the future of
        every real token. No padded token is included in character pooling.
        Left padding or holes are not supported by this path.
        """
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be long [batch, tokens]")
        b, t = input_ids.shape
        if t < 1 or t > self.model_config.context_length:
            raise ValueError("Token length outside foundation context limit")
        if (
            token_lengths.shape != (b,)
            or bool((token_lengths < 1).any())
            or bool((token_lengths > t).any())
        ):
            raise ValueError(
                "token_lengths must identify nonempty right-padded sequences"
            )
        positions = torch.arange(t, device=input_ids.device)[None].expand(b, -1)
        hidden = self.backbone.embedding_dropout(
            self.backbone.token_embedding(input_ids)
        )
        for block in self.backbone.blocks:
            if self.training and self.backbone.gradient_checkpointing:

                def apply_block(value, layer=block):
                    return layer(value, positions, attention_mask=None)[0]

                hidden = checkpoint(apply_block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden, positions, attention_mask=None)[0]
        return self.backbone.final_norm(hidden)
