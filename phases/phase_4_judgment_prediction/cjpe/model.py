"""Case-level ordered chunk hierarchy over the original own Phase-1 S5.

Every supplied case owns a nonempty ordered run of chunks. All its chunk
representations participate in one BiLSTM/attention-pooled outcome, and only
case labels supervise training. Labels never enter this model's forward path.
"""

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.checkpoint import checkpoint

from .adaptation import apply_lorma
from .foundation import causal_features, load_own_foundation, pool_chunk_features


class CaseHead(nn.Module):
    def __init__(self, model_width, hidden_size=256, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_norm = nn.LayerNorm(2 * model_width)
        self.projection = nn.Linear(2 * model_width, hidden_size)
        self.activation = nn.GELU()
        self.projection_dropout = nn.Dropout(dropout)
        self.context_lstm = nn.LSTM(
            hidden_size, hidden_size // 2, num_layers=1, batch_first=True, bidirectional=True
        )
        self.context_norm = nn.LayerNorm(hidden_size)
        # One shared learned query: no position/label-dependent chunk selection.
        self.attention_query = nn.Parameter(torch.empty(hidden_size))
        nn.init.normal_(self.attention_query, std=1.0 / math.sqrt(hidden_size))
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, pooled, case_offsets):
        counts = (case_offsets[1:] - case_offsets[:-1]).to(device="cpu", dtype=torch.long)
        batch, width = counts.numel(), int(counts.max())
        projected = self.projection_dropout(
            self.activation(self.projection(self.input_norm(pooled)))
        )
        # pad_sequence keeps gradients for every real chunk and original order.
        sequences = list(projected.split(counts.tolist(), dim=0))
        dense = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        packed = pack_padded_sequence(dense, counts, batch_first=True, enforce_sorted=False)
        encoded, _ = self.context_lstm(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=width)
        encoded = self.context_norm(encoded)
        mask = torch.arange(width, device=pooled.device)[None] < counts.to(pooled.device)[:, None]
        scores = (encoded * self.attention_query).sum(dim=-1)
        attention = scores.masked_fill(~mask, -torch.inf).softmax(dim=1)
        features = (attention[..., None] * encoded).sum(dim=1)
        logits = self.classifier(self.output_dropout(features))
        # Boolean flatten visits cases then their valid chunks in original order.
        return {"logits": logits, "chunk_attention": attention[mask], "case_features": features}


class CaseJudgmentClassifier(nn.Module):
    def __init__(self, config, load_foundation=True, initialize_adapters=True):
        super().__init__()
        self.backbone, self.model_config, self.foundation_report = load_own_foundation(
            config, load_foundation
        )
        self.foundation_manifest_path = self.foundation_report["manifest_path"]
        inputs = config.get("input", {})
        self.maximum_length = inputs.get("chunk_length", 1024)
        if (
            type(self.maximum_length) is not int
            or not 2 <= self.maximum_length <= self.model_config.context_length
        ):
            raise ValueError("Declared chunk length must fit the original own model context")
        adaptation = config.get("adaptation", {})
        if (
            set(adaptation) - {"method", "rank", "alpha"}
            or adaptation.get("method") != "lorma_plus"
        ):
            raise ValueError("Solution 1 requires output-side identity-inflated LoRMA+")
        training = config.get("training", {})
        self.encoder_micro_chunks = training.get("encoder_micro_chunks", 1)
        self.encoder_checkpointing = training.get("gradient_checkpointing", True)
        self.adaptation_report = apply_lorma(
            self.backbone,
            rank=adaptation.get("rank", 16),
            alpha=adaptation.get("alpha", 16),
            seed=training.get("seed", 42),
            initialize=initialize_adapters,
        )
        head = config.get("head", {})
        hidden, dropout = head.get("hidden_size", 256), head.get("dropout", 0.1)
        with torch.random.fork_rng(devices=[]):
            # Preserve the caller's per-rank CUDA dropout stream while making
            # the newly initialized CPU head identical on every rank.
            torch.random.default_generator.manual_seed(training.get("seed", 42))
            self.head = CaseHead(self.model_config.d_model, hidden, float(dropout))
        total = self.count_parameters()
        trained = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        head_trained = sum(
            parameter.numel() for parameter in self.head.parameters() if parameter.requires_grad
        )
        self.parameter_report = {
            "parameters": total,
            "trainable_parameters": trained,
            "trainable_head_parameters": head_trained,
            "trainable_backbone_adapter_parameters": trained - head_trained,
            "frozen_parameters": total - trained,
            "trainable_fraction": trained / total,
        }

    def count_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def head_parameters(self):
        yield from self.head.parameters()

    def backbone_features(self, input_ids, lengths):
        if input_ids.ndim != 2 or input_ids.shape[1] > self.maximum_length:
            raise ValueError("Chunk input exceeds the declared input cap")
        return causal_features(self.backbone, self.model_config, input_ids, lengths)

    def _encode_pooled(self, input_ids, lengths):
        hidden = self.backbone_features(input_ids, lengths)
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            return pool_chunk_features(hidden, lengths)

    def encode_chunks(self, input_ids, attention_mask):
        lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
        pooled = []
        for start in range(0, input_ids.shape[0], self.encoder_micro_chunks):
            local_lengths = lengths[start : start + self.encoder_micro_chunks]
            # Do not run needless trailing padding through the entire backbone.
            width = max(2, int(local_lengths.max()))
            local_ids = input_ids[start : start + self.encoder_micro_chunks, :width]
            if self.training and torch.is_grad_enabled() and self.encoder_checkpointing:
                value = checkpoint(
                    self._encode_pooled,
                    local_ids,
                    local_lengths,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                value = self._encode_pooled(local_ids, local_lengths)
            pooled.append(value)
        return torch.cat(pooled, dim=0)

    def forward(self, batch):
        input_ids = batch["input_ids"]
        offsets = batch["case_offsets"].detach().to(device="cpu")
        pooled = self.encode_chunks(input_ids, batch["attention_mask"])
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            return self.head(pooled.float(), offsets)


def build_model(config, load_foundation=True, initialize_adapters=True):
    return CaseJudgmentClassifier(config, load_foundation, initialize_adapters)
