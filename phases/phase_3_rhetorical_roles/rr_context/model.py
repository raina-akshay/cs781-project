"""Causal sentence features with a local bidirectional context head."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .adaptation import apply_rs_lora
from .foundation import causal_features, load_own_foundation, pool_sentence_features


class HierarchicalHead(nn.Module):
    def __init__(self, model_width, hidden_size=256, dropout=0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(2 * model_width)
        self.projection = nn.Linear(2 * model_width, hidden_size)
        self.activation = nn.GELU()
        self.projection_dropout = nn.Dropout(dropout)
        self.context_lstm = nn.LSTM(
            hidden_size, hidden_size // 2, num_layers=1, batch_first=True, bidirectional=True
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 13)
        self.hidden_size = hidden_size

    def forward(self, pooled, sentence_map, sentence_lengths):
        projected = self.projection_dropout(
            self.activation(self.projection(self.input_norm(pooled)))
        )
        batch, width = len(sentence_lengths), int(sentence_lengths.max())
        flat_index = sentence_map[:, 0] * width + sentence_map[:, 1]
        dense = projected.new_zeros(batch * width, self.hidden_size).index_copy(
            0, flat_index, projected
        )
        dense = dense.reshape(batch, width, self.hidden_size)
        packed = pack_padded_sequence(
            dense,
            sentence_lengths.detach().to(device="cpu", dtype=torch.long),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.context_lstm(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=width)
        features = self.output_norm(encoded)
        logits = self.classifier(self.output_dropout(features))
        return {"features": features, "logits": logits}


class ContextRhetoricalRoleClassifier(nn.Module):
    def __init__(self, config, load_foundation=True, initialize_adapters=True):
        super().__init__()
        context = config["context"]
        if context["mode"] != "local" or config["adaptation"]["method"] != "rs_lora":
            raise ValueError("This model uses local sentence context and rsLoRA")
        if config.get("prototype", {}).get("lambda", 0) != 0:
            raise ValueError("The selected model does not use prototype regularization")
        self.maximum_sentences = context["core_sentences"] + 2 * context["halo_sentences"]
        self.maximum_length = config["input"]["max_length"]
        self.backbone, self.model_config, self.foundation_report = load_own_foundation(
            config, load_foundation
        )
        self.foundation_manifest_path = self.foundation_report["manifest_path"]
        adaptation = config["adaptation"]
        seed = config["training"]["seed"]
        self.adaptation_report = apply_rs_lora(
            self.backbone,
            rank=adaptation["rank"],
            alpha=adaptation["alpha"],
            seed=seed,
            initialize=initialize_adapters,
        )
        head = config["head"]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.head = HierarchicalHead(
                self.model_config.d_model, head["hidden_size"], float(head["dropout"])
            )
        total = sum(parameter.numel() for parameter in self.parameters())
        trained = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        backbone_trained = sum(
            parameter.numel() for parameter in self.backbone.parameters() if parameter.requires_grad
        )
        head_trained = sum(
            parameter.numel() for parameter in self.head.parameters() if parameter.requires_grad
        )
        self.parameter_report = {
            "parameters": total,
            "trainable_parameters": trained,
            "trainable_backbone_adapter_parameters": backbone_trained,
            "trainable_head_parameters": head_trained,
            "trainable_prototype_parameters": 0,
            "frozen_parameters": total - trained,
            "trainable_fraction": trained / total,
        }

    def count_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def head_parameters(self):
        yield from self.head.parameters()

    def backbone_features(self, input_ids, lengths):
        if input_ids.shape[1] > self.maximum_length:
            raise ValueError("Input sentence width exceeds its declared cap")
        return causal_features(self.backbone, self.model_config, input_ids, lengths)

    def forward(self, batch):
        input_ids, lengths = batch["input_ids"], batch["lengths"]
        hidden = self.backbone_features(input_ids, lengths)
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            pooled = pool_sentence_features(hidden, lengths)
            return self.head(pooled, batch["sentence_map"], batch["sentence_lengths"])


def build_model(config, load_foundation=True, initialize_adapters=True):
    return ContextRhetoricalRoleClassifier(config, load_foundation, initialize_adapters)
