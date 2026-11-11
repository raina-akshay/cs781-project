"""Native hidden-state execution and target-only, case-weighted causal CE."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .adaptation import apply_lorma
from .foundation import load_own_foundation


def hidden_states(backbone, input_ids):
    if (
        input_ids.ndim != 2
        or input_ids.dtype != torch.long
        or not 0 < input_ids.shape[1] <= backbone.config.context_length
    ):
        raise ValueError("Invalid physical native input context")
    batch, width = input_ids.shape
    positions = torch.arange(width, device=input_ids.device)[None].expand(batch, -1)
    hidden = backbone.embedding_dropout(backbone.token_embedding(input_ids))
    for block in backbone.blocks:
        if backbone.training and backbone.gradient_checkpointing:

            def apply(value, layer=block):
                return layer(value, positions, attention_mask=None)[0]

            hidden = checkpoint(apply, hidden, use_reentrant=False, preserve_rng_state=True)
        else:
            hidden = block(hidden, positions, attention_mask=None)[0]
    # Right padding cannot influence earlier valid causal states. Padded states
    # are never supervised; cached generation separately requires a proper mask.
    return backbone.final_norm(hidden)


class Summarizer(nn.Module):
    def __init__(self, backbone, foundation_report, adaptation_report, ce_chunk_tokens=128):
        super().__init__()
        self.backbone = backbone
        self.foundation_report = foundation_report
        self.adaptation_report = adaptation_report
        self.ce_chunk_tokens = ce_chunk_tokens

    def forward(self, input_ids, labels, case_weights, lengths):
        mask = labels != -100
        hidden = hidden_states(self.backbone, input_ids)
        selected = hidden[mask]
        targets = labels[mask]
        weights = case_weights[:, None].expand_as(labels)[mask]
        total = selected.new_zeros((), dtype=torch.float32)
        for start in range(0, len(selected), self.ce_chunk_tokens):

            def projection_ce(value, target, weight):
                logits = F.linear(value, self.backbone.token_embedding.weight)
                losses = F.cross_entropy(logits.float(), target, reduction="none")
                return (losses * weight).sum()

            args = (
                selected[start : start + self.ce_chunk_tokens],
                targets[start : start + self.ce_chunk_tokens],
                weights[start : start + self.ce_chunk_tokens],
            )
            if self.training and torch.is_grad_enabled():
                loss = checkpoint(
                    projection_ce, *args, use_reentrant=False, preserve_rng_state=True
                )
            else:
                loss = projection_ce(*args)
            total = total + loss
        return total


def build_model(config, load_foundation=True):
    backbone, settings, report = load_own_foundation(config, load_foundation)
    a = config["adaptation"]
    adaptation = apply_lorma(backbone, a["rank"], a["alpha"], config["training"]["seed"])
    model = Summarizer(backbone, report, adaptation, config["training"]["ce_chunk_tokens"])
    return model
