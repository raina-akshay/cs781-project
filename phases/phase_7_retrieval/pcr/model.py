"""Encode legal case chunks with the partially tuned causal foundation model.

Each chunk is processed independently. Mean pooling excludes BOS and padding.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .foundation import BOS_ID, PAD_ID, load_own_foundation


def apply_partial_finetuning(backbone, last_n_blocks, train_final_norm=True):
    """Change trainability only; preserve every original parameter and value."""
    total_blocks = len(backbone.blocks)
    if type(last_n_blocks) is not int or not 1 <= last_n_blocks < total_blocks:
        raise ValueError(
            "Partial fine-tuning requires 1 <= last_n_blocks < total blocks"
        )
    backbone.requires_grad_(False)
    first = total_blocks - last_n_blocks
    for block in backbone.blocks[first:]:
        block.requires_grad_(True)
    backbone.final_norm.requires_grad_(train_final_norm)
    # This marker changes only checkpoint placement, never the native forward.
    backbone._pcr_trainable_suffix_start = first
    names = [
        name
        for name, parameter in backbone.named_parameters()
        if parameter.requires_grad
    ]
    trainable = sum(
        parameter.numel()
        for parameter in backbone.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in backbone.parameters())
    return {
        "method": "partial_layerwise",
        "last_n_blocks": last_n_blocks,
        "trainable_block_indices": list(range(first, total_blocks)),
        "train_final_norm": train_final_norm,
        "trainable_parameters": trainable,
        "frozen_backbone_parameters": total - trainable,
        "original_parameter_count": total,
        "new_parameters": 0,
        "trainable_names": names,
        "initialization": "unchanged original Phase 1 weights",
    }


def hidden_states(backbone, input_ids):
    """Execute exactly the original dense causal path without the LM head."""
    if (
        input_ids.ndim != 2
        or input_ids.dtype != torch.long
        or input_ids.shape[0] < 1
        or not 0 < input_ids.shape[1] <= backbone.config.context_length
    ):
        raise ValueError("Expected nonempty long input IDs within the native context")
    batch, width = input_ids.shape
    positions = torch.arange(width, device=input_ids.device)[None].expand(batch, -1)
    hidden = backbone.embedding_dropout(backbone.token_embedding(input_ids))
    for index, block in enumerate(backbone.blocks):
        if (
            backbone.training
            and backbone.gradient_checkpointing
            and torch.is_grad_enabled()
            and index >= getattr(backbone, "_pcr_trainable_suffix_start", 0)
        ):
            # The frozen prefix naturally has no graph. Checkpoint only the
            # trainable suffix; non-reentrant mode handles its detached input.
            def apply(value, layer=block):
                return layer(value, positions, attention_mask=None)[0]

            hidden = checkpoint(
                apply, hidden, use_reentrant=False, preserve_rng_state=True
            )
        else:
            hidden = block(hidden, positions, attention_mask=None)[0]
    return backbone.final_norm(hidden)


def pool_hidden_states(hidden, lengths, pooling="mean"):
    """Pool BOS+content chunks; lengths include BOS and exclude right padding."""
    if hidden.ndim != 3 or lengths.shape != (hidden.shape[0],):
        raise ValueError("Hidden states and lengths have incompatible shapes")
    if (
        lengths.dtype != torch.long
        or lengths.device != hidden.device
        or bool((lengths < 2).any())
        or bool((lengths > hidden.shape[1]).any())
    ):
        raise ValueError("Each chunk must contain BOS and at least one content token")
    if pooling == "mean":
        slots = torch.arange(hidden.shape[1], device=hidden.device)[None]
        content_mask = (slots > 0) & (slots < lengths[:, None])
        vector = (hidden.float() * content_mask[..., None]).sum(1)
        vector = vector / (lengths - 1).float()[:, None]
    elif pooling == "last":
        vector = hidden[
            torch.arange(len(hidden), device=hidden.device), lengths - 1
        ].float()
    else:
        raise ValueError("Pooling must be mean or last")
    return F.normalize(vector, p=2, dim=-1, eps=1e-8)


class RetrievalEncoder(nn.Module):
    """Shared query/candidate encoder; returns normalized FP32 chunk vectors."""

    def __init__(self, backbone, foundation_report, adaptation_report, pooling="mean"):
        super().__init__()
        if pooling not in {"mean", "last"}:
            raise ValueError("Pooling must be mean or last")
        self.backbone = backbone
        self.foundation_report = foundation_report
        self.adaptation_report = adaptation_report
        self.pooling = pooling
        self.output_dim = backbone.config.d_model

    def forward(self, input_ids, lengths):
        if (
            input_ids.ndim != 2
            or input_ids.dtype != torch.long
            or lengths.shape != (input_ids.shape[0],)
            or lengths.dtype != torch.long
            or lengths.device != input_ids.device
        ):
            raise ValueError(
                "Expected long [chunks,tokens] IDs and long [chunks] lengths"
            )
        if (
            input_ids.shape[0] < 1
            or input_ids.shape[1] < 2
            or bool((lengths < 2).any())
            or bool((lengths > input_ids.shape[1]).any())
        ):
            raise ValueError("Every retrieval chunk requires BOS plus nonempty content")
        slots = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        valid = slots < lengths[:, None]
        if (
            not bool((input_ids[:, 0] == BOS_ID).all())
            or bool((input_ids[~valid] != PAD_ID).any())
            or bool((input_ids[valid & (slots > 0)] < 3).any())
            or bool((input_ids >= self.backbone.config.vocab_size).any())
        ):
            raise ValueError("Chunks must be BOS + literal text tokens, then right PAD")
        return pool_hidden_states(
            hidden_states(self.backbone, input_ids), lengths, self.pooling
        )


def build_model(config):
    """Load original Phase-1 weights, then unfreeze the declared native suffix."""
    backbone, settings, foundation_report = load_own_foundation(config)
    # Exact gradient caching replays the encoder after differentiating a large
    # document-level batch. Disable both nn.Dropout and the attention module's
    # numeric SDPA dropout argument so first-pass and replay representations are
    # deterministic. This changes task-time regularization, not Phase-1 weights.
    for module in backbone.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        elif isinstance(getattr(module, "dropout", None), (int, float)):
            module.dropout = 0.0
    adaptation_config = config["adaptation"]
    if adaptation_config.get("method") != "partial_layerwise":
        raise ValueError("Solution 2 implements native partial/layer-wise fine-tuning")
    adaptation_report = apply_partial_finetuning(
        backbone,
        adaptation_config["last_n_blocks"],
        adaptation_config["train_final_norm"],
    )
    model = RetrievalEncoder(
        backbone,
        foundation_report,
        adaptation_report,
        config.get("retrieval", {}).get("pooling", "mean"),
    )
    model.representation_report = {
        "output_dim": settings.d_model,
        "chunk_pooling": model.pooling,
        "exclude_bos": True,
        "exclude_right_padding": True,
        "chunk_l2_normalization": True,
        "projection_head": None,
        "shared_query_candidate_parameters": True,
        "cross_chunk_state": False,
        "causal_backbone": True,
        "original_foundation_dropout": settings.dropout,
        "effective_encoder_dropout": 0.0,
    }
    return model
