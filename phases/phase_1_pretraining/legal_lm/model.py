"""Scratch Gated DeltaNet/full-attention hybrid with explicit mixed caches."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .gated_delta import GatedDeltaMixer


@dataclass(frozen=True)
class AttentionCache:
    key: Tensor
    value: Tensor


@dataclass(frozen=True)
class ModelCache:
    layers: tuple
    position: int
    # None denotes a dense, entirely valid history. Otherwise [batch, position].
    attention_mask: Tensor | None = None


@dataclass
class ModelOutput:
    logits: Tensor
    past_key_values: ModelCache | None = None


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, hidden):
        value = hidden.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * self.weight.float()).to(hidden.dtype)


def apply_rotary(x, positions, theta):
    """RoPE with explicit per-example logical token positions; no wrapping."""
    half = x.shape[-1] // 2
    inverse = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    phase = positions.float()[..., None] * inverse
    cosine, sine = phase.cos()[:, None], phase.sin()[:, None]
    left, right = x.float().chunk(2, dim=-1)
    return torch.cat((left * cosine - right * sine, right * cosine + left * sine), dim=-1).to(
        x.dtype
    )


class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.rope_theta = config.rope_theta
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        hidden,
        positions,
        cache=None,
        use_cache=False,
        attention_mask=None,
        previous_mask=None,
    ):
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).reshape(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = (qkv[:, :, index].transpose(1, 2) for index in range(3))
        query, key = apply_rotary(query, positions, self.rope_theta), apply_rotary(
            key, positions, self.rope_theta
        )
        past_length = 0 if cache is None else cache.key.shape[2]
        if cache is not None:
            key, value = torch.cat((cache.key, key), dim=2), torch.cat((cache.value, value), dim=2)
        mask = None
        causal = past_length == 0
        if attention_mask is not None or previous_mask is not None or (past_length and length > 1):
            queries = past_length + torch.arange(length, device=hidden.device)
            keys = torch.arange(past_length + length, device=hidden.device)
            mask = keys[None, :] <= queries[:, None]
            mask = mask[None, None].expand(batch, 1, length, -1)
            if attention_mask is not None or previous_mask is not None:
                old = (
                    torch.ones(batch, past_length, dtype=torch.bool, device=hidden.device)
                    if previous_mask is None
                    else previous_mask
                )
                new = (
                    torch.ones(batch, length, dtype=torch.bool, device=hidden.device)
                    if attention_mask is None
                    else attention_mask
                )
                mask = mask & torch.cat((old, new), dim=1)[:, None, None, :]
            causal = False
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        output = self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        if attention_mask is not None:
            output = output * attention_mask[..., None]
        return output, AttentionCache(key, value) if use_cache else None


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, hidden):
        return self.down(F.silu(self.gate(hidden)) * self.up(hidden))


class HybridBlock(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.is_recurrent = config.architecture == "gated_delta_hybrid" and index % 2 == 0
        self.norm1 = RMSNorm(config.d_model, config.layer_norm_eps)
        self.norm2 = RMSNorm(config.d_model, config.layer_norm_eps)
        self.mixer = GatedDeltaMixer(config) if self.is_recurrent else CausalAttention(config)
        self.feed_forward = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden,
        positions,
        cache=None,
        use_cache=False,
        attention_mask=None,
        previous_mask=None,
    ):
        if self.is_recurrent:
            mixed, present = self.mixer(self.norm1(hidden), cache, use_cache, attention_mask)
        else:
            mixed, present = self.mixer(
                self.norm1(hidden), positions, cache, use_cache, attention_mask, previous_mask
            )
        hidden = hidden + self.dropout(mixed)
        hidden = hidden + self.dropout(self.feed_forward(self.norm2(hidden)))
        if attention_mask is not None:
            hidden = hidden * attention_mask[..., None]
        return hidden, present


class LegalLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(HybridBlock(config, index) for index in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.layer_norm_eps)
        self.apply(self._initialize_module)
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            output = block.mixer.o_proj if block.is_recurrent else block.mixer.proj
            nn.init.normal_(output.weight, std=residual_std)
            nn.init.normal_(block.feed_forward.down.weight, std=residual_std)

    @staticmethod
    def _initialize_module(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def count_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        input_ids,
        past_key_values=None,
        use_cache=False,
        last_token_only=False,
        attention_mask=None,
    ):
        if (
            self.training
            and self.gradient_checkpointing
            and (use_cache or past_key_values is not None)
        ):
            raise ValueError("Gradient checkpointing during training is incompatible with caches")
        batch, length = input_ids.shape
        past_length = 0 if past_key_values is None else past_key_values.position
        old_mask = None if past_key_values is None else past_key_values.attention_mask
        if past_length + length > self.config.context_length:
            raise ValueError(
                f"Sequence requires {past_length + length} positions, exceeding context_length={self.config.context_length}; no truncation or wrapping"
            )
        if attention_mask is not None:
            if not bool(attention_mask.any(dim=1).all()):
                raise ValueError("Each input row must contain at least one unmasked token")
        if old_mask is None and attention_mask is None:
            positions = torch.arange(past_length, past_length + length, device=input_ids.device)[
                None
            ].expand(batch, -1)
            combined_mask = None
        else:
            previous = (
                torch.ones(batch, past_length, dtype=torch.bool, device=input_ids.device)
                if old_mask is None
                else old_mask
            )
            current = (
                torch.ones_like(input_ids, dtype=torch.bool)
                if attention_mask is None
                else attention_mask
            )
            positions = (previous.sum(1)[:, None] + current.long().cumsum(1) - 1).clamp_min(0)
            combined_mask = torch.cat((previous, current), dim=1) if use_cache else None
        hidden = self.embedding_dropout(self.token_embedding(input_ids))
        if attention_mask is not None:
            hidden = hidden * attention_mask[..., None]
        presents = []
        for index, block in enumerate(self.blocks):
            state = None if past_key_values is None else past_key_values.layers[index]
            if self.training and self.gradient_checkpointing:

                def block_forward(value, layer=block):
                    return layer(value, positions, attention_mask=attention_mask)[0]

                hidden = checkpoint(block_forward, hidden, use_reentrant=False)
                present = None
            else:
                hidden, present = block(
                    hidden, positions, state, use_cache, attention_mask, old_mask
                )
            if use_cache:
                presents.append(present)
        hidden = self.final_norm(hidden)
        if last_token_only:
            if attention_mask is not None:
                slots = torch.arange(length, device=input_ids.device)[None].expand(batch, -1)
                last = slots.masked_fill(~attention_mask, -1).amax(1)
                hidden = hidden[torch.arange(batch, device=input_ids.device), last][:, None]
            else:
                hidden = hidden[:, -1:, :]
        logits = F.linear(hidden, self.token_embedding.weight)
        present = (
            ModelCache(tuple(presents), past_length + length, combined_mask) if use_cache else None
        )
        return ModelOutput(logits, present)
