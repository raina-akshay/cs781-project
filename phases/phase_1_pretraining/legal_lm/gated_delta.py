"""Gated DeltaNet mixer with FLA kernels and a PyTorch reference implementation.

The projections, short convolutions, decay parameterization and output gate
follow fla/layers/gated_deltanet.py at commit
3e52d5abb6ad85312846f39c3e946c2e9881a823 (MIT; Songlin Yang et al.).
The convolutions keep an explicit cache. CUDA execution uses FLA; the
reference backend is selected explicitly for CPU calculations.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


FLA_COMMIT = "3e52d5abb6ad85312846f39c3e946c2e9881a823"


@dataclass(frozen=True)
class DeltaCache:
    # FLA's explicit V-first state layout: [batch, heads, value_dim, key_dim].
    recurrent: Tensor
    # Raw projected inputs, oldest first; each [batch, channels, conv_size-1].
    convolutions: tuple[Tensor, Tensor, Tensor]


def reference_gated_delta_rule(
    q, k, v, g, beta, *, initial_state=None, output_final_state=True, normalize_qk=True
):
    """Literal differentiable FP32 oracle; g is log decay, beta is activated.

    Inputs have [B,T,H,K/V] layout, g/beta [B,T,H]. State is [B,H,V,K].
    This Python token loop is intentionally confined to diagnostic tests.
    """
    q0, k0 = q, k
    if normalize_qk:
        # Match the upstream L2 norm: sum of squares plus 1e-6, not clamp(norm).
        q = (q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)).to(q0.dtype)
        k = (k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)).to(k0.dtype)
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    batch, length, heads, keys = q.shape
    values = v.shape[-1]
    state = (
        q.new_zeros(batch, heads, values, keys) if initial_state is None else initial_state.float()
    )
    outputs = []
    for index in range(length):
        key = k[:, index]
        decayed = state * g[:, index].exp()[..., None, None]
        predicted = torch.einsum("bhvk,bhk->bhv", decayed, key)
        correction = (v[:, index] - predicted) * beta[:, index, :, None]
        state = decayed + correction[..., None] * key[..., None, :]
        outputs.append(torch.einsum("bhvk,bhk->bhv", state, q[:, index] / math.sqrt(keys)))
    output = torch.stack(outputs, dim=1).to(q0.dtype)
    return output, state if output_final_state else None


def causal_short_convolution(x, weight, previous=None, *, output_final_state=False):
    """Vectorized causal depthwise convolution; no input/cache mutation."""
    width = weight.shape[-1]
    batch, _, channels = x.shape
    if previous is None:
        previous = x.new_zeros(batch, channels, width - 1)
    joined = torch.cat((previous, x.transpose(1, 2)), dim=-1)
    y = F.conv1d(joined, weight, bias=None, groups=channels).transpose(1, 2)
    final = joined[:, :, -(width - 1) :].clone() if output_final_state and width > 1 else None
    if output_final_state and width == 1:
        final = joined[:, :, :0].clone()
    return F.silu(y), final


class GatedRMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x, gate):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * self.weight.float() * F.silu(gate.float())).to(x.dtype)


class GatedDeltaMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backend = config.gdn_backend
        self.num_heads = config.gdn_num_heads
        self.head_k_dim = config.gdn_head_dim
        self.head_v_dim = config.gdn_value_dim
        self.conv_size = config.conv_size
        self.key_dim = self.num_heads * self.head_k_dim
        self.value_dim = self.num_heads * self.head_v_dim
        self.q_proj = nn.Linear(config.d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.value_dim, bias=False)
        self.a_proj = nn.Linear(config.d_model, self.num_heads, bias=False)
        self.b_proj = nn.Linear(config.d_model, self.num_heads, bias=False)
        # Preserve upstream time-scale initialization. Clamp the possible exact
        # zero endpoint before log so even an unlucky random draw stays finite.
        a = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(a.clamp_min(torch.finfo(torch.float32).tiny).log())
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp_min(1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True
        self.q_conv1d = nn.Conv1d(
            self.key_dim, self.key_dim, self.conv_size, groups=self.key_dim, bias=False
        )
        self.k_conv1d = nn.Conv1d(
            self.key_dim, self.key_dim, self.conv_size, groups=self.key_dim, bias=False
        )
        self.v_conv1d = nn.Conv1d(
            self.value_dim, self.value_dim, self.conv_size, groups=self.value_dim, bias=False
        )
        self.g_proj = nn.Linear(config.d_model, self.value_dim, bias=False)
        self.o_norm = GatedRMSNorm(self.head_v_dim, config.layer_norm_eps)
        self.o_proj = nn.Linear(self.value_dim, config.d_model, bias=False)

    def _dense(self, hidden, cache, use_cache):
        batch, length, _ = hidden.shape
        old_convs = (None, None, None) if cache is None else cache.convolutions
        projected = (self.q_proj(hidden), self.k_proj(hidden), self.v_proj(hidden))
        pairs = [
            causal_short_convolution(x, module.weight, previous, output_final_state=use_cache)
            for x, module, previous in zip(
                projected, (self.q_conv1d, self.k_conv1d, self.v_conv1d), old_convs
            )
        ]
        q, k = (
            pairs[index][0].reshape(batch, length, self.num_heads, self.head_k_dim)
            for index in (0, 1)
        )
        v = pairs[2][0].reshape(batch, length, self.num_heads, self.head_v_dim)
        # Compute log decay in FP32. Activating beta in projection precision
        # matches upstream's activated-beta interface to the FLA kernels.
        g = -self.A_log.float().exp() * F.softplus(
            self.a_proj(hidden).float() + self.dt_bias.float()
        )
        beta_logits = self.b_proj(hidden)
        beta = beta_logits.float().sigmoid().to(beta_logits.dtype)
        state = None if cache is None else cache.recurrent
        if self.backend == "reference":
            output, final = reference_gated_delta_rule(
                q, k, v, g, beta, initial_state=state, output_final_state=use_cache
            )
        else:
            from fla.ops.gated_delta_rule import (
                chunk_gated_delta_rule,
                fused_recurrent_gated_delta_rule,
            )

            operation = (
                fused_recurrent_gated_delta_rule
                if not torch.is_grad_enabled() and not self.training and length <= 64
                else chunk_gated_delta_rule
            )
            output, final = operation(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                state_v_first=True,
            )
        gate = self.g_proj(hidden).reshape(batch, length, self.num_heads, self.head_v_dim)
        output = self.o_proj(self.o_norm(output, gate).reshape(batch, length, self.value_dim))
        present = DeltaCache(final, tuple(pair[1] for pair in pairs)) if use_cache else None
        return output, present

    def forward(self, hidden, cache=None, use_cache=False, attention_mask=None):
        if attention_mask is None:
            return self._dense(hidden, cache, use_cache)
        # Explicit padding masks compress each row before convolution/recurrence.
        # Thus padding never decays memory or advances convolution history.
        # Production training uses dense causal document windows without caches;
        # production generation is one unpadded prompt per rank. This batched
        # padding path favors a simple correct state contract over extra kernels.
        rows, states = [], []
        for index in range(hidden.shape[0]):
            valid = attention_mask[index].nonzero().flatten()
            old = (
                None
                if cache is None
                else DeltaCache(
                    cache.recurrent[index : index + 1],
                    tuple(state[index : index + 1] for state in cache.convolutions),
                )
            )
            value, state = self._dense(
                hidden[index : index + 1].index_select(1, valid), old, use_cache
            )
            rows.append(
                hidden.new_zeros(1, hidden.shape[1], hidden.shape[2]).index_copy(
                    1, valid, value.to(hidden.dtype)
                )
            )
            states.append(state)
        present = None
        if use_cache:
            present = DeltaCache(
                torch.cat([state.recurrent for state in states]),
                tuple(torch.cat([state.convolutions[j] for state in states]) for j in range(3)),
            )
        return torch.cat(rows), present
