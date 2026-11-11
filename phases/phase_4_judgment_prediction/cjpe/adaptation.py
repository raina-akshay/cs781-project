"""Faithful factorized evaluation of the author's LoRMA+ identity inflation.

Author NLU mode='pre', do_rri=True at commit
ffb3c1de29a7f2ab880dc504a2e473b5a4b63292 computes
W' = W0 + (alpha/r) B (A W0), adding the original bias afterwards.
We preserve that exact algebra while associating products over activations to
avoid a dense weight reconstruction on every chunk. Floating-point association
can differ; no full-rank delta claim is made (rank(delta W) <= r).
Upstream attribution and the license declaration accompany this module.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


MIXER_TARGETS = frozenset({"q_proj", "k_proj", "v_proj", "g_proj", "o_proj", "qkv", "proj"})
MLP_TARGETS = frozenset({"gate", "up", "down"})


def target_linears(backbone):
    result = []
    for name, module in backbone.named_modules():
        parts = name.split(".")
        if len(parts) != 4 or parts[0] != "blocks" or not parts[1].isdigit():
            continue
        selected = (
            parts[2] == "mixer"
            and parts[3] in MIXER_TARGETS
            or parts[2] == "feed_forward"
            and parts[3] in MLP_TARGETS
        )
        if selected and isinstance(module, nn.Linear):
            result.append((name, module))
    if not result:
        raise ValueError("No compatible original block projection targets found")
    return result


class LoRMAPlusLinear(nn.Module):
    def __init__(self, original_linear, rank, alpha, initialize=True, generator=None):
        super().__init__()
        self.base_layer = original_linear
        self.base_layer.requires_grad_(False)
        self.in_features, self.out_features = (
            original_linear.in_features,
            original_linear.out_features,
        )
        self.rank, self.alpha = rank, float(alpha)
        self.scaling = self.alpha / rank
        self.lorma_A = nn.Parameter(original_linear.weight.new_zeros(rank, self.out_features))
        self.lorma_B = nn.Parameter(original_linear.weight.new_zeros(self.out_features, rank))
        if initialize:
            nn.init.kaiming_uniform_(self.lorma_A, a=math.sqrt(5), generator=generator)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, value):
        projected = F.linear(value, self.base_layer.weight, None)
        update = F.linear(F.linear(projected, self.lorma_A), self.lorma_B)
        result = projected + self.scaling * update
        if self.base_layer.bias is not None:
            result = result + self.base_layer.bias.to(result.dtype)
        return result


def apply_lorma(backbone, rank, alpha, seed=42, initialize=True):
    targets = target_linears(backbone)
    backbone.requires_grad_(False)
    frozen = sum(parameter.numel() for parameter in backbone.parameters())
    generators = {}
    for name, original in targets:
        generator = None
        if initialize:
            key = str(original.weight.device)
            if key not in generators:
                generators[key] = torch.Generator(device=original.weight.device).manual_seed(seed)
            generator = generators[key]
        layer = LoRMAPlusLinear(original, rank, alpha, initialize, generator)
        parent, leaf = name.rsplit(".", 1)
        setattr(backbone.get_submodule(parent), leaf, layer)
    adapters = sum(
        parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad
    )
    return {
        "method": "lorma_plus",
        "rank": rank,
        "alpha": float(alpha),
        "scaling": float(alpha) / rank,
        "adapter_parameters": adapters,
        "target_count": len(targets),
        "frozen_backbone_parameters": frozen,
    }
