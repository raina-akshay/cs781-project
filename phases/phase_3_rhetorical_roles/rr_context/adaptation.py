"""Rank-stabilized LoRA: W0 + alpha/sqrt(rank) * B @ A."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from rr.adaptation_targets import replace_submodule, target_linears


class RsLoRALinear(nn.Module):
    """Frozen base plus additive rank-stabilized update, without adapter dropout.

    ``initialize=False`` supplies zero factors for immediate strict checkpoint
    restoration; it is not a fresh-training initialization.  Local generators
    prevent adapter initialization from changing the caller's head/dropout RNG.
    """

    def __init__(self, original_linear, rank, alpha, initialize=True, generator=None):
        super().__init__()
        self.base_layer = original_linear
        self.base_layer.requires_grad_(False)
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / math.sqrt(rank)
        self.lora_A = nn.Parameter(original_linear.weight.new_zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(original_linear.weight.new_zeros(self.out_features, rank))
        if initialize:
            if generator is None:
                generator = torch.Generator(device=original_linear.weight.device).manual_seed(0)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=generator)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        baseline = F.linear(x, self.base_layer.weight, self.base_layer.bias)
        update = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return baseline + self.scaling * update


def apply_rs_lora(backbone, rank=16, alpha=4.0, seed=42, initialize=True):
    """Freeze the supplied backbone and adapt exactly the shared projection set."""
    targets = target_linears(backbone)
    frozen = sum(p.numel() for p in backbone.parameters())
    backbone.requires_grad_(False)
    generators = {}
    for name, original in targets:
        device = str(original.weight.device)
        if device not in generators and initialize:
            generators[device] = torch.Generator(device=original.weight.device).manual_seed(seed)
        adapter = RsLoRALinear(original, rank, alpha, initialize, generators.get(device))
        replace_submodule(backbone, name, adapter)
    count = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    return {
        "method": "rs_lora",
        "rank": rank,
        "alpha": float(alpha),
        "scaling": float(alpha) / math.sqrt(rank),
        "adapter_parameters": count,
        "target_count": len(targets),
        "frozen_backbone_parameters": frozen,
    }
