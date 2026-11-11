"""LoRMA+ output-side multiplicative adapters for our pretrained legal model.

Equation 8 of Bihany et al., Findings ACL 2025:
    W' = (I + (alpha / rank) B A) W0.

The identity-inflation initialization matches the authors' NLU implementation
at commit ffb3c1de29a7f2ab880dc504a2e473b5a4b63292: B=0, A=Kaiming uniform.
Our activation ordering avoids rebuilding a dense adapted weight every step.
The formula is independently implemented and tested against the explicit weight.
See the repository THIRD_PARTY_NOTICES.md for paper and code attribution.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from .adaptation_targets import replace_submodule, target_linears


class LoRMAPlusLinear(nn.Module):
    """Frozen linear map plus an identity-inflated output-space transform.

    Bias remains the original untransformed bias. ``initialize=False`` allocates
    the checkpoint schema with zero factors, for immediate strict restoration;
    it is not a separate initialization proposal for fresh training.
    """

    def __init__(self, original_linear, rank, alpha, initialize=True, generator=None):
        super().__init__()
        if not isinstance(original_linear, nn.Linear):
            raise TypeError("LoRMA+ wraps an ordinary nn.Linear")
        if type(rank) is not int or not 1 <= rank <= min(
            original_linear.in_features, original_linear.out_features
        ):
            raise ValueError(
                "rank must be a positive integer no larger than either linear dimension"
            )
        if (
            not isinstance(alpha, (float, int))
            or isinstance(alpha, bool)
            or not math.isfinite(alpha)
            or alpha <= 0
        ):
            raise ValueError("alpha must be finite and positive")
        self.base_layer = original_linear
        self.base_layer.requires_grad_(False)
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.lorma_A = nn.Parameter(
            original_linear.weight.new_zeros(rank, self.out_features)
        )
        self.lorma_B = nn.Parameter(
            original_linear.weight.new_zeros(self.out_features, rank)
        )
        if initialize:
            nn.init.kaiming_uniform_(self.lorma_A, a=math.sqrt(5), generator=generator)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        # W0x is bias-free because the adaptation transforms weights, not biases.
        projected = F.linear(x, self.base_layer.weight, None)
        update = F.linear(F.linear(projected, self.lorma_A), self.lorma_B)
        result = projected + self.scaling * update
        if self.base_layer.bias is not None:
            result = result + self.base_layer.bias.to(result.dtype)
        return result


def apply_lorma(backbone, rank, alpha, seed=42, initialize=True):
    """Freeze the backbone and insert the exact shared block-projection targets.

    Character-head parameters are untouched because callers pass only the
    backbone. Explicit local generators preserve the caller's head/dropout RNG.
    This function does not load any weights, data, tokenizer or external model.
    """
    targets = target_linears(backbone)
    backbone.requires_grad_(False)
    generators = {}
    report_targets = []
    frozen = sum(p.numel() for p in backbone.parameters())
    for name, original in targets:
        generator = None
        if initialize:
            device_key = str(original.weight.device)
            if device_key not in generators:
                generators[device_key] = torch.Generator(
                    device=original.weight.device
                ).manual_seed(seed)
            generator = generators[device_key]
        adapted = LoRMAPlusLinear(original, rank, alpha, initialize, generator)
        replace_submodule(backbone, name, adapted)
        report_targets.append(
            {
                "name": name,
                "in_features": original.in_features,
                "out_features": original.out_features,
                "rank": rank,
                "alpha": float(alpha),
                "scaling": adapted.scaling,
                "trainable_parameters": 2 * rank * original.out_features,
            }
        )
    adapters = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    return {
        "method": "lorma_plus",
        "variant": "output_side_identity_inflation",
        "formula": "W0 + (alpha/rank) * B @ (A @ W0)",
        "rank": rank,
        "alpha": float(alpha),
        "scaling": float(alpha) / rank,
        "seed": seed,
        "initialized_for_training": initialize,
        "initialization": "B=zeros,A=kaiming_uniform"
        if initialize
        else "zero_schema_for_strict_restore",
        "adapter_dropout": 0.0,
        "target_count": len(report_targets),
        "frozen_backbone_parameters": frozen,
        "adapter_parameters": adapters,
        "targets": report_targets,
    }
