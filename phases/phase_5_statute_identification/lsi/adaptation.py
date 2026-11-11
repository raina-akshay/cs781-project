"""Additive LoRA adapters initialized from TRAIN activation PCA."""

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


class EVALoRALinear(nn.Module):
    """W0 x + (alpha/r) B A x; the original bias is added exactly once.

    Initialize from a fitted PCA basis or restore trained factors before use.
    """

    def __init__(self, original_linear, rank, alpha, basis=None):
        super().__init__()
        if type(rank) is not int or not 1 <= rank <= min(
            original_linear.in_features, original_linear.out_features
        ):
            raise ValueError("rank must be positive and fit both projection dimensions")
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (float, int))
            or not math.isfinite(alpha)
            or alpha <= 0
        ):
            raise ValueError("alpha must be finite and positive")
        self.base_layer = original_linear.requires_grad_(False)
        self.in_features, self.out_features = (
            original_linear.in_features,
            original_linear.out_features,
        )
        self.rank, self.alpha, self.scaling = rank, float(alpha), float(alpha) / rank
        self.lora_A = nn.Parameter(original_linear.weight.new_zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(original_linear.weight.new_zeros(self.out_features, rank))
        self._eva_weights_ready = False
        if basis is not None:
            self.install_basis(basis)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    @torch.no_grad()
    def install_basis(self, basis):
        if (
            not isinstance(basis, torch.Tensor)
            or not basis.is_floating_point()
            or basis.shape != self.lora_A.shape
            or not bool(torch.isfinite(basis).all())
        ):
            raise ValueError("EVA basis must be finite floating [rank,in_features]")
        # The published non-whitened initializer has orthonormal rows. Test in
        # FP64 after copy-independent conversion, before casting master weights.
        value = basis.detach().to(device="cpu", dtype=torch.float64)
        if not torch.allclose(
            value @ value.T, torch.eye(self.rank, dtype=torch.float64), atol=2e-4, rtol=2e-4
        ):
            raise ValueError("EVA basis rows must be orthonormal (whitening is disabled)")
        self.lora_A.copy_(basis.to(device=self.lora_A.device, dtype=self.lora_A.dtype))
        self.lora_B.zero_()
        self._eva_weights_ready = True

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        names = [prefix + "lora_A", prefix + "lora_B"]
        if all(name in state_dict for name in names):
            self._eva_weights_ready = all(
                isinstance(state_dict[name], torch.Tensor)
                and state_dict[name].shape == getattr(self, name.rsplit(".", 1)[-1]).shape
                and bool(torch.isfinite(state_dict[name]).all())
                for name in names
            )

    def forward(self, value):
        if not self._eva_weights_ready:
            raise RuntimeError(
                "EVA schema requires a verified basis or strict trained-state restoration before forward"
            )
        return F.linear(value, self.weight, self.bias) + self.scaling * F.linear(
            F.linear(value, self.lora_A), self.lora_B
        )


def mark_eva_weights_ready(backbone):
    """Call only after the caller has strictly restored the complete task delta."""
    layers = [module for module in backbone.modules() if isinstance(module, EVALoRALinear)]
    if not layers:
        raise ValueError("No EVA LoRA adapters were restored")
    for module in layers:
        if (
            module.lora_A.shape != (module.rank, module.in_features)
            or module.lora_B.shape != (module.out_features, module.rank)
            or not bool(torch.isfinite(module.lora_A).all() & torch.isfinite(module.lora_B).all())
        ):
            raise ValueError("Restored EVA adapter shapes/values are invalid")
    for module in layers:
        module._eva_weights_ready = True


def apply_eva_lora(
    backbone,
    rank,
    alpha,
    seed=42,
    initialize=True,
    rank_pattern=None,
    *,
    initialization_path=None,
    initialization_sha256=None,
    basis=None,
    expected_foundation_sha256=None
):
    """Install additive LoRA with exact realized EVA ranks and constant scaling.

    Training supplies a fitted basis; inference restores the complete saved factors.
    """
    if (
        type(rank) is not int
        or rank < 1
        or isinstance(alpha, bool)
        or not isinstance(alpha, (float, int))
        or not math.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("Nominal rank and alpha must be positive")
    targets = target_linears(backbone)
    names = {name for name, _ in targets}
    artifact = None
    if initialization_path is not None:
        if basis is not None:
            raise ValueError("Initialization file cannot be combined with another basis source")
        from .eva import load_eva_initialization

        artifact = load_eva_initialization(
            initialization_path,
            expected_sha256=initialization_sha256,
            expected_foundation_sha256=expected_foundation_sha256,
        )
        if artifact["rank"] != rank or artifact["alpha"] != float(alpha):
            raise ValueError("EVA initializer nominal rank/alpha differ from configuration")
        if rank_pattern is not None and rank_pattern != artifact["rank_pattern"]:
            raise ValueError("Configured EVA rank pattern differs from the initialization file")
        rank_pattern, basis = artifact["rank_pattern"], artifact["components"]
        expected_shapes = {
            name: {"in_features": layer.in_features, "out_features": layer.out_features}
            for name, layer in targets
        }
        if artifact["target_shapes"] != expected_shapes:
            raise ValueError(
                "EVA initialization projection closure differs from the own foundation"
            )
    pattern = dict(rank_pattern) if rank_pattern is not None else {name: rank for name in names}
    if set(pattern) != names or any(
        type(value) is not int or value < 0 for value in pattern.values()
    ):
        raise ValueError("EVA rank pattern must cover exactly every original target")
    if sum(pattern.values()) != rank * len(targets):
        raise ValueError("EVA realized ranks must preserve the nominal total rank budget")
    if basis is not None and set(basis) != names:
        raise ValueError("EVA basis must include every target, including empty zero-rank bases")
    for name, layer in targets:
        if rank > min(layer.in_features, layer.out_features) or pattern[name] > min(
            layer.in_features, layer.out_features
        ):
            raise ValueError("EVA rank exceeds a projection dimension: " + name)
        if basis is not None and (
            not isinstance(basis[name], torch.Tensor)
            or basis[name].shape != (pattern[name], layer.in_features)
            or not basis[name].is_floating_point()
            or not bool(torch.isfinite(basis[name]).all())
        ):
            raise ValueError("EVA basis shape/values differ: " + name)
    backbone.requires_grad_(False)
    frozen = sum(parameter.numel() for parameter in backbone.parameters())
    scale, report = float(alpha) / rank, []
    for name, original in targets:
        current_rank = pattern[name]
        if current_rank:
            layer = EVALoRALinear(
                original,
                current_rank,
                scale * current_rank,
                basis=None if basis is None else basis[name],
            )
            parent, leaf = name.rsplit(".", 1)
            setattr(backbone.get_submodule(parent), leaf, layer)
        report.append(
            {
                "name": name,
                "in_features": original.in_features,
                "out_features": original.out_features,
                "rank": current_rank,
                "alpha": scale * current_rank,
                "scaling": scale if current_rank else None,
                "trainable_parameters": current_rank
                * (original.in_features + original.out_features),
            }
        )
    adapters = sum(
        parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad
    )
    return {
        "method": "eva_lora",
        "variant": "TRAIN_activation_PCA_additive_LoRA",
        "formula": "W0 + (alpha/base_rank) * B @ A",
        "rank": rank,
        "alpha": float(alpha),
        "scaling": scale,
        "seed": seed,
        "rank_pattern": pattern,
        "rank_budget": sum(pattern.values()),
        "adapter_dropout": 0.0,
        "initialization": "TRAIN_EVA_basis" if basis is not None else "restored_checkpoint",
        "initialized_for_training": basis is not None,
        "initialization_path": str(initialization_path)
        if initialization_path is not None
        else None,
        "initialization_sha256": initialization_sha256,
        "target_count": len(report),
        "active_target_count": sum(pattern[name] > 0 for name in names),
        "frozen_backbone_parameters": frozen,
        "adapter_parameters": adapters,
        "targets": report,
    }
