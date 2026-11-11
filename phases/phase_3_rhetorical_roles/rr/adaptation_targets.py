"""Select the backbone projections adapted by rsLoRA."""
from torch import nn

MIXER_TARGETS = frozenset({"q_proj", "k_proj", "v_proj", "g_proj", "o_proj", "qkv", "proj"})
MLP_TARGETS = frozenset({"gate", "up", "down"})


def target_linears(backbone):
    """Select block projections, excluding recurrence time/erase gates explicitly."""
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
        raise ValueError("No compatible backbone projection targets found")
    return result


def replace_submodule(backbone, name, replacement):
    parent, leaf = name.rsplit(".", 1)
    setattr(backbone.get_submodule(parent), leaf, replacement)
