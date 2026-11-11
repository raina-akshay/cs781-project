"""AdamW grouping for native partial fine-tuning, preserving mixer exemptions."""


import torch


def build_optimizer(model, config):
    """Return fresh AdamW; all trainable layers share the declared learning rate.

    The original recurrent mixer explicitly exempts its decay/time parameters
    from weight decay. Norms, biases and all other scalar/vector parameters are
    likewise exempt. Group order and member order follow named_parameters so
    strict checkpoint recovery has a stable optimizer schema.
    """
    settings = config["training"]
    learning_rate, weight_decay = settings["learning_rate"], settings["weight_decay"]
    buckets = {"decay": [], "no_decay": []}
    names = {"decay": [], "no_decay": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        key = (
            "decay"
            if parameter.ndim >= 2 and not getattr(parameter, "_no_weight_decay", False)
            else "no_decay"
        )
        buckets[key].append(parameter)
        names[key].append(name)
    groups, report = [], []
    for key in ("decay", "no_decay"):
        if not buckets[key]:
            continue
        decay = float(weight_decay) if key == "decay" else 0.0
        groups.append(
            {
                "params": buckets[key],
                "weight_decay": decay,
                "group_name": key,
                "parameter_names": names[key],
            }
        )
        report.append(
            {
                "group_name": key,
                "weight_decay": decay,
                "tensor_count": len(buckets[key]),
                "parameter_count": sum(p.numel() for p in buckets[key]),
                "parameter_names": names[key],
            }
        )
    optimizer = torch.optim.AdamW(groups, lr=float(learning_rate))
    optimizer.pcr_group_report = report
    return optimizer
