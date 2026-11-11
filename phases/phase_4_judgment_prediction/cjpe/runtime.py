"""Task checkpoints, distributed prediction and training utilities."""
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F

from .foundation import sha256_file


FORMAT = "cs781-phase4-case-delta-v1"
CUDA_RUNTIME_POLICY = {
    "format": "cs781-phase4-cuda-numerics-v1",
    "environment": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "FLA_DISABLE_BACKEND_DISPATCH": "1"},
    "deterministic_algorithms": True,
    "deterministic_warn_only": False,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "matmul_allow_tf32": False,
    "cudnn_allow_tf32": True,
}


def configure_cuda_runtime(device):
    """Apply the actually tested CUDA numerical policy before GPU execution.

    Launchers must supply the environment before Python starts; changing it
    after a CUDA/cuBLAS handle exists is not a supported way to configure a
    run. This helper checks the launcher values, pins Torch's actual flags and
    returns their readback. It never initializes a CUDA device itself. Repeated
    calls in an already configured diagnostic process are safe.
    """
    if torch.device(device).type != "cuda":
        raise ValueError("The CUDA runtime policy applies only to a CUDA execution")
    for key, value in CUDA_RUNTIME_POLICY["environment"].items():
        if os.environ.get(key) != value:
            raise RuntimeError(
                "CUDA launcher must set " + key + "=" + value + " before Python starts"
            )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    return dict(CUDA_RUNTIME_POLICY)


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def append_event(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at_utc": now(), **value}, allow_nan=False) + "\n")
        stream.flush()


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(value, device):
    if set(value) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("Incomplete per-rank RNG state")
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if device.type == "cuda":
        if value["cuda"] is None:
            raise ValueError("CUDA checkpoint RNG is missing")
        torch.cuda.set_rng_state(value["cuda"], device)
    elif value["cuda"] is not None:
        raise ValueError("Cannot restore a CUDA training trajectory on CPU")


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def move_batch(batch, device):
    # Packed sequence lengths/case offsets are intentionally CPU metadata.
    return {
        key: value.to(device, non_blocking=False)
        if torch.is_tensor(value) and key != "case_offsets"
        else value
        for key, value in batch.items()
    }


def unwrap(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def trainable_state(model):
    model = unwrap(model)
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def buffer_state(model):
    return {name: value.detach().cpu().clone() for name, value in unwrap(model).named_buffers()}


def frozen_state_hash(model):
    digest, count = hashlib.sha256(), 0
    for name, parameter in unwrap(model).named_parameters():
        if parameter.requires_grad:
            continue
        value = parameter.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        count += value.numel()
    return {"sha256": digest.hexdigest(), "parameters": count}


def restore_delta(model, checkpoint):
    model = unwrap(model)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Unknown task checkpoint format")
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    saved = checkpoint.get("trainable_state", {})
    buffers = dict(model.named_buffers())
    saved_buffers = checkpoint.get("buffer_state", {})
    if set(saved) != expected or set(saved_buffers) != set(buffers):
        raise ValueError("Task delta omits or adds a trainable tensor/buffer")
    if frozen_state_hash(model) != checkpoint["frozen_state"]:
        raise ValueError("Original frozen foundation differs from the task checkpoint")
    tensors = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in {**saved, **saved_buffers}.items():
            target = tensors[name] if name in saved else buffers[name]
            if (
                not torch.is_tensor(value)
                or value.shape != target.shape
                or value.dtype != target.dtype
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("Invalid task delta tensor: " + name)
            target.copy_(value.to(target.device))
    return model


def build_optimizer(model, config):
    training = config["training"]
    groups = {}
    for name, parameter in unwrap(model).named_parameters():
        if not parameter.requires_grad:
            continue
        head = name.startswith("head.")
        decay = parameter.ndim >= 2 and not name.endswith(".bias")
        key = (head, decay)
        groups.setdefault(key, []).append(parameter)
    return torch.optim.AdamW(
        [
            {
                "params": parameters,
                "lr": training["head_lr"] if head else training["adapter_lr"],
                "initial_lr": training["head_lr"] if head else training["adapter_lr"],
                "weight_decay": training["weight_decay"] if decay else 0.0,
                "group_name": ("head" if head else "adapter")
                + ("_decay" if decay else "_no_decay"),
            }
            for (head, decay), parameters in sorted(groups.items())
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def set_learning_rate(optimizer, step, total_steps, warmup_fraction, minimum_lr_ratio):
    warmup_steps = int(total_steps * warmup_fraction)
    if warmup_steps and step <= warmup_steps:
        factor = step / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        factor = minimum_lr_ratio + (1 - minimum_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor
    return factor


def _distributed(rank, world_size):
    if type(rank) is not int or type(world_size) is not int or not 0 <= rank < world_size:
        raise ValueError("Invalid distributed rank/world")
    if world_size > 1 and (
        not dist.is_initialized() or dist.get_rank() != rank or dist.get_world_size() != world_size
    ):
        raise ValueError("Process group differs from declared prediction ranks")


@torch.no_grad()
def predict_split(model, split_cache, config, device, rank=0, world_size=1):
    """Return ordered (prediction rows, logit rows, diagnostics) on rank zero.

    Other ranks return (None,None,None). TEST branches never read gold labels,
    including any accidentally supplied diagnostic metadata label. Every real
    case occurs on precisely one rank and is classified using all its chunks.
    """
    from .data import collate_cases
    from .metrics import validate_predictions

    _distributed(rank, world_size)
    model = unwrap(model)
    was_training = model.training
    model.eval()
    local_rows, loss_sum, cases_with_gold = [], 0.0, 0
    tokens, chunks = 0, 0
    evaluation_batch = config["training"]["eval_batch_cases"]
    indices = list(range(rank, len(split_cache), world_size))
    try:
        for start in range(0, len(indices), evaluation_batch):
            batch = collate_cases(
                split_cache,
                indices[start : start + evaluation_batch],
                config["input"]["chunk_length"],
            )
            moved = move_batch(batch, device)
            with amp(device):
                logits = model(moved)["logits"].float()
            if logits.shape != (len(batch["ids"]), 2) or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Invalid/nonfinite case prediction logits")
            values = logits.cpu().tolist()
            for identifier, value in zip(batch["ids"], values):
                local_rows.append(
                    {"id": identifier, "pred": int(value[1] > value[0]), "logits": value}
                )
            if split_cache.split != "test":
                gold = moved["labels"]
                loss_sum += float(F.cross_entropy(logits, gold, reduction="sum"))
                cases_with_gold += len(gold)
            tokens += int(batch["attention_mask"].sum())
            chunks += len(batch["input_ids"])
    finally:
        model.train(was_training)
    payload = {
        "rows": local_rows,
        "loss_sum": loss_sum,
        "gold_cases": cases_with_gold,
        "valid_tokens": tokens,
        "chunks": chunks,
    }
    if world_size > 1:
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(payload, gathered, dst=0)
    else:
        gathered = [payload]
    if rank != 0:
        return None, None, None
    rows = [row for item in gathered for row in item["rows"]]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(split_cache.ids):
        raise ValueError("Distributed prediction repeated, omitted or invented a case ID")
    ordered = [by_id[identifier] for identifier in split_cache.ids]
    predictions = [{"id": row["id"], "pred": row["pred"]} for row in ordered]
    validate_predictions(split_cache, predictions, require_order=True)
    logit_rows = [{"id": row["id"], "logits": row["logits"]} for row in ordered]
    gold_count = sum(item["gold_cases"] for item in gathered)
    diagnostics = {
        "split": split_cache.split,
        "cases": len(ordered),
        "unweighted_cross_entropy": sum(item["loss_sum"] for item in gathered) / gold_count
        if gold_count
        else None,
        "gold_used_for_loss": split_cache.split != "test",
        "valid_tokens": sum(item["valid_tokens"] for item in gathered),
        "chunks": sum(item["chunks"] for item in gathered),
    }
    return predictions, logit_rows, diagnostics
