"""Deterministic chunk recomputation and explicit summed-gradient distribution.

Physical encoder microbatches never change the query-balanced objective. The
encoder has no stochastic dropout; both passes use training-mode native kernels.
"""
from __future__ import annotations
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import torch.distributed as dist


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=path.name + ".", delete=False
    ) as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
        tmp = f.name
    os.replace(tmp, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    os.close(fd)
    try:
        torch.save(value, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for x in iter(lambda: f.read(2**20), b""):
            h.update(x)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_numerics():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def setup_distributed():
    configure_numerics()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    if world > 1:
        from datetime import timedelta

        dist.init_process_group("nccl", timeout=timedelta(minutes=45))
    return rank, world, device


def barrier():
    if dist.is_initialized():
        dist.barrier()


def autocast(device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def pack_chunks(chunks, device):
    lengths = torch.tensor([len(x) for x in chunks], dtype=torch.long, device=device)
    if not len(chunks) or int(lengths.min()) < 2:
        raise ValueError("Empty content chunk")
    width = max(len(x) for x in chunks)
    array = np.zeros((len(chunks), width), dtype=np.int64)
    for i, chunk in enumerate(chunks):
        array[i, : len(chunk)] = chunk
    return torch.from_numpy(array).to(device), lengths


def document_chunks(store, indices):
    chunks, owners = [], []
    for i, index in enumerate(indices):
        values = store.get_document(int(index))
        if not values:
            raise ValueError("Empty document")
        chunks.extend(values)
        owners.extend([i] * len(values))
    return chunks, owners


@torch.no_grad()
def encode_nograd(model, chunks, microbatch, device):
    model.train()  # Deterministic dropout=0; preserve native kernel choice.
    output = []
    for start in range(0, len(chunks), microbatch):
        ids, lengths = pack_chunks(chunks[start : start + microbatch], device)
        with autocast(device):
            value = model(ids, lengths)
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Nonfinite chunk embedding")
        output.append(value.float())
    return torch.cat(output, 0)


def backward_cached(model, chunks, gradients, microbatch, device):
    if len(chunks) != len(gradients):
        raise ValueError("Cached derivative size mismatch")
    model.train()
    for start in range(0, len(chunks), microbatch):
        ids, lengths = pack_chunks(chunks[start : start + microbatch], device)
        with autocast(device):
            value = model(ids, lengths)
        surrogate = (value.float() * gradients[start : start + len(ids)]).sum()
        surrogate.backward()


def trainable_state(model):
    return {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def restore_trainable(model, state):
    current = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(current) != set(state):
        raise ValueError("Trainable state keys differ")
    with torch.no_grad():
        for n, p in current.items():
            if p.shape != state[n].shape or not bool(torch.isfinite(state[n]).all()):
                raise ValueError("Invalid saved trainable tensor: " + n)
            p.copy_(state[n].to(p.device, p.dtype))


def summed_gradients(model):
    """SUM rank-local gradients; actual global query normalization happened once."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    bucket, elements = [], 0

    def flush(values):
        if not values:
            return
        flat = torch.cat(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                for p in values
            ]
        )
        if dist.is_initialized():
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        if not bool(torch.isfinite(flat).all()):
            raise FloatingPointError("Nonfinite trainable gradient")
        offset = 0
        for p in values:
            if p.grad is None:
                p.grad = torch.empty_like(p)
            p.grad.copy_(flat[offset : offset + p.numel()].view_as(p))
            offset += p.numel()

    for p in parameters:
        if bucket and elements + p.numel() > 8 * 1024 * 1024:
            flush(bucket)
            bucket, elements = [], 0
        bucket.append(p)
        elements += p.numel()
    flush(bucket)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def gather_objects(value, world):
    if world == 1:
        return [value]
    result = [None] * world
    dist.all_gather_object(result, value)
    return result


def source_identity(root, config, prepared):
    """Identify the configuration and prepared data needed for a resumed run."""
    payload = {
        "config": config,
        "prepared_manifest_sha256": digest(prepared),
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return payload
