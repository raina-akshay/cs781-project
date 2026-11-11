"""Strict delta checkpoints, numerical policy and actual global-case updates."""
from contextlib import nullcontext
import hashlib
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import torch.distributed as dist

from .common import FORMAT
from .data import collate, global_batches, rank_segments, segments_for


def configure_cuda():
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True


def amp(device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.device(device).type == "cuda"
        else nullcontext()
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def buffer_state(model):
    return {n: b.detach().cpu().clone() for n, b in model.named_buffers()}


def cpu_tree(x):
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: cpu_tree(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(cpu_tree(v) for v in x)
    if isinstance(x, list):
        return [cpu_tree(v) for v in x]
    return x


def frozen_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad:
            continue
        h.update(name.encode())
        h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def restore_delta(model, saved):
    if saved["format"] != FORMAT:
        raise ValueError("Wrong task checkpoint format")
    state = {n: p for n, p in model.named_parameters() if p.requires_grad}
    delta = saved["trainable_state"]
    buffers = dict(model.named_buffers())
    actual = saved["buffer_state"]
    if state.keys() != delta.keys() or buffers.keys() != actual.keys():
        raise ValueError("Incomplete/extra learned tensors")
    with torch.no_grad():
        for dsts, srcs in ((state, delta), (buffers, actual)):
            for name, dst in dsts.items():
                src = srcs[name]
                if (
                    src.shape != dst.shape
                    or src.dtype != dst.dtype
                    or not bool(torch.isfinite(src).all())
                ):
                    raise ValueError("Bad learned tensor " + name)
                dst.copy_(src)
    if frozen_hash(model) != saved["frozen_hash"]:
        raise ValueError("Original frozen foundation differs")


def rng_state(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def optimizer_for(model, config):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["training"]["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config["training"]["weight_decay"],
        foreach=False,
    )


def set_lr(optimizer, step, total, config):
    t = config["training"]
    warm = max(1, int(total * t["warmup_fraction"]))
    if step <= warm:
        factor = step / warm
    else:
        progress = (step - warm) / max(1, total - warm)
        factor = t["minimum_lr_ratio"] + (1 - t["minimum_lr_ratio"]) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )
    for g in optimizer.param_groups:
        g["lr"] = t["learning_rate"] * factor
    return factor


def backward_global(model, split, indices, formatter, config, world, rank, device):
    jobs = rank_segments(split, indices, config["sequence"]["block_tokens"], world, rank)
    dummy = not jobs
    if dummy:
        jobs = segments_for(split, [indices[0]], config["sequence"]["block_tokens"])[:1]
    micro = config["training"]["micro_segments"]
    batches = [jobs[i : i + micro] for i in range(0, len(jobs), micro)]
    total = 0.0
    tokens = 0
    windows = 0
    for k, items in enumerate(batches):
        batch = collate(split, items, formatter, device, dummy=dummy)
        sync = (
            model.no_sync() if hasattr(model, "no_sync") and k + 1 < len(batches) else nullcontext()
        )
        with sync:
            with amp(device):
                loss = model(**batch)
            (loss * (world / len(indices))).backward()
        total += float(loss.detach())
        if not dummy:
            tokens += sum(s.stop - s.start for s in items)
            windows += len(items)
    return {
        "local_case_normalized_loss_sum": total,
        "local_target_tokens": tokens,
        "local_segments": windows,
        "local_microsteps": len(batches),
    }


def update(
    model, split, indices, formatter, optimizer, config, step, total_steps, world, rank, device
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    factor = set_lr(optimizer, step, total_steps, config)
    stats = backward_global(model, split, indices, formatter, config, world, rank, device)
    params = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params):
        raise FloatingPointError("Missing or nonfinite intended adapter gradient")
    norm = torch.nn.utils.clip_grad_norm_(
        params, config["training"]["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in params):
        raise FloatingPointError("Nonfinite updated adapter")
    optimizer.zero_grad(set_to_none=True)
    stats.update(
        {
            "gradient_norm": float(norm),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "lr_factor": factor,
        }
    )
    sums = torch.tensor(
        [
            stats["local_case_normalized_loss_sum"],
            stats["local_target_tokens"],
            stats["local_segments"],
        ],
        dtype=torch.float64,
        device=device,
    )
    if world > 1:
        dist.all_reduce(sums)
    stats.update(
        {
            "global_cases": len(indices),
            "case_mean_nll": float(sums[0]) / len(indices),
            "global_target_tokens": int(sums[1]),
            "global_segments": int(sums[2]),
        }
    )
    return stats


def canonical_setup(model, split, formatter, config, world, rank, device):
    raw = unwrap(model)
    original = {
        "format": FORMAT,
        "trainable_state": trainable_state(raw),
        "buffer_state": buffer_state(raw),
        "frozen_hash": frozen_hash(raw),
    }
    rng = rng_state(device)
    modes = [(m, m.training) for m in model.modules()]
    batches = global_batches(
        split, 1, config["training"]["seed"], config["training"]["global_batch_cases"]
    )
    selected = batches[: config["training"]["canonical_backward_batches"]]
    for indices in selected:
        model.train()
        model.zero_grad(set_to_none=True)
        backward_global(model, split, indices, formatter, config, world, rank, device)
    model.zero_grad(set_to_none=True)
    restore_delta(raw, original)
    for m, mode in modes:
        m.training = mode
    restore_rng(rng, device)


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
    return {"path": str(path), "step": payload["progress"]["step"]}


def save_checkpoint(
    model, optimizer, progress, config, frozen, out, device, rank, world, best=False
):
    state = rng_state(device)
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, state)
    else:
        states = [state]
    receipt = None
    if rank == 0:
        raw = unwrap(model)
        payload = {
            "format": FORMAT,
            "config": config,
            "trainable_state": trainable_state(raw),
            "buffer_state": buffer_state(raw),
            "optimizer": cpu_tree(optimizer.state_dict()),
            "progress": progress,
            "rng_by_rank": states,
            "frozen_hash": frozen,
            "world_size": world,
        }
        if best:
            atomic_checkpoint(Path(out) / "best.pt", payload)
        receipt = atomic_checkpoint(Path(out) / "latest.pt", payload)
    if world > 1:
        rows = [receipt]
        dist.broadcast_object_list(rows, src=0)
        receipt = rows[0]
    return receipt
