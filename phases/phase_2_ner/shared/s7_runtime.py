"""Distributed training, checkpointing and reproducible random state."""
from contextlib import nullcontext
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "cs781-phase2-s7-span-distillation-charner-v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(value, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def identity(config, dataset, stage, world):
    cache_path = Path(config["cache_path"])
    if cache_path.is_dir():
        cache_path /= "cache.pkl"
    foundation = json.loads(Path(config["foundation_manifest"]).read_text())
    return {
        "config": config,
        "stage": stage,
        "world_size": world,
        "data_manifest": dataset.manifest,
        "cache_sha256": sha256_file(cache_path),
        "foundation_checkpoint_sha256": foundation["checkpoint_sha256"],
        "tokenizer_sha256": foundation["tokenizer_sha256"],
        "precision": "bf16",
        "format": FORMAT,
    }


def setup(expected_world=None):
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if expected_world is not None and world != expected_world:
        raise ValueError(
            f"Requested {expected_world} GPUs but process world is {world}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the production NER run")
    torch.cuda.set_device(local)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 support is required")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    device = torch.device("cuda", local)
    object_group = None
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        # CPU prediction/RNG gathers avoid serializing full-document logits on GPU.
        object_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=30))
    return rank, world, device, object_group


def close():
    if dist.is_initialized():
        dist.destroy_process_group()


def amp(device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def gather_objects(value, rank, world, group):
    if world == 1:
        return [value]
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(value, gathered, dst=0, group=group)
    return gathered


def broadcast_object(value, rank, world, group):
    values = [value]
    if world > 1:
        dist.broadcast_object_list(values, src=0, group=group)
    return values[0]


def sum_all(values, device, world):
    out = torch.as_tensor(values, dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(out)
    return out.cpu().tolist()


def require_finite(value, device, world, description):
    ok = (
        torch.isfinite(torch.as_tensor(value, device=device))
        .all()
        .to(dtype=torch.int32)
    )
    if world > 1:
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if not bool(ok.item()):
        raise FloatingPointError(
            f"Nonfinite {description}; stopped before optimizer update"
        )


def move_batch(batch, device):
    # pack_padded_sequence needs CPU lengths; metadata remains on CPU as well.
    return {
        key: (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            and key not in ("char_lengths", "token_lengths")
            else value
        )
        for key, value in batch.items()
    }


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != FORMAT:
        raise ValueError("Not a checkpoint generated by this Phase 2 NER pipeline")
    return state
