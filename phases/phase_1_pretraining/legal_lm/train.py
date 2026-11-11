"""Shared training helpers and token-normalized loss.

Every rank contributes a *sum* of losses. DDP averages gradients across ranks,
so each local sum is multiplied by world_size / global_valid_target_count.
Padding and differing document lengths therefore cannot bias the objective.
"""

from __future__ import annotations

from contextlib import nullcontext
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F

from .config import ModelConfig, TrainConfig
from .model import LegalLanguageModel


class EpochSampler:
    """Global shuffled stream, with deterministic replay from an example cursor."""

    def __init__(self, length, seed, position=0):
        if length < 1 or position < 0:
            raise ValueError("Sampler needs positive length and nonnegative position")
        self.length, self.seed, self.position = int(length), int(seed), int(position)
        self._epoch, self._order = None, None

    def take(self, count):
        if count < 0:
            raise ValueError("Negative sample count")
        result = []
        while count:
            epoch, offset = divmod(self.position, self.length)
            if self._epoch != epoch:
                self._order = np.random.default_rng(self.seed + epoch).permutation(self.length)
                self._epoch = epoch
            n = min(count, self.length - offset)
            result.extend(self._order[offset : offset + n].tolist())
            self.position += n
            count -= n
        return result


def lr_for_tokens(tokens_seen, cfg: TrainConfig):
    """Linear warmup followed by cosine decay over the training token budget."""
    stage_tokens = max(0.0, tokens_seen)
    warmup = cfg.max_tokens * cfg.warmup_fraction
    if warmup and stage_tokens < warmup:
        return cfg.learning_rate * stage_tokens / warmup
    progress = min(1.0, max(0.0, (stage_tokens - warmup) / (cfg.max_tokens - warmup)))
    return cfg.learning_rate * (
        cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
    )


def masked_cross_entropy_sum(logits, labels):
    # Float32 CE is deliberate even when Transformer matmuls use AMP.
    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss, int((labels != -100).sum().item())


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


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def setup_distributed(device_name):
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda":
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                "Requested CUDA is unavailable. Check the driver and CUDA_VISIBLE_DEVICES; no CPU fallback was used."
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("device must be cuda/cpu/auto")
    if world > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return rank, world, device


def choose_precision(requested, device):
    if requested == "auto":
        requested = (
            ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
            if device.type == "cuda"
            else "fp32"
        )
    if requested != "fp32" and device.type != "cuda":
        raise ValueError("CPU diagnostic training uses fp32; mixed precision requires CUDA here")
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested BF16 is unsupported by this CUDA device")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[requested]
    return requested, dtype


def amp_context(device, dtype):
    return (
        torch.autocast(device_type=device.type, dtype=dtype)
        if dtype != torch.float32
        else nullcontext()
    )


def _sum(value, device, world):
    tensor = torch.as_tensor(value, device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(tensor)
    return tensor.cpu().tolist()


def _keep_first_targets(batches, keep):
    """Mask the final budget excess, never turn padding back into supervision."""
    for batch in batches:
        labels = batch["labels"]
        valid_positions = (labels.reshape(-1) != -100).nonzero().flatten()
        n = min(keep, len(valid_positions))
        labels.reshape(-1)[valid_positions[n:]] = -100
        keep -= n


def _atomic_checkpoint(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", suffix=".pt", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path):
    # Only load this project's own trusted checkpoints: optimizer/RNG state
    # includes Python objects and cannot use PyTorch's weights_only format.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported training checkpoint format")
    return checkpoint


def load_model_checkpoint(path, device="cpu"):
    checkpoint = load_checkpoint(path)
    model = LegalLanguageModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def validate_lm(model, dataset, max_tokens, device, dtype, rank=0, world=1, *, trim_padding=False):
    """Same deterministic document-window prefix on every evaluation/resume."""
    was_training = model.training
    model.eval()
    sums = np.zeros((len(dataset.source_names), 2), dtype=np.float64)
    remaining = max_tokens
    try:
        # Shuffling is local and fixed: token files are source-ordered, so a
        # simple prefix could accidentally evaluate only the first source.
        order = np.random.default_rng(1729).permutation(len(dataset))
        for evaluation_index, index in enumerate(order.tolist()):
            count = min(remaining, int(dataset.target_counts[index]))
            if count <= 0:
                break
            remaining -= count
            if evaluation_index % world != rank:
                continue
            item = dataset[index]
            batch = {"labels": item["labels"].clone().unsqueeze(0)}
            _keep_first_targets([batch], count)
            ids = item["input_ids"].unsqueeze(0)
            labels = batch["labels"]
            if trim_padding:
                ids, labels = ids[:, :count].contiguous(), labels[:, :count].contiguous()
            ids, labels = ids.to(device), labels.to(device)
            with amp_context(device, dtype):
                loss, n = masked_cross_entropy_sum(model(ids).logits, labels)
            source = int(item["source_id"])
            sums[source] += (loss.item(), n)
        sums = np.asarray(_sum(sums, device, world))
    finally:
        model.train(was_training)
    total_loss, count = sums.sum(axis=0)
    if count == 0:
        raise ValueError("Development token split has no valid targets")
    mean = float(total_loss / count)
    return {
        "loss": mean,
        "perplexity": math.exp(mean) if mean < 700 else None,
        "target_tokens": int(count),
        "per_source": {
            name: {
                "target_tokens": int(sums[i, 1]),
                "loss": float(sums[i, 0] / sums[i, 1]) if sums[i, 1] else None,
            }
            for i, name in enumerate(dataset.source_names)
        },
    }
