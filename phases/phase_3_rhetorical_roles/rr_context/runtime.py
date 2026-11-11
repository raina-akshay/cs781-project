"""Checkpoint, batching and prediction helpers for contextual role tagging."""
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

FORMAT = "cs781-phase3-context-delta-v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


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
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(value, device):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if device.type == "cuda":
        if value["cuda"] is None:
            raise ValueError("CUDA checkpoint RNG is missing")
        torch.cuda.set_rng_state(value["cuda"], device)
    elif value["cuda"] is not None:
        raise ValueError("Cannot resume CUDA training RNG on CPU")


def tensor_hash_update(digest, name, tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest.update(name.encode())
    digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())


def frozen_digest(model):
    digest, count = hashlib.sha256(), 0
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            tensor_hash_update(digest, name, parameter)
            count += parameter.numel()
    return digest.hexdigest(), count


def delta_state(model):
    trainable = {name: cpu_tree(p) for name, p in model.named_parameters() if p.requires_grad}
    parameter_names = {name for name, _ in model.named_parameters()}
    buffers = {
        name: cpu_tree(t) for name, t in model.state_dict().items() if name not in parameter_names
    }
    return trainable, buffers


def restore_delta(model, saved):
    if saved.get("format") != FORMAT:
        raise ValueError("Unexpected RR task checkpoint format")
    expected, expected_buffers = delta_state(model)
    if set(saved["trainable_state"]) != set(expected) or set(saved["buffer_state"]) != set(
        expected_buffers
    ):
        raise ValueError("Task delta keys differ from the exact model schema")
    state = model.state_dict()
    for values, reference in (
        (saved["trainable_state"], expected),
        (saved["buffer_state"], expected_buffers),
    ):
        for name, value in values.items():
            if value.shape != reference[name].shape or value.dtype != reference[name].dtype:
                raise ValueError("Task delta tensor schema differs: " + name)
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite task delta: " + name)
            state[name] = value
    model.load_state_dict(state, strict=True)
    actual, count = frozen_digest(model)
    if actual != saved["frozen_sha256"] or count != saved["frozen_parameters"]:
        raise ValueError("Restored frozen own foundation differs")


def make_optimizer(model, training):
    groups = []
    for backbone in (True, False):
        for decay in (True, False):
            values = [
                p
                for name, p in model.named_parameters()
                if p.requires_grad
                and name.startswith("backbone.") == backbone
                and (p.ndim >= 2) == decay
            ]
            if values:
                rate = training["adapter_lr"] if backbone else training["head_lr"]
                groups.append(
                    {
                        "params": values,
                        "lr": rate,
                        "initial_lr": rate,
                        "weight_decay": training["weight_decay"] if decay else 0.0,
                    }
                )
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def set_learning_rate(optimizer, step, total_steps, warmup_fraction, minimum_ratio):
    warmup = max(1, math.ceil(total_steps * warmup_fraction))
    if step <= warmup:
        factor = step / warmup
    else:
        progress = (step - warmup) / max(1, total_steps - warmup)
        factor = minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor
    return factor


def class_weights(cache, power, cap):
    counts = [0] * len(cache["labels"])
    for doc in cache["documents"]:
        if doc["split"] == "train":
            for sentence in doc["sentences"]:
                counts[sentence["label"]] += 1
    if not 0 <= power <= 1 or cap < 1 or not all(counts):
        raise ValueError("Class weights require positive training counts and valid power/cap")
    total, classes = sum(counts), len(counts)
    weights = [(min(cap, max(1 / cap, (total / (classes * count)) ** power))) for count in counts]
    return torch.tensor(weights, dtype=torch.float32), counts


def window_cost(window, maximum_length):
    # Both active sentence count and padded token width determine encoder cost.
    return len(window["tokens"]) * min(maximum_length, max(len(t) + 2 for t in window["tokens"]))


def bucket_batches(windows, effective_windows, maximum_length, seed, epoch, shuffle=True):
    indices = list(range(len(windows)))
    randomizer = random.Random(seed + epoch * 104729)
    if shuffle:
        randomizer.shuffle(indices)
    batches = []
    pool_size = effective_windows * 20
    for start in range(0, len(indices), pool_size):
        pool = indices[start : start + pool_size]
        pool.sort(key=lambda index: window_cost(windows[index], maximum_length))
        batches.extend(
            pool[i : i + effective_windows] for i in range(0, len(pool), effective_windows)
        )
    if shuffle:
        randomizer.shuffle(batches)
    return batches


def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def append_event(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps({"at_utc": now(), **event}, ensure_ascii=False, allow_nan=False) + "\n"
        )


@torch.no_grad()
def predict(model, cache, split, config, device):
    from .window_data import WindowDataset, build_windows, collate_windows
    from rr.metrics import reconstruct_predictions

    context = config["context"]
    windows = build_windows(cache, split, context["core_sentences"], context["halo_sentences"])
    maximum_length = config["input"]["max_length"]
    dataset = WindowDataset(windows, maximum_length)
    order = sorted(range(len(windows)), key=lambda i: window_cost(windows[i], maximum_length))
    batch_size = config["training"]["eval_windows"]
    rows, logits_rows, loss_sum, count = [], [], 0.0, 0
    model.eval()
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch = move_batch(collate_windows([dataset[i] for i in indices]), device)
        with amp(device):
            output = model(batch)
        active = batch["target_mask"]
        logits = output["logits"].float()
        if not bool(torch.isfinite(logits[active]).all()):
            raise FloatingPointError("Nonfinite RR context inference logits")
        predicted = logits.argmax(-1).cpu()
        for j, index in enumerate(indices):
            window = windows[index]
            for position, is_target in enumerate(window["target_mask"]):
                if is_target:
                    sentence_index = window["sentence_indices"][position]
                    rows.append((window["doc_id"], sentence_index, int(predicted[j, position])))
                    logits_rows.append(
                        {
                            "doc_id": window["doc_id"],
                            "sentence_index": sentence_index,
                            "logits": logits[j, position].cpu().tolist(),
                        }
                    )
        if split != "test":
            gold = batch["labels"][active]
            if not bool(((gold >= 0) & (gold < 13)).all()):
                raise ValueError("Every evaluation core target needs one public gold label")
            loss_sum += float(
                torch.nn.functional.cross_entropy(logits[active], gold, reduction="sum")
            )
            count += int(active.sum())
    return reconstruct_predictions(cache, split, rows), {
        "unweighted_cross_entropy": loss_sum / count if count else None,
        "sentences": len(rows),
        "windows": len(windows),
        "logit_rows": logits_rows,
    }
