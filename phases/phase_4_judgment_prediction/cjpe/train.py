"""Whole-case, three-GPU DDP training with exact case-mean CE and recovery.

One global batch contributes exactly one optimizer update. Its unique cases
are partitioned by chunk work; ranks can own different case counts. Multiplying
each rank's CE sum by world_size/global_cases before DDP averaging therefore
optimizes the ordinary mean over original cases, not a mean of unequal means.
"""

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from .model import build_model
from .runtime import (
    FORMAT,
    amp,
    append_event,
    atomic_json,
    atomic_torch,
    buffer_state,
    build_optimizer,
    cpu_tree,
    configure_cuda_runtime,
    frozen_state_hash,
    move_batch,
    now,
    predict_split,
    restore_delta,
    restore_rng,
    rng_state,
    seed_everything,
    set_learning_rate,
    sha256_file,
    trainable_state,
    unwrap,
)


def scaled_case_loss(logits, labels, global_case_count, world_size):
    """Exact original-case mean after DDP's all-rank gradient average."""
    if (
        type(global_case_count) is not int
        or type(world_size) is not int
        or not 1 <= len(labels) <= global_case_count
        or not 1 <= world_size <= global_case_count
    ):
        raise ValueError("Invalid local/global case ownership for CE normalization")
    if logits.shape != (len(labels), 2) or not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Nonfinite/invalid training logits")
    if labels.dtype != torch.long or labels.ndim != 1 or bool(((labels < 0) | (labels > 1)).any()):
        raise ValueError("Original cases must carry binary public TRAIN labels")
    numerator = F.cross_entropy(logits.float(), labels, reduction="sum")
    loss = numerator * (world_size / global_case_count)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite whole-case CE objective")
    return loss, numerator


def train_update(
    model,
    optimizer,
    batch,
    global_case_count,
    config,
    device,
    step,
    total_steps,
    rank=0,
    world_size=1,
):
    """One DDP forward/backward, internally checkpointing all owned chunks."""
    if len(batch["ids"]) != len(set(batch["ids"])) or len(batch["ids"]) != len(batch["labels"]):
        raise ValueError("Each local original case must appear exactly once")
    if world_size > 1 and (not dist.is_initialized() or dist.get_world_size() != world_size):
        raise ValueError("Training process group differs from its normalization world size")
    training = config["training"]
    factor = set_learning_rate(
        optimizer, step, total_steps, training["warmup_fraction"], training["minimum_lr_ratio"]
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch = move_batch(batch, device)
    with amp(device):
        logits = model(batch)["logits"]
    loss, numerator = scaled_case_loss(logits, batch["labels"], global_case_count, world_size)
    loss.backward()
    trainable = [parameter for parameter in unwrap(model).parameters() if parameter.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(
        trainable, training["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    totals = torch.tensor(
        [
            float(numerator.detach()),
            len(batch["ids"]),
            int(batch["attention_mask"].sum()),
            len(batch["input_ids"]),
        ],
        dtype=torch.float64,
        device=device,
    )
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if int(totals[1]) != global_case_count:
        raise ValueError("DDP local case counts do not cover the declared global batch")
    return {
        "unweighted_cross_entropy": float(totals[0]) / global_case_count,
        "unweighted_loss_sum": float(totals[0]),
        "cases": global_case_count,
        "valid_tokens": int(totals[2]),
        "chunks": int(totals[3]),
        "gradient_norm": float(norm),
        "lr_factor": factor,
        "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
    }


def _broadcast(value, rank, world_size):
    if world_size == 1:
        return value
    container = [value if rank == 0 else None]
    dist.broadcast_object_list(container, src=0)
    return container[0]


def _rng_all_ranks(device, rank, world_size):
    value = rng_state(device)
    if world_size == 1:
        return [value]
    values = [None] * world_size if rank == 0 else None
    dist.gather_object(value, values, dst=0)
    return values


def train_stage(config, resume=None):
    """Train complete cases across CUDA ranks and select on development Macro-F1."""
    from .data import CacheBundle, collate_cases, global_case_batches, rank_case_indices
    from .metrics import metrics_from_rows

    training = config["training"]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != training["world_size"] or not 0 <= rank < world_size:
        raise ValueError("torchrun world/rank differs from the training configuration")
    owns_group = False
    device = torch.device("cuda", local_rank)
    configure_cuda_runtime(device)
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group("nccl", timeout=timedelta(minutes=60))
            owns_group = True
        if dist.get_world_size() != world_size or dist.get_rank() != rank:
            raise ValueError("Existing process group differs from torchrun")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    latest_path, best_path, final_path = (
        output / name for name in ("latest.pt", "best.pt", "final.pt")
    )
    event_path = output / "events.jsonl"
    progress = None
    start_time = time.monotonic()
    try:
        error = None
        if rank == 0:
            existing = any(
                path.exists()
                for path in (latest_path, best_path, final_path, output / "result.json")
            )
            if resume is None and existing:
                error = "Existing task artifacts require an explicit resume, never a silent restart"
            if resume is not None and not Path(resume).is_file():
                error = "Explicit resume checkpoint does not exist"
        error = _broadcast(error, rank, world_size)
        if error:
            raise RuntimeError(error)
        cache = CacheBundle(config["data_manifest"])
        if (
            cache.splits["train"].labels is None
            or cache.splits["dev"].labels is None
            or cache.splits["test"].labels is not None
        ):
            raise ValueError("Only TRAIN and DEV may expose labels; TEST must redact them")
        initial_batches = global_case_batches(
            cache.splits["train"].lengths,
            config["input"]["chunk_length"],
            training["global_batch_cases"],
            world_size,
            training["seed"],
            1,
            shuffle=True,
        )
        steps_per_epoch = len(initial_batches)
        if not steps_per_epoch:
            raise ValueError("Training split must own a complete global batch")
        planned_steps = steps_per_epoch * training["max_epochs"]
        metadata = {
            "train_cases": len(cache.splits["train"]),
            "dev_cases": len(cache.splits["dev"]),
            "optimizer_splits": ["train"],
            "selection_split": "dev",
            "test_labels_used": False,
            "loss": "unweighted_original_case_mean_cross_entropy",
            "DDP_normalization": "local_CE_sum * world_size / actual_global_case_count",
            "ddp_static_graph": True,
            "world_size": world_size,
            "steps_per_epoch": steps_per_epoch,
            "schedule_epochs": training["max_epochs"],
            "planned_steps": planned_steps,
            "official_dev_used_for_optimization": False,
            "selection": "highest fixed-binary-label DEV macro-F1; ties within 1e-12 keep earliest",
        }
        saved = (
            torch.load(resume, map_location="cpu", weights_only=False)
            if resume is not None
            else None
        )
        if saved is not None and (saved["config"] != config or saved["world_size"] != world_size):
            raise ValueError("Resume configuration or world size differs from the checkpoint")
        seed_everything(training["seed"] + rank)
        model = build_model(config, load_foundation=True, initialize_adapters=saved is None)
        frozen = frozen_state_hash(model)
        if saved is not None:
            unwrap(model).adaptation_report = saved["adaptation_report"]
            restore_delta(model, saved)
        model.to(device)
        optimizer = build_optimizer(model, config)
        if world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        if saved is not None:
            optimizer.load_state_dict(saved["optimizer"])
            progress = saved["progress"]
            if len(saved.get("rng_by_rank", [])) != world_size:
                raise ValueError("Resume checkpoint omits a rank's RNG stream")
            restore_rng(saved["rng_by_rank"][rank], device)
            if rank == 0:
                append_event(
                    event_path,
                    {
                        "event": "resumed",
                        "step": progress["step"],
                        "next_epoch": progress["next_epoch"],
                        "next_batch": progress["next_batch"],
                    },
                )
            del saved
        else:
            progress = {
                "next_epoch": 1,
                "next_batch": 0,
                "epochs_completed": 0,
                "step": 0,
                "best_epoch": 0,
                "best_score": None,
                "patience_count": 0,
                "history": [],
                "last_gradient_norms": [],
            }
            if rank == 0:
                append_event(
                    event_path,
                    {
                        "event": "started",
                        "training_metadata": metadata,
                        "parameters": unwrap(model).parameter_report,
                    },
                )
        initial_step = progress["step"]

        def save(path):
            all_rng = _rng_all_ranks(device, rank, world_size)
            if rank == 0:
                atomic_torch(
                    path,
                    {
                        "format": FORMAT,
                        "config": config,
                        "stage": "holdout",
                        "world_size": world_size,
                        "trainable_state": trainable_state(model),
                        "buffer_state": buffer_state(model),
                        "optimizer": cpu_tree(optimizer.state_dict()),
                        "rng_by_rank": all_rng,
                        "progress": cpu_tree(progress),
                        "frozen_state": frozen,
                        "adaptation_report": unwrap(model).adaptation_report,
                        "parameter_report": unwrap(model).parameter_report,
                        "training_metadata": metadata,
                    },
                )
            if world_size > 1:
                dist.barrier()

        def status(state, **extra):
            if rank != 0:
                return
            elapsed = time.monotonic() - start_time
            updates = progress["step"] - initial_step
            seconds_per_step = elapsed / updates if updates else None
            atomic_json(
                output / "status.json",
                {
                    "at_utc": now(),
                    "state": state,
                    "stage": "holdout",
                    **progress,
                    "planned_steps": planned_steps,
                    "elapsed_seconds_this_process": elapsed,
                    "seconds_per_update_this_process": seconds_per_step,
                    "maximum_budget_eta_seconds": (planned_steps - progress["step"])
                    * seconds_per_step
                    if seconds_per_step is not None
                    else None,
                    **extra,
                },
            )

        status("running")
        already_stopped = (
            progress["epochs_completed"] >= training["min_epochs"]
            and progress["patience_count"] >= training["patience"]
        )
        termination = "early_stopping" if already_stopped else "maximum_epochs"
        epochs = (
            () if already_stopped else range(progress["next_epoch"], training["max_epochs"] + 1)
        )
        for epoch in epochs:
            batches = global_case_batches(
                cache.splits["train"].lengths,
                config["input"]["chunk_length"],
                training["global_batch_cases"],
                world_size,
                training["seed"],
                epoch,
                shuffle=True,
            )
            for batch_index in range(progress["next_batch"], len(batches)):
                global_indices = batches[batch_index]
                if len(global_indices) != len(set(global_indices)):
                    raise ValueError("A global update repeated an original training case")
                local_indices = rank_case_indices(
                    global_indices,
                    cache.splits["train"].lengths,
                    config["input"]["chunk_length"],
                    world_size,
                    rank,
                )
                batch = collate_cases(
                    cache.splits["train"], local_indices, config["input"]["chunk_length"]
                )
                update_started = time.monotonic()
                stats = train_update(
                    model,
                    optimizer,
                    batch,
                    len(global_indices),
                    config,
                    device,
                    progress["step"] + 1,
                    planned_steps,
                    rank,
                    world_size,
                )
                progress["step"] += 1
                progress["next_batch"] = batch_index + 1
                progress["last_gradient_norms"] = (
                    progress["last_gradient_norms"] + [stats["gradient_norm"]]
                )[-50:]
                if rank == 0 and (
                    progress["step"] == 1 or progress["step"] % training["log_every_steps"] == 0
                ):
                    append_event(
                        event_path,
                        {
                            "event": "train",
                            "epoch": epoch,
                            "step": progress["step"],
                            "batch": batch_index + 1,
                            "update_seconds": time.monotonic() - update_started,
                            **stats,
                        },
                    )
                    status("running", last_update=stats)
                if progress["step"] % training["save_every_steps"] == 0:
                    save(latest_path)
            predictions, logits, diagnostics = predict_split(
                model, cache.splits["dev"], config, device, rank, world_size
            )
            entry = None
            if rank == 0:
                metrics = metrics_from_rows(cache.splits["dev"], predictions)
                score = metrics["macro_f1"]
                if not math.isfinite(score) or not math.isfinite(
                    diagnostics["unweighted_cross_entropy"]
                ):
                    raise FloatingPointError("Nonfinite public-DEV evaluation")
                improved = progress["best_score"] is None or score > progress["best_score"] + 1e-12
                entry = {
                    "epoch": epoch,
                    "step": progress["step"],
                    "metrics": metrics,
                    "unweighted_dev_cross_entropy": diagnostics["unweighted_cross_entropy"],
                    "selected_new_best": improved,
                }
            entry = _broadcast(entry, rank, world_size)
            if entry["selected_new_best"]:
                progress["best_score"], progress["best_epoch"] = entry["metrics"]["macro_f1"], epoch
                progress["patience_count"] = 0
            else:
                progress["patience_count"] += 1
            progress["history"].append(entry)
            progress["epochs_completed"] = epoch
            progress["next_epoch"], progress["next_batch"] = epoch + 1, 0
            if entry["selected_new_best"]:
                save(best_path)
                if rank == 0:
                    atomic_json(output / "best-dev-prediction.json", predictions)
                    atomic_json(output / "best-dev-logits.json", logits)
                    atomic_json(output / "best-dev-metrics.json", entry)
            save(latest_path)
            if rank == 0:
                append_event(
                    event_path,
                    {
                        "event": "validation",
                        **entry,
                        "best_epoch": progress["best_epoch"],
                        "patience_count": progress["patience_count"],
                    },
                )
                status("running")
            if (
                epoch >= training["min_epochs"]
                and progress["patience_count"] >= training["patience"]
            ):
                termination = "early_stopping"
                break
        if progress["best_epoch"] < 1 or not best_path.exists():
            raise RuntimeError("No trained-epoch development checkpoint was selected")
        save(final_path)
        save(latest_path)
        selected = torch.load(best_path, map_location="cpu", weights_only=False)
        if (
            selected["progress"]["best_epoch"] != progress["best_epoch"]
            or selected["progress"]["best_score"] != progress["best_score"]
        ):
            raise ValueError("Selected checkpoint contradicts final selection history")
        restore_delta(model, selected)
        del selected
        predictions, logits, diagnostics = predict_split(
            model, cache.splits["dev"], config, device, rank, world_size
        )
        result = None
        if rank == 0:
            metrics = metrics_from_rows(cache.splits["dev"], predictions)
            best_entry = next(
                item for item in progress["history"] if item["epoch"] == progress["best_epoch"]
            )
            atomic_json(output / "best-dev-prediction.json", predictions)
            atomic_json(output / "best-dev-logits.json", logits)
            atomic_json(output / "best-dev-metrics.json", best_entry)
            result = {
                "completed": True,
                "state": "complete",
                "best_epoch": progress["best_epoch"],
                "best_score": progress["best_score"],
                "best_metrics": metrics,
                "metrics": metrics,
                "epochs_completed": progress["epochs_completed"],
                "step": progress["step"],
                "termination": termination,
                "history": progress["history"],
                "selected_checkpoint": str(best_path.resolve()),
                "selected_checkpoint_sha256": sha256_file(best_path),
                "final_checkpoint": str(final_path.resolve()),
                "final_checkpoint_sha256": sha256_file(final_path),
                "best_prediction_path": str((output / "best-dev-prediction.json").resolve()),
                "frozen_state": frozen,
                "parameters": unwrap(model).parameter_report,
                "training_metadata": metadata,
                "elapsed_seconds_this_process": time.monotonic() - start_time,
            }
            atomic_json(output / "result.json", result)
            status("completed", termination=termination)
            append_event(
                event_path,
                {
                    "event": "completed",
                    "best_epoch": progress["best_epoch"],
                    "best_score": progress["best_score"],
                    "termination": termination,
                },
            )
        return _broadcast(result, rank, world_size)
    finally:
        if owns_group and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", nargs="?", const="latest")
    arguments = parser.parse_args()
    config = json.loads(Path(arguments.config).read_text())
    resume = arguments.resume
    if resume == "latest":
        resume = str(Path(config["output_dir"]) / "latest.pt")
    result = train_stage(config, resume)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
