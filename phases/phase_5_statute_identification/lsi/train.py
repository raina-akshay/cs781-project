"""Four-GPU EVA-LoRA complete-case training with disjoint DEV panels."""
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from .data import CacheBundle, collate_cases, global_case_batches, rank_case_indices
from .metrics import assess_dev
from .model import build_model
from .runtime import (
    FORMAT,
    amp,
    append_event,
    atomic_json,
    atomic_torch,
    buffer_state,
    build_optimizer,
    configure_cuda_runtime,
    cpu_tree,
    frozen_state_hash,
    move_batch,
    now,
    predict_split,
    restore_delta,
    restore_rng,
    rng_state,
    seed_everything,
    set_learning_rate,
    trainable_state,
    unwrap,
)


def scaled_case_loss(logits, labels, global_cases, world):
    losses = F.binary_cross_entropy_with_logits(
        logits.float(), labels.float(), reduction="none"
    ).mean(-1)
    total = losses.sum()
    return total * (world / global_cases), total.detach()


def train_update(model, batch, optimizer, config, step, total_steps, global_cases, world, device):
    t = config["training"]
    factor = set_learning_rate(
        optimizer, step, total_steps, t["warmup_fraction"], t["minimum_lr_ratio"]
    )
    optimizer.zero_grad(set_to_none=True)
    moved = move_batch(batch, device)
    with amp(device):
        logits = model(moved)["logits"]
        loss, local_sum = scaled_case_loss(logits, moved["labels"], global_cases, world)
    loss.backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise FloatingPointError("Missing/nonfinite intended gradient")
    norm = torch.nn.utils.clip_grad_norm_(parameters, t["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        raise FloatingPointError("Nonfinite updated parameter")
    return {
        "local_loss_sum": float(local_sum),
        "local_cases": len(batch["ids"]),
        "valid_tokens": int(batch["attention_mask"].sum()),
        "chunks": len(batch["input_ids"]),
        "gradient_norm": float(norm),
        "lr_factor": factor,
        "learning_rates": [group["lr"] for group in optimizer.param_groups],
    }


def canonical_reducer_setup(model, train, config, world, rank, device):
    """Prepare the same DDP reducer before every fresh or resumed trajectory.

    Static DDP changes its reduction layout during its first few backwards.
    Reconstruct that transient using the original initialized model and the
    same three TRAIN batches, before an optimizer exists or a checkpoint is
    loaded. No learned update occurs. Exact tensors, buffers, modes and RNG
    are restored afterward; the caller then loads its desired checkpoint.
    """
    raw = unwrap(model)
    initial = {
        "format": FORMAT,
        "trainable_state": trainable_state(raw),
        "buffer_state": buffer_state(raw),
        "frozen_state": frozen_state_hash(raw),
    }
    initial_rng = rng_state(device)
    modes = [(module, module.training) for module in model.modules()]
    t = config["training"]
    batches = global_case_batches(
        train.lengths, config["input"]["chunk_length"], t["global_batch_cases"], world, t["seed"], 1
    )
    count = 3
    for selected in batches[:count]:
        indices = rank_case_indices(
            selected, train.lengths, config["input"]["chunk_length"], world, rank
        )
        batch = move_batch(collate_cases(train, indices, config["input"]["chunk_length"]), device)
        model.train()
        model.zero_grad(set_to_none=True)
        with amp(device):
            loss, _ = scaled_case_loss(
                model(batch)["logits"], batch["labels"], len(selected), world
            )
        loss.backward()  # Deliberately no optimizer, scheduler or progress update.
    model.zero_grad(set_to_none=True)
    del loss, batch
    # copy_ restoration preserves parameter objects and cuDNN LSTM aliases.
    restore_delta(raw, initial)
    for module, mode in modes:
        module.training = mode
    restore_rng(initial_rng, device)


def run(config_path, resume=None):
    config = json.loads(Path(config_path).read_text())
    initialization = Path(config["adaptation"]["initialization_path"])
    if not initialization.is_absolute():
        initialization = Path(__file__).resolve().parents[1] / initialization
    config["adaptation"]["initialization_path"] = str(initialization.resolve())
    rank, world, local = (
        int(os.environ["RANK"]),
        int(os.environ["WORLD_SIZE"]),
        int(os.environ["LOCAL_RANK"]),
    )
    if world != config["training"]["world_size"]:
        raise ValueError("Rank count differs from the training configuration")
    device = torch.device("cuda", local)
    configure_cuda_runtime(device)
    torch.cuda.set_device(local)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=60))
    started = time.monotonic()
    try:
        cache = CacheBundle(config["data_manifest"])
        train, dev = cache.splits["train"], cache.splits["dev"]
        t = config["training"]
        output = Path(config["output_dir"])
        if not resume and any(
            (output / name).exists()
            for name in ("latest.pt", "best.pt", "result.json", "events.jsonl")
        ):
            raise FileExistsError("Existing training state requires --resume")
        seed_everything(t["seed"] + rank)
        raw = build_model(config).to(device)
        frozen = frozen_state_hash(raw)
        model = DistributedDataParallel(
            raw,
            device_ids=[local],
            output_device=local,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
            bucket_cap_mb=25,
        )
        canonical_reducer_setup(model, train, config, world, rank, device)
        optimizer = build_optimizer(model, config)
        steps = len(
            global_case_batches(train.lengths, 2048, t["global_batch_cases"], world, t["seed"], 1)
        )
        planned = steps * t["max_epochs"]
        progress = {
            "step": 0,
            "next_epoch": 1,
            "next_batch": 0,
            "epochs_completed": 0,
            "best_epoch": 0,
            "best_score": None,
            "patience_count": 0,
            "history": [],
        }
        if resume:
            saved = torch.load(resume, map_location="cpu", weights_only=False)
            if (
                saved.get("format") != FORMAT
                or saved.get("config") != config
                or saved.get("world_size") != world
                or saved.get("planned_steps") != planned
                or len(saved.get("rng_by_rank", [])) != world
            ):
                raise ValueError("Resume configuration, rank count or schedule differs")
            progress = saved["progress"]
            restore_delta(raw, saved)
            optimizer.load_state_dict(saved["optimizer"])
            restore_rng(saved["rng_by_rank"][rank], device)
            del saved
        output.mkdir(parents=True, exist_ok=True)
        last_checkpoint = {}

        def checkpoint(improved=False):
            state = rng_state(device)
            states = [None] * world if rank == 0 else None
            dist.gather_object(state, states, dst=0)
            if rank == 0:
                value = {
                    "format": FORMAT,
                    "config": config,
                    "world_size": world,
                    "planned_steps": planned,
                    "trainable_state": trainable_state(raw),
                    "buffer_state": buffer_state(raw),
                    "optimizer": cpu_tree(optimizer.state_dict()),
                    "rng_by_rank": states,
                    "progress": cpu_tree(progress),
                    "frozen_state": frozen,
                    "parameter_report": raw.parameter_report,
                    "adaptation_report": raw.adaptation_report,
                }
                if improved:
                    atomic_torch(output / "best.pt", value)
                atomic_torch(output / "latest.pt", value)
                last_checkpoint.update(
                    {"path": str(output / "latest.pt"), "step": progress["step"]}
                )

        def status(state, last=None):
            if rank == 0:
                atomic_json(
                    output / "status.json",
                    {
                        "state": state,
                        "at_utc": now(),
                        "step": progress["step"],
                        "epochs_completed": progress["epochs_completed"],
                        "next_epoch": progress["next_epoch"],
                        "next_batch": progress["next_batch"],
                        "best_epoch": progress["best_epoch"],
                        "best_selection_macro_f1": progress["best_score"],
                        "patience_count": progress["patience_count"],
                        "elapsed_seconds": time.monotonic() - started,
                        "steps_per_epoch": steps,
                        "planned_steps": planned,
                        "last_update": last,
                        "last_checkpoint": last_checkpoint,
                    },
                )

        if rank == 0:
            append_event(
                output / "events.jsonl",
                {
                    "event": "resumed" if resume else "started",
                    "parameters": raw.parameter_report,
                    "planned_steps": planned,
                },
            )
        train_hashes = {row["text_sha256"] for row in train.records}
        for epoch in range(progress["next_epoch"], t["max_epochs"] + 1):
            if (
                progress["epochs_completed"] >= t["min_epochs"]
                and progress["patience_count"] >= t["patience"]
            ):
                break
            epoch_start = time.monotonic()
            batches = global_case_batches(
                train.lengths, 2048, t["global_batch_cases"], world, t["seed"], epoch
            )
            model.train()
            for batch_index in range(progress["next_batch"], len(batches)):
                selected = batches[batch_index]
                indices = rank_case_indices(selected, train.lengths, 2048, world, rank)
                batch = collate_cases(train, indices, 2048)
                tick = time.monotonic()
                record = train_update(
                    model,
                    batch,
                    optimizer,
                    config,
                    progress["step"] + 1,
                    planned,
                    len(selected),
                    world,
                    device,
                )
                summary = torch.tensor(
                    [
                        record["local_loss_sum"],
                        record["local_cases"],
                        record["valid_tokens"],
                        record["chunks"],
                    ],
                    dtype=torch.float64,
                    device=device,
                )
                dist.all_reduce(summary)
                record.update(
                    {
                        "unweighted_case_mean_bce": float(summary[0] / summary[1]),
                        "global_cases": int(summary[1]),
                        "global_valid_tokens": int(summary[2]),
                        "global_chunks": int(summary[3]),
                        "update_seconds": time.monotonic() - tick,
                    }
                )
                progress["step"] += 1
                progress["next_batch"] = batch_index + 1
                if progress["step"] == 1 or progress["step"] % t["log_every_steps"] == 0:
                    status("running", record)
                    if rank == 0:
                        append_event(
                            output / "events.jsonl",
                            {"event": "train", "epoch": epoch, "step": progress["step"], **record},
                        )
                if progress["step"] % t["save_every_steps"] == 0 and progress["next_batch"] < steps:
                    checkpoint()
            # Restore dropout/global RNG after clean evaluation before saving the next trajectory state.
            before_eval = rng_state(device)
            _, logits, diagnostics = predict_split(raw, dev, config, device, rank, world)
            restore_rng(before_eval, device)
            improved = False
            if rank == 0:
                predictions, assessment = assess_dev(dev, logits, train_text_hashes=train_hashes)
                score = assessment["selection"]["macro_f1"]
                improved = progress["best_score"] is None or score > progress["best_score"] + 1e-12
                directory = output / f"epoch-{epoch:02d}"
                paths = {}
                for name, value in (
                    ("dev-logits.json", logits),
                    ("dev-prediction.json", predictions),
                    ("assessment.json", assessment),
                ):
                    atomic_json(directory / name, value)
                    paths[name] = str(directory / name)
                entry = {
                    "epoch": epoch,
                    "step": progress["step"],
                    "selected_new_best": improved,
                    "selection_macro_f1": score,
                    "full_dev_macro_f1": assessment["full_dev"]["macro_f1"],
                    "calibration_macro_f1": assessment["calibration"]["macro_f1"],
                    "unweighted_dev_bce": diagnostics["unweighted_bce"],
                    "epoch_seconds": time.monotonic() - epoch_start,
                    "evidence": paths,
                }
                progress["history"].append(entry)
                if improved:
                    progress.update(best_epoch=epoch, best_score=score, patience_count=0)
                else:
                    progress["patience_count"] += 1
                progress.update(epochs_completed=epoch, next_epoch=epoch + 1, next_batch=0)
                append_event(output / "events.jsonl", {"event": "epoch", **entry})
            payload = [progress if rank == 0 else None, improved if rank == 0 else None]
            dist.broadcast_object_list(payload, src=0)
            progress, improved = payload
            checkpoint(improved)
            status("running")
        if rank == 0:
            best = progress["history"][progress["best_epoch"] - 1]
            assessment = json.loads(Path(best["evidence"]["assessment.json"]).read_text())
            result = {
                "completed": True,
                "config": config,
                "epochs_completed": progress["epochs_completed"],
                "step": progress["step"],
                "best_epoch": progress["best_epoch"],
                "best_score": progress["best_score"],
                "history": progress["history"],
                "selected_epoch_evidence": best["evidence"],
                "selected_checkpoint": str(output / "best.pt"),
                "selected_assessment": assessment,
                "parameters": raw.parameter_report,
                "frozen_state": frozen,
                "thresholds_fitted_on": "dev_calibration_only",
                "checkpoint_selected_on": "dev_selection_only",
                "test_labels_used": False,
                "termination": "early_stopping"
                if progress["patience_count"] >= t["patience"]
                else "maximum_epochs",
                "elapsed_seconds": time.monotonic() - started,
            }
            atomic_json(output / "result.json", result)
            status("completed")
        dist.barrier(device_ids=[local])
        return {"completed": True, "step": progress["step"]}
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.resume)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result), flush=True)
