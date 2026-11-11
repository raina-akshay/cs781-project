"""Complete-reference LoRMA SFT with full decoded validation every epoch."""
import argparse
from datetime import timedelta
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .common import ROOT, append_event, atomic_json, now, read_json
from .data import Cache, Formatter, global_batches, own_tokenizer
from .evaluation import evaluate
from .metrics import SummaryScorer
from .model import build_model
from .runtime import (
    canonical_setup,
    configure_cuda,
    frozen_hash,
    optimizer_for,
    restore_delta,
    restore_rng,
    save_checkpoint,
    seed_all,
    update,
)


def initial_progress(steps, epochs):
    return {
        "step": 0,
        "next_epoch": 1,
        "next_batch": 0,
        "epochs_completed": 0,
        "best_epoch": 0,
        "best_score": None,
        "bad_epochs": 0,
        "history": [],
        "steps_per_epoch": steps,
        "planned_steps": steps * epochs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = read_json(args.config)
    t = config["training"]
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local)
    if world != t["world_size"]:
        raise ValueError("Rank count differs from the training configuration")
    torch.cuda.set_device(device)
    configure_cuda()
    dist.init_process_group("nccl", timeout=timedelta(hours=6))
    start = time.monotonic()
    out = args.output or ROOT / "artifacts/run"
    out.mkdir(parents=True, exist_ok=True)
    try:
        if not args.resume and any(
            (out / name).exists()
            for name in ("latest.pt", "best.pt", "events.jsonl", "result.json")
        ):
            raise FileExistsError("Existing training state requires explicit --resume")
        seed_all(t["seed"])
        cache = Cache(config["data_manifest"])
        model = build_model(config).to(device)
        frozen = frozen_hash(model)
        model = DDP(
            model,
            device_ids=[local],
            broadcast_buffers=False,
            static_graph=False,
            gradient_as_bucket_view=True,
        )
        tokenizer = own_tokenizer(config["foundation_manifest"])
        formatter = Formatter(tokenizer, config["sequence"])
        # Same scorer construction in fresh/resumed scored runs; all RNG restoration follows it.
        scorer = SummaryScorer(config["metric_snapshot"], device)
        seed_all(t["seed"] + 1000 + rank)
        canonical_setup(model, cache.splits["train"], formatter, config, world, rank, device)
        optimizer = optimizer_for(model, config)
        steps = math.ceil(len(cache.splits["train"]) / t["global_batch_cases"])
        progress = initial_progress(steps, t["max_epochs"])
        if args.resume:
            saved = torch.load(args.resume, map_location="cpu", weights_only=False)
            if saved["config"] != config or saved["world_size"] != world:
                raise ValueError("Resume configuration or rank count differs")
            restore_delta(model.module, saved)
            optimizer.load_state_dict(saved["optimizer"])
            progress = saved["progress"]
            restore_rng(saved["rng_by_rank"][rank], device)
        if rank == 0:
            append_event(
                out / "events.jsonl",
                {
                    "event": "started",
                    "resume": str(args.resume) if args.resume else None,
                    "global_batch_cases": t["global_batch_cases"],
                    "planned_steps": progress["planned_steps"],
                },
            )
        last_checkpoint = {}

        def save(best=False):
            nonlocal last_checkpoint
            last_checkpoint = save_checkpoint(
                model, optimizer, progress, config, frozen, out, device, rank, world, best=best
            )

        def status(state, last_update=None):
            if rank == 0:
                atomic_json(
                    out / "status.json",
                    {
                        "state": state,
                        "at_utc": now(),
                        **progress,
                        "elapsed_seconds": time.monotonic() - start,
                        "last_update": last_update,
                        "last_checkpoint": last_checkpoint,
                    },
                )

        while progress["next_epoch"] <= t["max_epochs"]:
            epoch = progress["next_epoch"]
            if (
                progress["epochs_completed"] >= t["min_epochs"]
                and progress["bad_epochs"] >= t["patience"]
            ):
                break
            batches = global_batches(
                cache.splits["train"], epoch, t["seed"], t["global_batch_cases"]
            )
            for batch_index in range(progress["next_batch"], len(batches)):
                before = time.monotonic()
                indices = batches[batch_index]
                result = update(
                    model,
                    cache.splits["train"],
                    indices,
                    formatter,
                    optimizer,
                    config,
                    progress["step"] + 1,
                    progress["planned_steps"],
                    world,
                    rank,
                    device,
                )
                progress["step"] += 1
                progress["next_batch"] = batch_index + 1
                result["update_seconds"] = time.monotonic() - before
                if progress["step"] == 1 or progress["step"] % t["save_every_steps"] == 0:
                    save()
                if progress["step"] == 1 or progress["step"] % t["log_every_steps"] == 0:
                    status("running", result)
                    if rank == 0:
                        append_event(
                            out / "events.jsonl",
                            {
                                "event": "update",
                                "step": progress["step"],
                                "epoch": epoch,
                                "batch": batch_index + 1,
                                **result,
                            },
                        )
            status("validation")
            # Commit an exact pre-validation cursor; a crash can replay the same epoch's validation.
            save()
            assessment, _ = evaluate(
                model.module,
                cache.splits["val"],
                formatter,
                config,
                scorer,
                rank,
                world,
                device,
                out / ("epoch-%02d" % epoch),
            )
            improved = (
                progress["best_score"] is None
                or assessment["bertscore_f1"] > progress["best_score"]
            )
            if improved:
                progress["best_score"] = assessment["bertscore_f1"]
                progress["best_epoch"] = epoch
                progress["bad_epochs"] = 0
            else:
                progress["bad_epochs"] += 1
            progress["history"].append(
                {
                    "epoch": epoch,
                    "step": progress["step"],
                    "assessment": assessment,
                    "selected_new_best": improved,
                }
            )
            progress["epochs_completed"] = epoch
            progress["next_epoch"] = epoch + 1
            progress["next_batch"] = 0
            save(best=improved)
            status("running")
            if rank == 0:
                append_event(
                    out / "events.jsonl",
                    {
                        "event": "validation_completed",
                        "epoch": epoch,
                        "assessment": assessment,
                        "selected_new_best": improved,
                    },
                )
        status("completed")
        if rank == 0:
            atomic_json(
                out / "result.json",
                {
                    "completed": True,
                    "at_utc": now(),
                    "progress": progress,
                    "termination": "early_stopping"
                    if progress["next_epoch"] <= t["max_epochs"]
                    else "maximum_epochs",
                    "selected_checkpoint": str((out / "best.pt").resolve()),
                    "config": config,
                    "frozen_hash": frozen,
                },
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
