"""Single-GPU contextual RR training with exact owned-target normalization.

Only public train labels optimize parameters. Halos provide features; every
sentence has exactly one supervised core ownership per epoch. Official dev
selects checkpoints, never refits. Each logical update normalizes weighted cross-entropy over its core targets.
"""

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from rr.data import flat_records, load_cache
from .window_data import WindowDataset, build_windows, collate_windows
from rr.metrics import metrics_from_predictions
from .model import build_model
from .runtime import (
    FORMAT,
    amp,
    append_event,
    atomic_json,
    atomic_torch,
    bucket_batches,
    class_weights,
    cpu_tree,
    delta_state,
    frozen_digest,
    make_optimizer,
    move_batch,
    now,
    predict,
    restore_delta,
    restore_rng,
    rng_state,
    seed_everything,
    set_learning_rate,
    sha256,
)


def train_update(model, optimizer, rows, dataset, weights, config, device, step, total_steps):
    """Normalize weighted cross-entropy over all core targets in the update."""
    training = config["training"]
    micro = training["micro_windows"]
    if (
        weights.ndim != 1
        or weights.numel() != 13
        or not bool(torch.isfinite(weights).all())
        or not bool((weights > 0).all())
    ):
        raise ValueError("Every official role needs a finite positive training weight")
    examples = [dataset[index] for index in rows]
    labels_list, owners = [], set()
    for example in examples:
        for position, active in enumerate(example["target_mask"]):
            label = example["labels"][position]
            if active:
                owner = (example["doc_id"], example["sentence_indices"][position])
                if owner in owners or type(label) is not int or not 0 <= label < 13:
                    raise ValueError("Each training core target needs one unique public gold label")
                owners.add(owner)
                labels_list.append(label)
            elif label != -100:
                raise ValueError("Halo positions must redact labels")
    labels = torch.tensor(labels_list, dtype=torch.long)
    if not labels.numel():
        raise ValueError("A logical update must own at least one real target")
    denominator = float(weights.detach().cpu()[labels].double().sum())
    target_count = labels.numel()
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Invalid logical weighted CE denominator")
    weights_device = weights.to(device=device, dtype=torch.float32)
    factor = set_learning_rate(
        optimizer, step, total_steps, training["warmup_fraction"], training["minimum_lr_ratio"]
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    numerator = 0.0
    token_count = active_sentences = 0
    for start in range(0, len(examples), micro):
        batch = move_batch(collate_windows(examples[start : start + micro]), device)
        with amp(device):
            output = model(batch)
        active = batch["target_mask"]
        logits = output["logits"].float()[active]
        gold = batch["labels"][active]
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Nonfinite training logits")
        loss_sum = F.cross_entropy(logits, gold, weight=weights_device, reduction="sum")
        loss = loss_sum / denominator
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite weighted context objective")
        loss.backward()
        numerator += float(loss_sum.detach())
        token_count += int(batch["lengths"].sum())
        active_sentences += len(batch["lengths"])
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(
        trainable, training["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    objective = numerator / denominator
    return {
        "weighted_cross_entropy": numerator / denominator,
        "weighted_loss_sum": numerator,
        "weight_denominator": denominator,
        "objective": objective,
        "windows": len(rows),
        "sentences": target_count,
        "encoded_sentences": active_sentences,
        "valid_tokens": token_count,
        "gradient_norm": float(norm),
        "lr_factor": factor,
        "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
    }


def _checkpoint(model, optimizer, config, progress, device, frozen, metadata):
    trainable, buffers = delta_state(model)
    return {
        "format": FORMAT,
        "config": config,
        "stage": "holdout",
        "trainable_state": trainable,
        "buffer_state": buffers,
        "optimizer": cpu_tree(optimizer.state_dict()),
        "rng": rng_state(device),
        "frozen_sha256": frozen[0],
        "frozen_parameters": frozen[1],
        "progress": cpu_tree(progress),
        "adaptation_report": model.adaptation_report,
        "parameter_report": model.parameter_report,
        "training_metadata": metadata,
    }


def train_stage(config, resume=False):
    """Train on public training labels and select checkpoints on development data."""
    training = config["training"]
    output = Path(config["output_dir"]) / "holdout"
    output.mkdir(parents=True, exist_ok=True)
    latest_path, best_path, final_path = (
        output / name for name in ("latest.pt", "best.pt", "final.pt")
    )
    log_path = output / "events.jsonl"
    if not resume and any(
        path.exists() for path in (latest_path, best_path, final_path, output / "result.json")
    ):
        raise FileExistsError(
            "Existing task artifacts require an explicit --resume; no silent restart"
        )
    if resume and not latest_path.exists():
        raise FileNotFoundError("Explicit resume requires latest.pt")
    start_time = time.monotonic()
    progress = None
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    cache = load_cache(config["data_cache"])
    records = flat_records(cache, splits=("train",))
    if not records or any(row["label"] is None for row in records):
        raise ValueError("Training must contain only public train labels")
    context = config["context"]
    windows = build_windows(cache, "train", context["core_sentences"], context["halo_sentences"])
    dataset = WindowDataset(windows, config["input"]["max_length"])
    weights, counts = class_weights(
        cache, config["loss"]["class_weight_power"], config["loss"]["class_weight_cap"]
    )
    batches_per_epoch = math.ceil(len(windows) / training["effective_windows"])
    total_steps = batches_per_epoch * training["max_epochs"]
    metadata = {
        "train_sentences": len(records),
        "train_windows": len(windows),
        "dev_sentences": len(flat_records(cache, ("dev",))),
        "optimizer_splits": ["train"],
        "selection_split": "dev",
        "official_dev_used_for_optimization": False,
        "class_weight_fit_split": "train",
        "class_counts": counts,
        "class_weights": weights.tolist(),
        "steps_per_epoch": batches_per_epoch,
        "schedule_epochs": training["max_epochs"],
        "planned_steps": total_steps,
        "selection": "highest combined fixed-13-label dev macro-F1; ties within 1e-12 keep earliest",
    }
    saved = torch.load(latest_path, map_location="cpu", weights_only=False) if resume else None
    if saved is not None and saved["config"] != config:
        raise ValueError("Resume configuration differs from the checkpoint")
    seed_everything(training["seed"])
    model = build_model(config, load_foundation=True, initialize_adapters=saved is None)
    frozen = frozen_digest(model)
    if saved is not None:
        model.adaptation_report = saved["adaptation_report"]
        restore_delta(model, saved)
    model.to(device)
    optimizer = make_optimizer(model, training)
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
        progress = saved["progress"]
        restore_rng(saved["rng"], device)
        append_event(
            log_path,
            {
                "event": "resumed",
                "step": progress["step"],
                "next_epoch": progress["next_epoch"],
                "next_batch": progress["next_batch"],
            },
        )
        del saved
    else:
        _, initial = predict(model, cache, "dev", config, device)
        initial_ce = initial["unweighted_cross_entropy"]
        if initial_ce is None or not math.isfinite(initial_ce):
            raise FloatingPointError("Initial public-dev cross entropy is not finite")
        progress = {
            "next_epoch": 1,
            "next_batch": 0,
            "epochs_completed": 0,
            "step": 0,
            "best_epoch": 0,
            "best_score": None,
            "patience_count": 0,
            "history": [],
            "initial_dev_cross_entropy": initial_ce,
        }
        append_event(
            log_path,
            {
                "event": "started",
                "metadata": metadata,
                "parameters": model.parameter_report,
                "initial_dev_cross_entropy": initial_ce,
            },
        )
    if (output / "result.json").exists():
        result = json.loads((output / "result.json").read_text())
        return result
    initial_step = progress["step"]

    def save(path):
        atomic_torch(
            path, _checkpoint(model, optimizer, config, progress, device, frozen, metadata)
        )

    def status(state, **extra):
        elapsed = time.monotonic() - start_time
        newly_trained = progress["step"] - initial_step
        seconds_per_step = elapsed / newly_trained if newly_trained else None
        atomic_json(
            output / "status.json",
            {
                "at_utc": now(),
                "state": state,
                "stage": "holdout",
                **progress,
                "planned_steps": total_steps,
                "elapsed_seconds_this_process": elapsed,
                "maximum_budget_eta_seconds": (
                    (total_steps - progress["step"]) * seconds_per_step
                    if seconds_per_step is not None
                    else None
                ),
                **extra,
            },
        )

    status("running")
    already_early_stopped = (
        progress["epochs_completed"] >= training["min_epochs"]
        and progress["patience_count"] >= training["patience"]
    )
    termination = "early_stopping" if already_early_stopped else "maximum_epochs"
    epochs_to_run = (
        () if already_early_stopped else range(progress["next_epoch"], training["max_epochs"] + 1)
    )
    for epoch in epochs_to_run:
        batches = bucket_batches(
            windows,
            training["effective_windows"],
            config["input"]["max_length"],
            training["seed"],
            epoch,
            shuffle=True,
        )
        for batch_index in range(progress["next_batch"], len(batches)):
            step = progress["step"] + 1
            update_start = time.monotonic()
            stats = train_update(
                model,
                optimizer,
                batches[batch_index],
                dataset,
                weights,
                config,
                device,
                step,
                total_steps,
            )
            progress["step"] = step
            progress["next_batch"] = batch_index + 1
            if step == 1 or step % training["log_every_steps"] == 0:
                append_event(
                    log_path,
                    {
                        "event": "train",
                        "epoch": epoch,
                        "step": step,
                        "batch": batch_index + 1,
                        "update_seconds": time.monotonic() - update_start,
                        **stats,
                    },
                )
                status("running", last_update=stats)
            if step % training["save_every_steps"] == 0:
                save(latest_path)
        predictions, diagnostics = predict(model, cache, "dev", config, device)
        dev_ce = diagnostics["unweighted_cross_entropy"]
        metrics = metrics_from_predictions(cache, predictions, split="dev")
        score = metrics["combined"]["macro_f1"]
        if dev_ce is None or not math.isfinite(dev_ce) or not math.isfinite(score):
            raise FloatingPointError("Nonfinite public-dev evaluation")
        improved = progress["best_score"] is None or score > progress["best_score"] + 1e-12
        if improved:
            progress["best_score"], progress["best_epoch"] = score, epoch
            progress["patience_count"] = 0
        else:
            progress["patience_count"] += 1
        entry = {
            "epoch": epoch,
            "step": progress["step"],
            "unweighted_dev_cross_entropy": dev_ce,
            "metrics": metrics,
            "selected_new_best": improved,
        }
        progress["history"].append(entry)
        progress["epochs_completed"] = epoch
        progress["next_epoch"], progress["next_batch"] = epoch + 1, 0
        if improved:
            save(best_path)
            atomic_json(output / "best-dev-prediction.json", predictions)
            atomic_json(
                output / "best-dev-metrics.json",
                {
                    "epoch": epoch,
                    "step": progress["step"],
                    "metrics": metrics,
                    "unweighted_cross_entropy": dev_ce,
                },
            )
        save(latest_path)
        append_event(
            log_path,
            {
                "event": "validation",
                **entry,
                "best_epoch": progress["best_epoch"],
                "patience_count": progress["patience_count"],
            },
        )
        status("running")
        if epoch >= training["min_epochs"] and progress["patience_count"] >= training["patience"]:
            termination = "early_stopping"
            break
    if progress["best_epoch"] < 1 or not best_path.exists():
        raise RuntimeError("No trained-epoch checkpoint was selected")
    save(final_path)
    save(latest_path)
    best_entry = next(
        entry for entry in progress["history"] if entry["epoch"] == progress["best_epoch"]
    )
    # Export the checkpoint chosen by development Macro-F1.
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    restore_delta(model, selected)
    best_predictions, _ = predict(model, cache, "dev", config, device)
    best_prediction_path = output / "best-dev-prediction.json"
    atomic_json(best_prediction_path, best_predictions)
    del selected
    result = {
        "completed": True,
        "state": "complete",
        "stage": "holdout",
        "best_epoch": progress["best_epoch"],
        "best_score": progress["best_score"],
        "best_metrics": best_entry["metrics"],
        "metrics": best_entry["metrics"],
        "epochs_completed": progress["epochs_completed"],
        "step": progress["step"],
        "termination": termination,
        "history": progress["history"],
        "selected_checkpoint": str(best_path.resolve()),
        "selected_checkpoint_sha256": sha256(best_path),
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256(best_path),
        "best_prediction_path": str(best_prediction_path.resolve()),
        "best_prediction_sha256": sha256(best_prediction_path),
        "final_checkpoint": str(final_path.resolve()),
        "final_checkpoint_sha256": sha256(final_path),
        "frozen_parameters": frozen[1],
        "frozen_sha256": frozen[0],
        "early_stopped": termination == "early_stopping",
        "reason": termination,
        "parameters": model.parameter_report,
        "training_metadata": metadata,
    }
    atomic_json(output / "result.json", result)
    status("completed", termination=termination, selected_checkpoint=str(best_path.resolve()))
    append_event(
        log_path,
        {
            "event": "completed",
            "best_epoch": progress["best_epoch"],
            "best_score": progress["best_score"],
            "epochs_completed": progress["epochs_completed"],
            "termination": termination,
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = train_stage(config, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
