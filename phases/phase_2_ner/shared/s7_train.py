"""Train LoRMA NER, select on holdout data, then refit for the selected duration."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import zipfile
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .ner_contract import load_json
from .s7_span import load_cache, collate_windows
from .s7_loss import (
    prepare_loss,
    summed_loss,
    summed_role_loss,
    summed_ranking_loss,
    summed_span_loss,
)
from .s7_runtime import (
    ROOT,
    FORMAT,
    amp,
    atomic_checkpoint,
    atomic_json,
    broadcast_object,
    close,
    gather_objects,
    identity,
    load_checkpoint,
    move_batch,
    now,
    require_finite,
    restore_rng,
    rng_state,
    seed_everything,
    setup,
    sum_all,
)


def distributed_batches(dataset, epoch, seed, micro, accum, world):
    """Every real window once/epoch; missing final slots have zero loss weight.

    Yield optimizer-update groups, each containing equally sized rank batches.
    Batch entries are (window_index, is_real_supervised_slot). Shuffling is a pure
    function of epoch and seed, so an exact cursor suffices for resumption.
    """
    if min(micro, accum, world) < 1:
        raise ValueError("Batch, accumulation and world size must be positive")
    order = list(range(len(dataset.windows)))
    if not order:
        raise ValueError("No training windows")
    random.Random(seed + 1000003 * epoch).shuffle(order)
    global_micro = micro * world
    global_update = global_micro * accum
    for start in range(0, len(order), global_update):
        group = []
        for a in range(accum):
            begin = start + a * global_micro
            if begin >= len(order):
                break
            slots = [(i, True) for i in order[begin : begin + global_micro]]
            slots += [(order[0], False)] * (global_micro - len(slots))
            group.append([slots[r * micro : (r + 1) * micro] for r in range(world)])
        yield group


def make_optimizer(model, training):
    groups = []
    for backbone, lrkey in ((True, "backbone_lr"), (False, "head_lr")):
        for decay in (True, False):
            params = [
                p
                for name, p in model.named_parameters()
                if p.requires_grad
                and name.startswith("backbone.") == backbone
                and (p.ndim >= 2 and not getattr(p, "_no_weight_decay", False)) == decay
            ]
            if params:
                groups.append(
                    {
                        "params": params,
                        "lr": training[lrkey],
                        "initial_lr": training[lrkey],
                        "weight_decay": training["weight_decay"] if decay else 0.0,
                    }
                )
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def global_owned_count(batches, device, world):
    local = sum(int((b["loss_mask"] & (b["labels"] != -100)).sum()) for b in batches)
    return int(sum_all([local], device, world)[0])


def global_entity_count(batches, device, world):
    local = sum(
        int(b.get("role_entity_mask", torch.zeros(0, dtype=torch.bool)).sum())
        for b in batches
    )
    return int(sum_all([local], device, world)[0])


def global_comparison_count(batches, device, world):
    local = sum(int(b["rank_comparison_mask"].sum()) for b in batches)
    return int(sum_all([local], device, world)[0])


def global_span_count(batches, device, world):
    count = sum(int(batch["span_comparison_mask"].sum()) for batch in batches)
    return int(sum_all([count], device, world)[0])


def learning_rate_factor(step, total, warmup_fraction):
    warmup = max(1, math.ceil(total * warmup_fraction))
    if step < warmup:
        return (step + 1) / warmup
    fraction = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * fraction))


def log_event(path, event, rank):
    if rank == 0:
        event = {"at_utc": now(), **event}
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
        print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def restore_adaptation_report(model, state):
    model.adaptation_report = state["adaptation_report"]


def train_stage(config, stage, stop_after_steps=None):
    training = config["training"]
    rank, world, device, object_group = setup(training["world_size"])
    output = Path(config["output_dir"]) / stage
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "events.jsonl"
    try:
        from .s7_model import build_model
        from .s7_predict import evaluate_model

        role_enabled = float(config.get("role_loss", {}).get("lambda", 0.0)) > 0
        dataset = load_cache(
            config["cache_path"],
            "train" if stage == "holdout" else "all_train",
            role_auxiliary=role_enabled,
        )
        validation = (
            load_cache(config["cache_path"], "validation")
            if stage == "holdout"
            else None
        )
        ident = identity(config, dataset, stage, world)
        loss_weights_cpu, loss_metadata = prepare_loss(config)
        lambda_boundary = float(loss_metadata["lambda_boundary"])
        lambda_role = float(loss_metadata.get("lambda_role", 0.0))
        lambda_ranking = float(loss_metadata["lambda_ranking"])
        lambda_span = float(loss_metadata["lambda_span"])
        loss_weights = (
            loss_weights_cpu.to(device=device, dtype=torch.float32)
            if loss_weights_cpu is not None
            else None
        )
        if stage == "refit":
            selected = load_json(Path(config["output_dir"]) / "holdout" / "result.json")
            if not selected.get("completed"):
                raise ValueError("Refit requires completed holdout model selection")
            epochs = int(selected["best_epoch"])
            if not 1 <= epochs <= training["max_epochs"]:
                raise ValueError("Invalid selected refit duration")
        else:
            epochs = training["max_epochs"]
        micro = training["micro_batch_per_gpu"]
        accum = training["gradient_accumulation_steps"]
        steps_per_epoch = math.ceil(len(dataset.windows) / (micro * world * accum))
        # Refit stops at the selected epoch on the same planned maximum-epoch LR curve.
        # Compressing the scheduler to the selected duration changes the validated protocol.
        schedule_epochs = training["max_epochs"]
        total_steps = steps_per_epoch * schedule_epochs
        latest_path = output / "latest.pt"
        state = load_checkpoint(latest_path) if latest_path.exists() else None
        if state is not None and state["identity"] != ident:
            raise ValueError("Resume identity mismatch; no silent restart is permitted")
        if state is not None and len(state.get("rng_states", [])) != world:
            raise ValueError("Resume checkpoint does not contain every rank RNG state")
        seed_everything(training["seed"])
        model = build_model(
            config, dataset.char_vocab, dataset.labels, load_foundation=state is None
        )
        model.backbone.gradient_checkpointing = training.get(
            "gradient_checkpointing", False
        )
        if state is not None:
            restore_adaptation_report(model, state)
            model.load_state_dict(state["model"], strict=True)
        model = model.to(device)
        optimizer = make_optimizer(model, training)
        if state is not None:
            optimizer.load_state_dict(state["optimizer"])
        epoch_start = state["next_epoch"] if state else 0
        cursor = state["next_update_in_epoch"] if state else 0
        step = state["step"] if state else 0
        best_f1 = state["best_f1"] if state else -1.0
        best_ce = state["best_ce"] if state else None
        best_epoch = state["best_epoch"] if state else 0
        best_checkpoint = state["best_checkpoint"] if state else None
        patience = state["patience"] if state else 0
        windows_seen = state["windows_seen"] if state else 0
        chars_seen = state["chars_seen"] if state else 0
        if state:
            if best_checkpoint and not Path(best_checkpoint).exists():
                raise ValueError("Resume checkpoint references a missing best model")
            restore_rng(state["rng_states"][rank], device)
        else:
            # Identical initialization, independent per-rank dropout streams.
            seed_everything(training["seed"] + rank)
        wrapped = (
            DDP(model, device_ids=[device.index], broadcast_buffers=False)
            if world > 1
            else model
        )
        del state
        model.train()
        started = time.monotonic()

        def checkpoint(next_epoch, next_update, *, is_best=False):
            nonlocal best_checkpoint
            all_rng = gather_objects(rng_state(device), rank, world, object_group)
            if rank == 0:
                previous_best = best_checkpoint
                if is_best:
                    best_checkpoint = str(output / f"best-epoch-{best_epoch:02d}.pt")
                    atomic_checkpoint(
                        best_checkpoint,
                        {
                            "format": FORMAT,
                            "identity": ident,
                            "model": model.state_dict(),
                            "step": step,
                            "epoch": best_epoch,
                            "best_f1": best_f1,
                            "best_ce": best_ce,
                            "model_config": asdict(model.model_config),
                            "head_config": asdict(model.head_config),
                            "adaptation_report": getattr(
                                model, "adaptation_report", None
                            ),
                            "loss_metadata": loss_metadata,
                        },
                    )
                payload = {
                    "format": FORMAT,
                    "identity": ident,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_states": all_rng,
                    "step": step,
                    "next_epoch": next_epoch,
                    "next_update_in_epoch": next_update,
                    "best_f1": best_f1,
                    "best_ce": best_ce,
                    "best_epoch": best_epoch,
                    "best_checkpoint": best_checkpoint,
                    "patience": patience,
                    "windows_seen": windows_seen,
                    "chars_seen": chars_seen,
                    "model_config": asdict(model.model_config),
                    "head_config": asdict(model.head_config),
                    "loss_metadata": loss_metadata,
                    "adaptation_report": getattr(model, "adaptation_report", None),
                }
                atomic_checkpoint(latest_path, payload)
                # Keep best+latest, removing only an obsolete model from this stage.
                if previous_best and previous_best != best_checkpoint:
                    Path(previous_best).unlink(missing_ok=True)
            best_checkpoint = broadcast_object(
                best_checkpoint, rank, world, object_group
            )
            if world > 1:
                dist.barrier()

        log_event(
            log_path,
            {
                "event": "resume" if latest_path.exists() else "start",
                "stage": stage,
                "world_size": world,
                "step": step,
                "windows": len(dataset.windows),
                "epochs": epochs,
                "schedule_epochs": schedule_epochs,
                "lr_schedule_total_steps": total_steps,
                "steps_per_epoch": steps_per_epoch,
                "micro_batch_per_gpu": micro,
                "gradient_accumulation_steps": accum,
                "parameters": model.count_parameters(),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "adaptation_report": getattr(model, "adaptation_report", None),
                "supervised_duplication": False,
                "loss_metadata": loss_metadata,
                "span_metadata": dataset.span_metadata,
            },
            rank,
        )
        if step == 0:
            checkpoint(0, 0)
        finished_early = (
            stage == "holdout"
            and epoch_start >= training.get("min_epochs", 6)
            and patience >= training.get("patience", 5)
        )
        epoch = epoch_start - 1
        epoch_range = () if finished_early else range(epoch_start, epochs)
        for epoch in epoch_range:
            epoch_loss = 0.0
            epoch_objective_loss = 0.0
            epoch_normalizer = 0.0
            epoch_chars = 0
            epoch_norms = []
            epoch_role_loss = 0.0
            epoch_role_entities = 0
            epoch_ranking_loss = 0.0
            epoch_ranking_comparisons = 0
            epoch_span_loss = 0.0
            epoch_span_comparisons = 0
            for update, group in enumerate(
                distributed_batches(
                    dataset, epoch, training["seed"], micro, accum, world
                )
            ):
                if epoch == epoch_start and update < cursor:
                    continue
                cpu_batches = []
                local_real_windows = 0
                for slots in group:
                    rank_slots = slots[rank]
                    batch = collate_windows([dataset.windows[i] for i, _ in rank_slots])
                    for row, (_, supervised) in enumerate(rank_slots):
                        if not supervised:
                            batch["loss_mask"][row].zero_()
                            if "role_entity_mask" in batch:
                                batch["role_entity_mask"][
                                    batch["role_entities"][:, 0] == row
                                ] = False
                            batch["rank_comparison_mask"][
                                batch["rank_comparisons"][:, 0] == row
                            ] = False
                            batch["span_comparison_mask"][
                                batch["span_comparisons"][:, 0] == row
                            ] = False
                        else:
                            local_real_windows += 1
                    cpu_batches.append(batch)
                count = global_owned_count(cpu_batches, device, world)
                entity_count = (
                    global_entity_count(cpu_batches, device, world)
                    if role_enabled
                    else 0
                )
                comparison_count = global_comparison_count(cpu_batches, device, world)
                span_count = global_span_count(cpu_batches, device, world)
                normalizer = float(count)
                if count == 0:
                    continue
                optimizer.zero_grad(set_to_none=True)
                factor = learning_rate_factor(
                    step, total_steps, training["warmup_fraction"]
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["initial_lr"] * factor
                local_loss = torch.zeros((), device=device, dtype=torch.float64)
                local_raw_loss = torch.zeros((), device=device, dtype=torch.float64)
                local_aux_loss = torch.zeros((), device=device, dtype=torch.float64)
                local_role_loss = torch.zeros((), device=device, dtype=torch.float64)
                local_ranking_loss = torch.zeros((), device=device, dtype=torch.float64)
                local_span_loss = torch.zeros((), device=device, dtype=torch.float64)
                update_started = time.monotonic()
                for mi, cpu_batch in enumerate(cpu_batches):
                    batch = move_batch(cpu_batch, device)
                    sync = (
                        wrapped.no_sync()
                        if world > 1 and mi + 1 < len(cpu_batches)
                        else nullcontext()
                    )
                    with sync:
                        with amp(device):
                            outputs = wrapped(batch)
                            mask = batch["loss_mask"] & (batch["labels"] != -100)
                            # Main BIO CE stays unweighted. The auxiliary start/end
                            # BCE uses the same global owned-character denominator.
                            loss, _, _, raw_loss, auxiliary_loss = summed_loss(
                                outputs,
                                batch["labels"],
                                mask,
                                loss_weights,
                                batch["boundary_targets"],
                                lambda_boundary=lambda_boundary,
                            )
                            normalized = loss * world / normalizer
                            if role_enabled:
                                role_loss, _ = summed_role_loss(
                                    outputs["logits"],
                                    batch["role_entities"],
                                    batch["role_entity_mask"],
                                )
                                normalized = (
                                    normalized
                                    + lambda_role
                                    * role_loss
                                    * world
                                    / max(1, entity_count)
                                )
                            else:
                                role_loss = outputs["logits"].sum() * 0.0
                            ranking_loss, _ = summed_ranking_loss(
                                outputs["boundary_logits"],
                                batch["rank_comparisons"],
                                batch["rank_comparison_mask"],
                            )
                            normalized = (
                                normalized
                                + lambda_ranking
                                * ranking_loss
                                * world
                                / max(1, comparison_count)
                            )
                            span_loss, _ = summed_span_loss(
                                outputs["logits"],
                                batch["span_comparisons"],
                                batch["span_comparison_mask"],
                            )
                            normalized = (
                                normalized
                                + lambda_span * span_loss * world / max(1, span_count)
                            )
                        require_finite(loss, device, world, "character loss")
                        require_finite(role_loss, device, world, "entity role loss")
                        require_finite(
                            ranking_loss, device, world, "boundary ranking loss"
                        )
                        require_finite(span_loss, device, world, "direct BIO span loss")
                        require_finite(
                            normalized, device, world, "complete normalized objective"
                        )
                        normalized.backward()
                    local_loss += loss.detach().double()
                    local_raw_loss += raw_loss.detach().double()
                    local_aux_loss += auxiliary_loss.detach().double()
                    local_role_loss += role_loss.detach().double()
                    local_ranking_loss += ranking_loss.detach().double()
                    local_span_loss += span_loss.detach().double()
                    # Release the large character features and device batch
                    # before the next accumulated forward. Detached scalar
                    # totals above are the only values needed for reporting.
                    del outputs, batch, normalized, span_loss, role_loss
                    del ranking_loss, loss, raw_loss, auxiliary_loss
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training["gradient_clip"]
                )
                require_finite(norm, device, world, "gradient norm")
                norm_value = float(norm)
                optimizer.step()
                step += 1
                totals = sum_all(
                    [
                        float(local_loss),
                        float(local_raw_loss),
                        local_real_windows,
                        float(local_aux_loss),
                        float(local_role_loss),
                        float(local_ranking_loss),
                    ],
                    device,
                    world,
                )
                span_total = sum_all([float(local_span_loss)], device, world)[0]
                epoch_span_loss += span_total
                epoch_span_comparisons += span_count
                extra_objective = lambda_span * span_total / max(1, span_count)
                windows_seen += int(totals[2])
                chars_seen += count
                epoch_objective_loss += totals[0]
                epoch_normalizer += normalizer
                epoch_loss += totals[1]
                epoch_chars += count
                epoch_norms.append(norm_value)
                epoch_role_loss += totals[4]
                epoch_role_entities += entity_count
                epoch_ranking_loss += totals[5]
                epoch_ranking_comparisons += comparison_count
                if step <= 2 or step % training.get("log_every_steps", 5) == 0:
                    torch.cuda.synchronize(device)
                    elapsed_update = time.monotonic() - update_started
                    log_event(
                        log_path,
                        {
                            "event": "train",
                            "stage": stage,
                            "epoch": epoch + 1,
                            "update_in_epoch": update + 1,
                            "step": step,
                            "character_ce": totals[1] / count,
                            "joint_objective": (
                                totals[0] / normalizer
                                + lambda_role * totals[4] / max(1, entity_count)
                                + lambda_ranking * totals[5] / max(1, comparison_count)
                                + extra_objective
                            ),
                            "auxiliary_boundary_bce": totals[3] / normalizer,
                            "lambda_boundary": lambda_boundary,
                            "loss_normalizer": normalizer,
                            "lambda_role": lambda_role,
                            "role_entities": entity_count,
                            "auxiliary_role_ce": totals[4] / entity_count
                            if entity_count
                            else None,
                            "lambda_ranking": lambda_ranking,
                            "ranking_comparisons": comparison_count,
                            "auxiliary_boundary_ranking": totals[5] / comparison_count
                            if comparison_count
                            else None,
                            "lambda_span": lambda_span,
                            "span_comparisons": span_count,
                            "direct_bio_span_ranking": span_total / span_count
                            if span_count
                            else None,
                            "owned_characters": count,
                            "characters_seen": chars_seen,
                            "windows_seen": windows_seen,
                            "pre_clip_gradient_norm": norm_value,
                            "gradient_clipped": norm_value > training["gradient_clip"],
                            "backbone_lr": training["backbone_lr"] * factor,
                            "head_lr": training["head_lr"] * factor,
                            "update_seconds": elapsed_update,
                            "characters_per_second": count / elapsed_update,
                            "peak_reserved_gib": torch.cuda.max_memory_reserved(device)
                            / 2**30,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        rank,
                    )
                if step % training.get("save_every_steps", 50) == 0:
                    checkpoint(epoch, update + 1)
                if (
                    stop_after_steps is not None
                    and step >= stop_after_steps
                    and update + 1 < steps_per_epoch
                ):
                    checkpoint(epoch, update + 1)
                    log_event(
                        log_path, {"event": "diagnostic_stop", "step": step}, rank
                    )
                    return
            cursor = 0
            epoch_extra_objective = (
                lambda_span * epoch_span_loss / max(1, epoch_span_comparisons)
            )
            epoch_extra_report = {
                "epoch_span_comparisons": epoch_span_comparisons,
                "epoch_direct_bio_span_ranking": epoch_span_loss
                / epoch_span_comparisons
                if epoch_span_comparisons
                else None,
            }
            if validation is not None:
                _, evaluation = evaluate_model(
                    model,
                    validation,
                    device,
                    rank,
                    world,
                    object_group,
                    training["eval_batch_per_gpu"],
                    prediction_policy=config.get("prediction_policy", "raw"),
                )
                ce = evaluation["character_cross_entropy"]
                f1 = evaluation["metrics"]["strict_macro_f1"]
                require_finite(
                    torch.tensor([ce, f1], device=device),
                    device,
                    world,
                    "validation result",
                )
                better = f1 > best_f1 + 1e-12 or (
                    abs(f1 - best_f1) <= 1e-12 and (best_ce is None or ce < best_ce)
                )
                if better:
                    best_f1, best_ce, best_epoch = f1, ce, epoch + 1
                    patience = 0
                else:
                    patience += 1
                log_event(
                    log_path,
                    {
                        "event": "validation",
                        "stage": stage,
                        "epoch": epoch + 1,
                        "step": step,
                        **evaluation,
                        "best_epoch": best_epoch,
                        "best_f1": best_f1,
                        "patience_used": patience,
                        "epoch_training_ce": epoch_loss / epoch_chars
                        if epoch_chars
                        else None,
                        "epoch_joint_objective": (
                            epoch_objective_loss / epoch_normalizer
                            + lambda_role
                            * epoch_role_loss
                            / max(1, epoch_role_entities)
                            + lambda_ranking
                            * epoch_ranking_loss
                            / max(1, epoch_ranking_comparisons)
                            + epoch_extra_objective
                        )
                        if epoch_normalizer
                        else None,
                        **epoch_extra_report,
                        "epoch_role_entities": epoch_role_entities,
                        "epoch_role_ce": epoch_role_loss / epoch_role_entities
                        if epoch_role_entities
                        else None,
                        "epoch_ranking_comparisons": epoch_ranking_comparisons,
                        "epoch_boundary_ranking": epoch_ranking_loss
                        / epoch_ranking_comparisons
                        if epoch_ranking_comparisons
                        else None,
                        "epoch_median_pre_clip_norm": statistics.median(epoch_norms)
                        if epoch_norms
                        else None,
                        "epoch_clipped_fraction": sum(
                            n > training["gradient_clip"] for n in epoch_norms
                        )
                        / len(epoch_norms)
                        if epoch_norms
                        else None,
                    },
                    rank,
                )
                if rank == 0:
                    atomic_json(
                        output / f"validation-epoch-{epoch+1:02d}.json", evaluation
                    )
                checkpoint(epoch + 1, 0, is_best=better)
                if epoch + 1 >= training.get(
                    "min_epochs", 6
                ) and patience >= training.get("patience", 5):
                    finished_early = True
                    break
            else:
                log_event(
                    log_path,
                    {
                        "event": "refit_epoch",
                        "epoch": epoch + 1,
                        "step": step,
                        "epoch_training_ce": epoch_loss / epoch_chars
                        if epoch_chars
                        else None,
                        "epoch_joint_objective": (
                            epoch_objective_loss / epoch_normalizer
                            + lambda_role
                            * epoch_role_loss
                            / max(1, epoch_role_entities)
                            + lambda_ranking
                            * epoch_ranking_loss
                            / max(1, epoch_ranking_comparisons)
                            + epoch_extra_objective
                        )
                        if epoch_normalizer
                        else None,
                        **epoch_extra_report,
                        "epoch_role_entities": epoch_role_entities,
                        "epoch_role_ce": epoch_role_loss / epoch_role_entities
                        if epoch_role_entities
                        else None,
                        "epoch_ranking_comparisons": epoch_ranking_comparisons,
                        "epoch_boundary_ranking": epoch_ranking_loss
                        / epoch_ranking_comparisons
                        if epoch_ranking_comparisons
                        else None,
                    },
                    rank,
                )
                checkpoint(epoch + 1, 0)
            if stop_after_steps is not None and step >= stop_after_steps:
                log_event(
                    log_path,
                    {
                        "event": "diagnostic_stop",
                        "step": step,
                        "completed_epoch": epoch + 1,
                    },
                    rank,
                )
                return
        if rank == 0:
            final_path = None
            if stage == "refit":
                final_path = str(output / "final.pt")
                atomic_checkpoint(
                    final_path,
                    {
                        "format": FORMAT,
                        "identity": ident,
                        "model": model.state_dict(),
                        "step": step,
                        "epoch": epochs,
                        "model_config": asdict(model.model_config),
                        "head_config": asdict(model.head_config),
                        "adaptation_report": getattr(model, "adaptation_report", None),
                        "loss_metadata": loss_metadata,
                    },
                )
            result = {
                "completed": True,
                "stage": stage,
                "finished_at_utc": now(),
                "step": step,
                "epochs_completed": epoch + 1,
                "early_stopped": finished_early,
                "schedule_epochs": schedule_epochs,
                "lr_schedule_total_steps": total_steps,
                "best_epoch": best_epoch if stage == "holdout" else epochs,
                "best_f1": best_f1 if stage == "holdout" else None,
                "best_ce": best_ce,
                "best_checkpoint": best_checkpoint,
                "final_checkpoint": final_path,
                "elapsed_seconds": time.monotonic() - started,
                "windows_seen": windows_seen,
                "characters_seen": chars_seen,
                "identity": ident,
            }
            atomic_json(output / "result.json", result)
    finally:
        close()


def worker_command(config_path, world, *args):
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world}",
        "-m",
        "shared.s7_train",
        "--config",
        str(config_path),
        *args,
    ]


def supervisor(config_path, no_refit=False, stop_after_steps=None):
    config = load_json(config_path)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    world = config["training"]["world_size"]
    status_path = output / "pipeline-status.json"

    def run(stage, cmd):
        atomic_json(
            status_path,
            {
                "state": "running",
                "stage": stage,
                "updated_at_utc": now(),
                "command": cmd,
                "config_path": str(config_path),
            },
        )
        with open(output / f"{stage}.log", "a", encoding="utf-8") as log:
            subprocess.run(
                cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
            )

    holdout_result = output / "holdout" / "result.json"
    if not holdout_result.exists():
        extra = ["--stage", "holdout"]
        if stop_after_steps is not None:
            extra += ["--stop-after-steps", str(stop_after_steps)]
        run("holdout-training", worker_command(config_path, world, *extra))
    if stop_after_steps is not None and not holdout_result.exists():
        atomic_json(
            status_path,
            {
                "state": "diagnostic_complete",
                "stage": "holdout",
                "updated_at_utc": now(),
            },
        )
        return
    selected = load_json(holdout_result)
    expected_holdout_identity = identity(
        config, load_cache(config["cache_path"], "train"), "holdout", world
    )
    if selected["identity"] != expected_holdout_identity:
        raise ValueError(
            "Existing holdout result has different training configuration or data"
        )
    if not selected["completed"]:
        raise ValueError("Holdout training did not complete")
    if no_refit or stop_after_steps is not None:
        atomic_json(
            status_path,
            {
                "state": "holdout_complete",
                "stage": "holdout",
                "updated_at_utc": now(),
                "best_macro_f1": selected["best_f1"],
                "best_checkpoint": selected["best_checkpoint"],
            },
        )
        return
    refit_result = output / "refit" / "result.json"
    if not refit_result.exists():
        run("all-data-refit", worker_command(config_path, world, "--stage", "refit"))
    refit = load_json(refit_result)
    expected_refit_identity = identity(
        config, load_cache(config["cache_path"], "all_train"), "refit", world
    )
    if refit["identity"] != expected_refit_identity:
        raise ValueError(
            "Existing refit result has different training configuration or data"
        )
    if not refit["completed"]:
        raise ValueError("Full-data refit did not complete")
    submission = output / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    prediction = submission / "prediction.json"
    run(
        "test-inference",
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={world}",
            "-m",
            "shared.s7_predict",
            "--config",
            str(config_path),
            "--checkpoint",
            refit["final_checkpoint"],
            "--partition",
            "test",
            "--output",
            str(prediction),
        ],
    )
    predictions = load_json(prediction)
    zip_path = submission / "submission.zip"
    temporary = submission / ".submission.tmp.zip"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(prediction, arcname="prediction.json")
    os.replace(temporary, zip_path)
    atomic_json(
        status_path,
        {
            "state": "complete",
            "stage": "submission-ready",
            "updated_at_utc": now(),
            "submission_zip": str(zip_path),
            "documents": len(predictions),
            "entities": sum(map(len, predictions.values())),
            "holdout_best_macro_f1": selected["best_f1"],
            "selected_refit_epochs": selected["best_epoch"],
            "portal_uploaded": False,
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", choices=("holdout", "refit"))
    ap.add_argument("--stop-after-steps", type=int)
    ap.add_argument("--no-refit", action="store_true")
    args = ap.parse_args()
    if args.stop_after_steps is not None and args.stop_after_steps < 1:
        ap.error("--stop-after-steps must be positive")
    if args.stage:
        train_stage(load_json(args.config), args.stage, args.stop_after_steps)
    else:
        supervisor(Path(args.config).resolve(), args.no_refit, args.stop_after_steps)


if __name__ == "__main__":
    main()
