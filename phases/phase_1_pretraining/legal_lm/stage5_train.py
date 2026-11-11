"""Train the foundation from scratch on deduplicated next-token windows.

Loss is normalized by the number of supervised targets across all workers.
"""
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .io import atomic_json, sha256_file
from .model import LegalLanguageModel
from .train import (
    EpochSampler,
    _atomic_checkpoint,
    _sum,
    amp_context,
    choose_precision,
    load_checkpoint,
    lr_for_tokens,
    masked_cross_entropy_sum,
    restore_rng,
    rng_state,
    seed_everything,
    setup_distributed,
    validate_lm,
)


class SourceQuotaSampler:
    """Deterministic shuffled source queues balanced by physical target debt.

    Every rank plans the same global rows. A source's deficit is its desired
    physical-target share minus targets actually retained so far. Selection
    has whole-window granularity, so quotas are approximate, never claimed
    exact. Final-budget masking is included in these counters. Queue cursors
    advance only for genuinely selected rows, not zero-target DDP fillers.
    """

    def __init__(self, datasets, source_targets, seed):
        self.datasets = datasets
        self.sources = next(iter(datasets.values())).sources
        raw = np.asarray(source_targets, dtype=np.float64)
        if raw.shape != (len(self.sources),) or not np.isfinite(raw).all() or (raw <= 0).any():
            raise ValueError("Original source target quotas must all be positive")
        self.fractions = raw / raw.sum()
        self.source_targets = [0] * len(self.sources)
        self.queues = {}
        for component, dataset in datasets.items():
            if dataset.sources != self.sources:
                raise ValueError("Dataset source order differs between components")
            if (np.asarray(dataset.valid_counts) <= 0).any():
                raise ValueError("Every training row must contain a valid target")
            queues = []
            for source, indices in enumerate(dataset.indices_by_source):
                if len(indices) == 0:
                    raise ValueError(
                        f"No {component} training rows for source {self.sources[source]}"
                    )
                # Ordinary queue seed is identical in all arms; task queues
                # are independent of ordinary queues and identical for C/D.
                component_seed = 0 if component == "ordinary" else 1_000_003
                queues.append(EpochSampler(len(indices), seed + component_seed + source * 1009))
            self.queues[component] = queues

    def take_global_batch(self, component, size, remaining_targets):
        dataset = self.datasets[component]
        rows = []
        for _ in range(size):
            if remaining_targets == 0:
                rows.append((0, 0))  # A differentiable zero-loss DDP filler.
                continue
            total = sum(self.source_targets)
            debt = self.fractions * total - np.asarray(self.source_targets)
            source = int(np.argmax(debt))
            ordinal = self.queues[component][source].take(1)[0]
            index = int(dataset.indices_by_source[source][ordinal])
            count = min(int(dataset.valid_counts[index]), remaining_targets)
            rows.append((index, count))
            self.source_targets[source] += count
            remaining_targets -= count
        return rows

    def state_dict(self):
        return {
            "format_version": 1,
            "fractions": self.fractions.tolist(),
            "source_targets": list(self.source_targets),
            "positions": {
                component: [queue.position for queue in queues]
                for component, queues in self.queues.items()
            },
        }

    def load_state_dict(self, state):
        if (
            state.get("format_version") != 1
            or state.get("fractions") != self.fractions.tolist()
            or set(state.get("positions", {})) != set(self.queues)
        ):
            raise ValueError("Saved source sampler identity differs")
        counts = state.get("source_targets")
        if (
            not isinstance(counts, list)
            or len(counts) != len(self.sources)
            or any(type(n) is not int or n < 0 for n in counts)
        ):
            raise ValueError("Invalid saved source target counts")
        for component, queues in self.queues.items():
            positions = state["positions"][component]
            if len(positions) != len(queues) or any(type(n) is not int or n < 0 for n in positions):
                raise ValueError("Invalid saved source queue positions")
            for queue, position in zip(queues, positions):
                queue.position = position
                queue._epoch, queue._order = None, None
        self.source_targets = list(counts)


def _batch(dataset, rows, trim_padding):
    items = [dataset[index] for index, _ in rows]
    batch = {
        key: torch.stack([item[key] for item in items]).clone()
        for key in ("input_ids", "labels", "weights", "continuation_mask")
    }
    batch["source_id"] = torch.tensor([int(item["source_id"]) for item in items])
    # These compute counters include inputs even when final-budget clipping
    # removes their labels. Attention work still processed those positions.
    batch["input_tokens"] = int((batch["input_ids"] != dataset.pad_id).sum())
    if trim_padding:
        used = (
            ((batch["input_ids"] != dataset.pad_id) | (batch["labels"] != -100))
            .any(dim=0)
            .nonzero()
            .flatten()
        )
        width = int(used[-1]) + 1 if len(used) else 1
        for key in ("input_ids", "labels", "weights", "continuation_mask"):
            batch[key] = batch[key][:, :width].contiguous()
    for row, (_, keep) in enumerate(rows):
        valid = (batch["labels"][row] != -100).nonzero().flatten()
        removed = valid[keep:]
        batch["labels"][row, removed] = -100
        batch["weights"][row, removed] = 0
        batch["continuation_mask"][row, removed] = False
    batch["processed_positions"] = batch["input_ids"].numel()
    return batch


def train_scratch(
    config,
    view_dir,
    run_dir,
    *,
    resume=None,
    device_name="cuda",
    max_steps=None,
    allow_small_model=False,
    allow_cpu=False,
):
    rank, world, device = setup_distributed(device_name)
    try:
        return _train(
            config,
            Path(view_dir),
            Path(run_dir),
            resume,
            device,
            rank,
            world,
            max_steps,
            allow_small_model,
            allow_cpu,
        )
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def _train(
    config, view_dir, run_dir, resume, device, rank, world, max_steps, allow_small_model, allow_cpu
):
    from .data import TokenWindowDataset
    from .stage3_data import Stage3Dataset

    cfg = config.training
    dataset = Stage3Dataset(view_dir, component="ordinary", deduplicate=True, verify=False)
    manifest = dataset.token_manifest
    token_dir = Path(dataset.token_dir)
    data_fingerprint = sha256_file(token_dir / "manifest.json")
    tokenizer_json = Path(manifest["tokenizer_path"]).read_text(encoding="utf-8")
    if (
        dataset.manifest["token_manifest_sha256"] != data_fingerprint
        or config.model.context_length != dataset.context_length
        or config.model.vocab_size != manifest["vocab_size"]
    ):
        raise ValueError("Model or prepared view differs from original token provenance")
    precision, dtype = choose_precision(cfg.precision, device)
    seed_everything(cfg.seed)
    model = LegalLanguageModel(config.model).to(device)
    parameters = model.count_parameters()
    model.gradient_checkpointing = cfg.gradient_checkpointing
    decay, no_decay = [], []
    for parameter in model.parameters():
        (
            decay
            if parameter.ndim >= 2 and not getattr(parameter, "_no_weight_decay", False)
            else no_decay
        ).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=cfg.betas,
        eps=cfg.adam_epsilon,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16", init_scale=4096.0)
    sampler = SourceQuotaSampler(
        {"ordinary": dataset},
        [dataset.manifest["source_target_tokens"][s] for s in dataset.sources],
        cfg.seed,
    )
    identity = {
        "config": config.to_dict(),
        "data_fingerprint": data_fingerprint,
        "world_size": world,
        "precision": precision,
        "diagnostic_only": bool(allow_small_model),
        "solution5": {
            "format_version": 1,
            "initialization": "random_scratch",
            "objective": "ordinary_next_token_global_physical_target_mean",
            "training_view": "exact_dedup_ordinary",
            "view_fingerprint": dataset.manifest["fingerprint"],
            "source_policy": "largest_actual_physical_target_deficit_then_source_id",
            "sample_order_policy": "global_microbatch_then_rank_then_row",
            "device_type": device.type,
            "rng_policy": "seed_plus_rank",
        },
    }
    checkpoint = load_checkpoint(resume) if resume else None
    if checkpoint:
        previous = checkpoint["run_identity"]
        same_run = all(
            previous[key] == identity[key]
            for key in ("config", "data_fingerprint", "world_size", "precision")
        )
        same_view = all(
            previous["solution5"][key] == value for key, value in identity["solution5"].items()
        )
        if not same_run or not same_view or checkpoint["tokenizer_json"] != tokenizer_json:
            raise ValueError(
                "Resume requires the same model, data, tokenizer and training settings"
            )
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=bool(resume))
    if world > 1:
        dist.barrier()
    source_count = len(dataset.sources)
    state = {
        "step": 0,
        "tokens_seen": 0,
        "input_tokens": 0,
        "processed_positions": 0,
        "source_targets": [0] * source_count,
        "best_lm_loss": None,
    }
    last_metrics = None
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        saved = checkpoint["training_state"]
        state = {key: saved[key] for key in state}
        sampler.load_state_dict(saved["source_sampler"])
        if (
            state["source_targets"] != sampler.source_targets
            or sum(state["source_targets"]) != state["tokens_seen"]
            or not 0 <= state["tokens_seen"] <= cfg.max_tokens
            or len(checkpoint["rng_states"]) != world
        ):
            raise ValueError("Saved physical-target/source/RNG counters are inconsistent")
        last_metrics = checkpoint.get("last_lm_evaluation")
    wrapped = (
        DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
        if world > 1
        else model
    )
    if checkpoint:
        restore_rng(checkpoint["rng_states"][rank], device)
    else:
        seed_everything(cfg.seed + rank)
    del checkpoint
    dev_data = TokenWindowDataset(token_dir, "dev", config.model.context_length)
    if len(dev_data) == 0:
        raise ValueError("Original development windows are empty")
    if rank == 0:
        atomic_json(
            run_dir / "run.json",
            {
                "run_identity": identity,
                "parameter_count": parameters,
                "view_dir": str(view_dir.resolve()),
                "token_dir": str(token_dir.resolve()),
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU",
            },
        )
    start_tokens, start_step, start_time = state["tokens_seen"], state["step"], time.monotonic()
    last_save = start_time

    def status(value):
        if rank == 0:
            atomic_json(
                run_dir / "status.json",
                {
                    "status": value,
                    "step": state["step"],
                    "tokens_seen": state["tokens_seen"],
                    "target_budget": cfg.max_tokens,
                    "architecture": config.model.architecture,
                    "diagnostic_only": bool(allow_small_model),
                    "checkpoint": str((run_dir / "latest.pt").resolve()),
                },
            )

    def save(name):
        nonlocal last_save
        all_rng = [None] * world
        if world > 1:
            dist.all_gather_object(all_rng, rng_state(device))
        else:
            all_rng[0] = rng_state(device)
        if rank == 0:
            _atomic_checkpoint(
                run_dir / name,
                {
                    "format_version": 1,
                    "model_config": asdict(config.model),
                    "model": model.state_dict(),
                    "parameter_count": parameters,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "rng_states": all_rng,
                    "run_identity": identity,
                    "token_manifest": manifest,
                    "tokenizer_json": tokenizer_json,
                    "token_dir": str(token_dir.resolve()),
                    "training_state": {
                        **state,
                        "source_sampler": sampler.state_dict(),
                        "target_scope": "physical_supervised_occurrences",
                        "position_exposure_policy": "no_global_unique_claim",
                    },
                    "last_lm_evaluation": last_metrics,
                },
            )
        if world > 1:
            dist.barrier()
        last_save = time.monotonic()

    model.train()
    status("running")
    while state["tokens_seen"] < cfg.max_tokens and (
        max_steps is None or state["step"] - start_step < max_steps
    ):
        remaining = cfg.max_tokens - state["tokens_seen"]
        desired = min(cfg.tokens_per_update, remaining)
        batches, count = [], 0
        while count < desired:
            plan = sampler.take_global_batch(
                "ordinary", cfg.micro_batch_size * world, remaining - count
            )
            count += sum(keep for _, keep in plan)
            lo = rank * cfg.micro_batch_size
            batches.append(_batch(dataset, plan[lo : lo + cfg.micro_batch_size], cfg.trim_padding))
        lr = lr_for_tokens(state["tokens_seen"] + count, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        local_loss = 0.0
        for index, batch in enumerate(batches):
            sync = wrapped.no_sync() if world > 1 and index < len(batches) - 1 else nullcontext()
            with sync:
                ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
                with amp_context(device, dtype):
                    # Right suffix padding cannot influence valid causal positions.
                    # Cache/state is not retained between independent rows.
                    loss, _ = masked_cross_entropy_sum(wrapped(ids).logits, labels)
                finite = torch.tensor(int(torch.isfinite(loss)), device=device)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not int(finite):
                    raise FloatingPointError("Nonfinite training loss; no optimizer update applied")
                local_loss += float(loss.detach())
                scaler.scale(loss * (world / count)).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        finite = torch.tensor(int(torch.isfinite(norm)), device=device)
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not int(finite):
            raise FloatingPointError("Nonfinite gradients; optimizer did not advance")
        scaler.step(optimizer)
        scaler.update()
        sources = [0] * source_count
        for batch in batches:
            for row, source in enumerate(batch["source_id"].tolist()):
                sources[source] += int((batch["labels"][row] != -100).sum())
        totals = _sum(
            [
                local_loss,
                sum(b["input_tokens"] for b in batches),
                sum(b["processed_positions"] for b in batches),
                *sources,
            ],
            device,
            world,
        )
        previous = state["tokens_seen"]
        state["tokens_seen"] += count
        state["step"] += 1
        state["input_tokens"] += int(totals[1])
        state["processed_positions"] += int(totals[2])
        state["source_targets"] = [a + int(b) for a, b in zip(state["source_targets"], totals[3:])]
        step = state["step"]
        if step % cfg.log_every_steps == 0 or step == start_step + 1:
            if rank == 0:
                row = {
                    **state,
                    "source_targets": dict(zip(dataset.sources, state["source_targets"])),
                    "loss": totals[0] / count,
                    "global_target_tokens": count,
                    "learning_rate": lr,
                    "grad_norm_before_clip": float(norm),
                    "tokens_per_second_this_invocation": (state["tokens_seen"] - start_tokens)
                    / (time.monotonic() - start_time),
                    "microbatches": len(batches),
                }
                if device.type == "cuda":
                    row["rank0_cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                    row["rank0_cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                with (run_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row), flush=True)
            status("running")
        final = state["tokens_seen"] == cfg.max_tokens or (
            max_steps is not None and step - start_step >= max_steps
        )
        milestones = [n for n in cfg.checkpoint_tokens if previous < n <= state["tokens_seen"]]
        if step % cfg.eval_every_steps == 0 or state["tokens_seen"] == cfg.max_tokens or milestones:
            last_metrics = validate_lm(
                model,
                dev_data,
                cfg.eval_max_tokens,
                device,
                dtype,
                rank,
                world,
                trim_padding=cfg.trim_padding,
            )
            if not math.isfinite(last_metrics["loss"]):
                raise FloatingPointError("Nonfinite development loss")
            state["best_lm_loss"] = min(
                state["best_lm_loss"] if state["best_lm_loss"] is not None else math.inf,
                last_metrics["loss"],
            )
            if rank == 0:
                with (run_dir / "lm-evaluations.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {"step": step, "tokens_seen": state["tokens_seen"], **last_metrics},
                            allow_nan=False,
                        )
                        + "\n"
                    )
        for milestone in milestones:
            save(f"candidate-{milestone:012d}.pt")
        time_save = torch.tensor(
            int(time.monotonic() - last_save >= cfg.save_every_minutes * 60), device=device
        )
        if world > 1:
            dist.all_reduce(time_save, op=dist.ReduceOp.MAX)
        if step % cfg.save_every_steps == 0 or int(time_save) or final:
            save("latest.pt")
    if state["tokens_seen"] == start_tokens:
        save("latest.pt")
    result = "complete" if state["tokens_seen"] == cfg.max_tokens else "invocation_limit"
    status(result)
    return {
        "status": result,
        "step": state["step"],
        "tokens_seen": state["tokens_seen"],
        "parameter_count": parameters,
        "run_dir": str(run_dir),
        "diagnostic_only": bool(allow_small_model),
        "checkpoint": str((run_dir / "latest.pt").resolve()),
    }
