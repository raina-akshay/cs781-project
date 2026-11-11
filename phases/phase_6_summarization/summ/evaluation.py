"""Full-population generation and document-mean scoring on three disjoint shards."""
import argparse
from datetime import timedelta
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from .common import atomic_json, now, read_json, validate_predictions
from .data import Cache, Formatter, own_tokenizer
from .generation import generate_batch
from .metrics import SummaryScorer, mean_metrics
from .model import build_model
from .runtime import configure_cuda, restore_delta, seed_all


def evaluate(model, split, formatter, config, scorer, rank, world, device, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    order = sorted(range(len(split)), key=lambda i: (-len(split.source_ids(i)), i))
    indices = order[rank::world]
    batch_size = config["training"]["eval_batch_cases"]
    references = None
    if split.name != "test":
        references = {
            r["id"]: r["summary"]
            for r in read_json(
                Path(config["data_manifest"]).parent / (split.name + ".references.json")
            )
        }
    rows = []
    started = time.monotonic()
    model.eval()
    for begin in range(0, len(indices), batch_size):
        selected = indices[begin : begin + batch_size]
        generated = generate_batch(
            model, [split.source_ids(i) for i in selected], formatter, config["generation"], device
        )
        for i, result in zip(selected, generated):
            rows.append({"index": i, "id": split.ids[i], **result})
        atomic_json(
            out / ("rank-" + str(rank) + "-status.json"),
            {
                "at_utc": now(),
                "state": "generating",
                "split": split.name,
                "done": len(rows),
                "total": len(indices),
                "elapsed_seconds": time.monotonic() - started,
                "generated_tokens": sum(r["generated_tokens"] for r in rows),
                "rank": rank,
            },
        )
    if references is not None:
        scores = scorer.score([r["summary"] for r in rows], [references[r["id"]] for r in rows])
        for row, metric in zip(rows, scores):
            row["metrics"] = metric
    atomic_json(
        out / ("rank-" + str(rank) + "-status.json"),
        {
            "at_utc": now(),
            "state": "completed",
            "split": split.name,
            "done": len(rows),
            "total": len(indices),
            "elapsed_seconds": time.monotonic() - started,
            "rank": rank,
        },
    )
    gathered = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, rows)
    else:
        gathered = [rows]
    merged = sorted([r for part in gathered for r in part], key=lambda r: r["index"])
    predictions = [{"id": r["id"], "summary": r["summary"]} for r in merged]
    validate_predictions(predictions, split.ids)
    assessment = {
        "split": split.name,
        "documents": len(split),
        "max_new_token_hits": sum(r["stop"] == "max_new_tokens" for r in merged),
        "eos_stops": sum(r["stop"] == "eos" for r in merged),
        "mean_generated_tokens": sum(r["generated_tokens"] for r in merged) / len(merged),
    }
    if references is not None:
        assessment.update(mean_metrics([r["metrics"] for r in merged]))
    if rank == 0:
        atomic_json(out / "prediction.json", predictions)
        # Generation timing varies by replay. Token IDs, stops and metrics are semantic evidence.
        semantic = [{k: v for k, v in r.items() if k != "elapsed_batch_seconds"} for r in merged]
        atomic_json(out / "generation.json", semantic)
        atomic_json(out / "assessment.json", assessment)
        atomic_json(
            out / "timing.json", {"at_utc": now(), "elapsed_seconds": time.monotonic() - started}
        )
    if world > 1:
        dist.barrier()
    return assessment, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test", "train"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    configure_cuda()
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=6))
    try:
        seed_all(42)
        cache = Cache(config["data_manifest"])
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = build_model(config).to(device)
        restore_delta(model, saved)
        tokenizer = own_tokenizer(config["foundation_manifest"])
        formatter = Formatter(tokenizer, config["sequence"])
        scorer = None if args.split == "test" else SummaryScorer(config["metric_snapshot"], device)
        assessment, _ = evaluate(
            model,
            cache.splits[args.split],
            formatter,
            config,
            scorer,
            rank,
            world,
            device,
            args.output,
        )
        if rank == 0:
            print(assessment, flush=True)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
