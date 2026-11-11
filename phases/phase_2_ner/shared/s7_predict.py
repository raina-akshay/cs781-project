"""Full-document character BIO inference.

Auxiliary losses are training-only. Prediction uses full-document BIO decoding
and the selected leading-whitespace policy.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .ner_contract import score
from .ner_data import load_cache, collate_windows, decode_bio, bio_to_spans
from .span_boundary_policy import apply_policy
from .s7_runtime import (
    amp,
    atomic_json,
    broadcast_object,
    close,
    gather_objects,
    identity,
    load_checkpoint,
    move_batch,
    now,
    setup,
    sum_all,
)


@torch.no_grad()
def evaluate_model(
    model,
    dataset,
    device,
    rank,
    world,
    object_group,
    batch_size,
    prediction_policy="raw",
):
    """Partition windows without replication, then decode complete original documents.

    Only owned positions contribute logits/CE. Exactly one owner is required for
    every nonempty document character, including whitespace and punctuation.
    """
    was_training = model.training
    model.eval()
    try:
        rows = []
        loss_sum = 0.0
        count = 0
        local = list(range(rank, len(dataset.windows), world))
        for offset in range(0, len(local), batch_size):
            windows = [dataset.windows[i] for i in local[offset : offset + batch_size]]
            batch = move_batch(collate_windows(windows), device)
            with amp(device):
                logits = model(batch)["logits"]
            if not bool(torch.isfinite(logits).all().item()):
                raise FloatingPointError("Nonfinite inference logits")
            owned = batch["loss_mask"]
            supervised = (
                (owned & (batch["labels"] != -100))
                if dataset.gold is not None
                else torch.zeros_like(owned)
            )
            if supervised.any():
                loss_sum += float(
                    F.cross_entropy(
                        logits.float()[supervised],
                        batch["labels"][supervised],
                        reduction="sum",
                    )
                )
                count += int(supervised.sum())
            logits_cpu = logits.float().cpu().numpy()
            for row, window in enumerate(windows):
                positions = np.flatnonzero(np.asarray(window.owned_mask, dtype=bool))
                rows.append(
                    (
                        window.document_id,
                        window.char_start + positions,
                        logits_cpu[row, positions].copy(),
                    )
                )
        totals = sum_all([loss_sum, count], device, world)
        gathered = gather_objects(rows, rank, world, object_group)
        result = None
        assembly_error = None
        if rank == 0:
            try:
                scores = {
                    did: np.zeros((len(text), 25), dtype=np.float32)
                    for did, text in dataset.texts.items()
                }
                coverage = {
                    did: np.zeros(len(text), dtype=np.uint8)
                    for did, text in dataset.texts.items()
                }
                for shard in gathered:
                    for did, positions, values in shard:
                        if np.any(coverage[did][positions]):
                            raise ValueError(
                                f"Duplicated character ownership at inference for {did}"
                            )
                        scores[did][positions] = values
                        coverage[did][positions] += 1
                predictions = {}
                for did, text in dataset.texts.items():
                    if len(text) and not np.all(coverage[did] == 1):
                        raise ValueError(
                            f"Missing character coverage at inference for {did}"
                        )
                    predictions[did] = (
                        bio_to_spans(
                            decode_bio(scores[did], dataset.labels), dataset.labels
                        )
                        if text
                        else []
                    )
                raw_metrics = (
                    score(dataset.texts, dataset.gold, predictions, dataset.labels)
                    if dataset.gold is not None
                    else None
                )
                predictions, policy_changes = apply_policy(
                    dataset.texts, predictions, dataset.labels, prediction_policy
                )
                result = {
                    "prediction": predictions,
                    "characters": int(sum(map(len, dataset.texts.values()))),
                    "owned_supervised_characters": int(totals[1]),
                    "character_cross_entropy": totals[0] / totals[1]
                    if totals[1]
                    else None,
                    "predicted_entities": sum(map(len, predictions.values())),
                    "evaluated_at_utc": now(),
                    "prediction_policy": prediction_policy,
                    "policy_changes": policy_changes,
                }
                if dataset.gold is not None:
                    result["metrics"] = score(
                        dataset.texts, dataset.gold, predictions, dataset.labels
                    )
                    result["raw_metrics"] = raw_metrics
            except Exception as exc:
                assembly_error = repr(exc)
        assembly_error = broadcast_object(assembly_error, rank, world, object_group)
        if assembly_error is not None:
            raise ValueError("Distributed document assembly failed: " + assembly_error)
        # Only metrics are broadcast; each rank does not need all document predictions.
        summary = (
            None
            if result is None
            else {k: v for k, v in result.items() if k != "prediction"}
        )
        summary = broadcast_object(summary, rank, world, object_group)
        return result, summary

    finally:
        model.train(was_training)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--partition",
        choices=("train", "validation", "all_train", "test"),
        default="test",
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text())
    rank, world, device, group = setup(config["training"]["world_size"])
    try:
        from .s7_model import build_model

        dataset = load_cache(config["cache_path"], args.partition)
        checkpoint = load_checkpoint(args.checkpoint)
        checkpoint_identity = checkpoint["identity"]
        if checkpoint_identity != identity(
            config, dataset, checkpoint_identity["stage"], world
        ):
            raise ValueError(
                "Prediction training configuration or data differs from the checkpoint"
            )
        model = build_model(
            config, dataset.char_vocab, dataset.labels, load_foundation=False
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        model = model.to(device)
        result, _ = evaluate_model(
            model,
            dataset,
            device,
            rank,
            world,
            group,
            config["training"]["eval_batch_per_gpu"],
            prediction_policy=config.get("prediction_policy", "raw"),
        )
        if rank == 0:
            atomic_json(args.output, result["prediction"])
            atomic_json(
                str(args.output) + ".report.json",
                {k: v for k, v in result.items() if k != "prediction"},
            )
    finally:
        close()


if __name__ == "__main__":
    main()
