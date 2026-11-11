"""Full-pool distributed chunk encoding and exactly shared retrieval scoring."""
from pathlib import Path
import time
import numpy as np
import torch
from .runtime import atomic_json, barrier, document_chunks, encode_nograd, now
from .scoring import document_vectors, score_components


def validate_identity(raw, store):
    if raw.candidate_ids != store.ids:
        raise ValueError("Candidate cache ID order differs")
    texts = dict(zip(raw.candidate_ids, raw.candidate_texts))
    for q, text in zip(raw.query_ids, raw.query_texts):
        if texts.get(q) != text:
            raise ValueError("Query reuse requires identical candidate source: " + q)


def encode_pool(model, store, config, rank, world, device, output):
    """Write deterministic per-rank shards; avoid GPU all-gather of entire pool."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    indices = list(range(rank, len(store.ids), world))
    width = config["representation"]["max_chunks"]
    dim = model.output_dim
    arrays = []
    masks = []
    for offset in range(0, len(indices), 16):
        selected = indices[offset : offset + 16]
        chunks, owners = document_chunks(store, selected)
        vectors = encode_nograd(
            model, chunks, config["training"]["eval_microbatch_chunks"], device
        )
        packed, mask = document_vectors(vectors, owners, len(selected), width)
        arrays.append(packed.cpu().numpy())
        masks.append(mask.cpu().numpy())
        atomic_json(
            output / f"rank-{rank}-progress.json",
            {
                "at": now(),
                "documents": min(offset + 16, len(indices)),
                "total_documents": len(indices),
                "seconds": time.monotonic() - started,
            },
        )
    local = (
        np.concatenate(arrays)
        if arrays
        else np.empty((0, width, dim), dtype=np.float32)
    )
    valid = np.concatenate(masks) if masks else np.empty((0, width), dtype=bool)
    shard = output / f"rank-{rank}.npz"
    tmp = output / f"rank-{rank}.npz.tmp"
    with tmp.open("wb") as f:
        np.savez(
            f, indices=np.asarray(indices, dtype=np.int64), vectors=local, masks=valid
        )
    tmp.replace(shard)
    del arrays, masks, local, valid
    torch.cuda.empty_cache()
    barrier()
    if rank == 0:
        combined = np.empty((len(store.ids), width, dim), dtype=np.float32)
        all_masks = np.zeros((len(store.ids), width), dtype=bool)
        seen = []
        for other in range(world):
            with np.load(output / f"rank-{other}.npz", allow_pickle=False) as data:
                idx = data["indices"]
                combined[idx] = data["vectors"]
                all_masks[idx] = data["masks"]
                seen.extend(idx.tolist())
        if (
            sorted(seen) != list(range(len(store.ids)))
            or not np.isfinite(combined).all()
            or not all_masks.any(1).all()
        ):
            raise ValueError("Candidate encoding coverage/finite check failed")
        np.save(output / "document-chunks.npy", combined, allow_pickle=False)
        np.save(output / "chunk-masks.npy", all_masks, allow_pickle=False)
        atomic_json(
            output / "encoding-timing.json",
            {
                "seconds": time.monotonic() - started,
                "documents": len(store.ids),
                "at": now(),
            },
        )
        for other in range(world):
            (output / f"rank-{other}.npz").unlink()
    barrier()


@torch.no_grad()
def score_pool(raw, store, config, rank, world, device, output):
    """Every query against every candidate; same function as differentiable training."""
    output = Path(output)
    started = time.monotonic()
    vectors = torch.from_numpy(
        np.load(output / "document-chunks.npy", allow_pickle=False)
    ).to(device)
    masks = torch.from_numpy(
        np.load(output / "chunk-masks.npy", allow_pickle=False)
    ).to(device)
    rows = list(range(rank, len(raw.query_ids), world))
    scores = {
        name: np.empty((len(rows), len(store.ids)), dtype=np.float32)
        for name in ["joint", "global", "local"]
    }
    for offset in range(0, len(rows), 8):
        selected = rows[offset : offset + 8]
        qpos = [store.id_to_index[raw.query_ids[i]] for i in selected]
        for start in range(0, len(store.ids), 128):
            parts = score_components(
                vectors[qpos],
                masks[qpos],
                vectors[start : start + 128],
                masks[start : start + 128],
                config["retrieval"]["local_weight"],
            )
            for name, value in parts.items():
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("Nonfinite retrieval score")
                scores[name][
                    offset : offset + len(selected), start : start + len(value[0])
                ] = value.cpu().numpy()
    del vectors, masks
    shard = output / f"scores-rank-{rank}.npz"
    tmp = output / f"scores-rank-{rank}.npz.tmp"
    with tmp.open("wb") as f:
        np.savez(f, indices=np.asarray(rows, dtype=np.int64), **scores)
    tmp.replace(shard)
    del scores
    torch.cuda.empty_cache()
    barrier()
    result = None
    if rank == 0:
        result = {
            name: np.empty((len(raw.query_ids), len(store.ids)), dtype=np.float32)
            for name in ["joint", "global", "local"]
        }
        seen = []
        for other in range(world):
            with np.load(
                output / f"scores-rank-{other}.npz", allow_pickle=False
            ) as data:
                idx = data["indices"]
                seen.extend(idx.tolist())
                for name in result:
                    result[name][idx] = data[name]
        if sorted(seen) != list(range(len(raw.query_ids))) or any(
            not np.isfinite(v).all() for v in result.values()
        ):
            raise ValueError("Distributed score coverage differs")
        for name, value in result.items():
            np.save(output / f"{name}-scores.npy", value, allow_pickle=False)
        for other in range(world):
            (output / f"scores-rank-{other}.npz").unlink()
        atomic_json(
            output / "scoring-timing.json",
            {
                "seconds": time.monotonic() - started,
                "at": now(),
                "query_count": len(raw.query_ids),
                "full_candidate_pool": True,
            },
        )
    barrier()
    return result
