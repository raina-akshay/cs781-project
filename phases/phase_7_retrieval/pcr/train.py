"""Complete three-GPU PCR training, DEV selection and verified final export."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
import torch
import torch.distributed as dist

from .data import ChunkStore, load_split, read_json, verify_cache
from .evaluation import evaluate_ranking, predictions_from_ranking, rank_scores
from .lexical import TrainBM25
from .model import build_model
from .objective import retrieval_loss_scores, sampled_candidates
from .scoring import document_vectors, score_components
from .optim import build_optimizer
from .inference import validate_identity, encode_pool, score_pool
from .runtime import (
    trainable_state,
    atomic_json,
    atomic_torch,
    backward_cached,
    barrier,
    digest,
    document_chunks,
    encode_nograd,
    gather_objects,
    now,
    restore_trainable,
    restore_rng,
    rng_state,
    seed_all,
    setup_distributed,
    source_identity,
    summed_gradients,
)

ROOT = Path(__file__).resolve().parents[1]


def fused_ranking(dense, lexical, raw, alpha, rank_constant):
    dense_rank = rank_scores(dense, raw.query_ids, raw.candidate_ids)
    if alpha == 1:
        return dense_rank
    lex_rank = rank_scores(lexical, raw.query_ids, raw.candidate_ids)
    rows = np.arange(len(raw.query_ids))[:, None]
    dr = np.full(dense.shape, dense.shape[1] + 1, dtype=np.float64)
    lr = dr.copy()
    places = np.arange(dense_rank.shape[1])[None, :] + 1
    dr[rows, dense_rank] = places
    lr[rows, lex_rank] = places
    scores = alpha / (rank_constant + dr) + (1 - alpha) / (rank_constant + lr)
    return rank_scores(scores, raw.query_ids, raw.candidate_ids)


def evaluate(model, raw, store, lexical, config, rank, world, device, output):
    output.mkdir(parents=True, exist_ok=True)
    encode_pool(model, store, config, rank, world, device, output)
    components = score_pool(raw, store, config, rank, world, device, output)
    selection = None
    if rank == 0:
        dense = components["joint"]
        recipes = []
        for alpha in config["retrieval"]["dense_weights"]:
            ranking = fused_ranking(
                dense, lexical, raw, alpha, config["retrieval"]["rrf_constant"]
            )
            recipes.append(
                {
                    "dense_weight": alpha,
                    "metrics": evaluate_ranking(
                        ranking, raw.query_ids, raw.candidate_ids, raw.labels
                    ),
                }
            )
        selection = max(recipes, key=lambda x: (x["metrics"]["tp"], x["dense_weight"]))
        ranking = fused_ranking(
            dense,
            lexical,
            raw,
            selection["dense_weight"],
            config["retrieval"]["rrf_constant"],
        )
        atomic_json(
            output / "prediction.json",
            predictions_from_ranking(ranking, raw.query_ids, raw.candidate_ids),
        )
        ablations = {
            name: evaluate_ranking(
                rank_scores(scores, raw.query_ids, raw.candidate_ids),
                raw.query_ids,
                raw.candidate_ids,
                raw.labels,
            )
            for name, scores in components.items()
        }
        atomic_json(
            output / "selection.json",
            {
                "at": now(),
                "selected": selection,
                "recipes": recipes,
                "component_diagnostics": ablations,
                "scope": "DEV only; global/local components are diagnostics, not selection candidates",
            },
        )
    if world > 1:
        values = [selection]
        dist.broadcast_object_list(values, src=0)
        selection = values[0]
    return selection


def train_step(
    model,
    raw,
    store,
    query_indices,
    epoch,
    config,
    dense_rankings,
    lexical_rankings,
    device,
    global_count,
):
    if not query_indices:
        return 0.0, 0, 0, []
    tr = config["training"]
    candidate_indices, links = sampled_candidates(
        raw, query_indices, epoch, tr, dense_rankings, lexical_rankings
    )
    query_ids = [raw.query_ids[i] for i in query_indices]
    all_indices = sorted(
        set(candidate_indices) | {store.id_to_index[q] for q in query_ids}
    )
    position = {idx: i for i, idx in enumerate(all_indices)}
    chunks, owners = document_chunks(store, all_indices)
    cached = encode_nograd(model, chunks, tr["microbatch_chunks"], device)
    leaf = cached.detach().requires_grad_(True)
    docs, masks = document_vectors(
        leaf, owners, len(all_indices), config["representation"]["max_chunks"]
    )
    qpos = [position[store.id_to_index[q]] for q in query_ids]
    cpos = [position[c] for c in candidate_indices]
    scores = score_components(
        docs[qpos],
        masks[qpos],
        docs[cpos],
        masks[cpos],
        config["retrieval"]["local_weight"],
    )["joint"]
    candidate_ids = [raw.candidate_ids[c] for c in candidate_indices]
    loss = retrieval_loss_scores(
        scores, query_ids, candidate_ids, raw.labels, config["retrieval"]["temperature"]
    )
    scaled = loss / global_count
    gradients = torch.autograd.grad(scaled, leaf)[0].detach()
    if not bool(torch.isfinite(gradients).all()):
        raise FloatingPointError("Nonfinite cached derivative")
    loss_value = float(loss.detach())
    del leaf, docs, masks, scores, scaled, loss, cached
    backward_cached(model, chunks, gradients, tr["microbatch_chunks"], device)
    return loss_value, len(chunks), sum(map(len, chunks)), links


def mine_negatives(
    model, raw, store, config, rank, world, device, run, identity, source_epoch
):
    if raw.split != "train" or raw.labels is None:
        raise ValueError("Mining must use TRAIN only")
    output = run / f"mining/source-epoch-{source_epoch:03d}"
    if rank == 0:
        atomic_json(
            run / "status.json",
            {
                "stage": "train_negative_mining",
                "at": now(),
                "source_epoch": source_epoch,
            },
        )
    barrier()
    encode_pool(model, store, config, rank, world, device, output)
    parts = score_pool(raw, store, config, rank, world, device, output)
    ref = None
    if rank == 0:
        ranking = rank_scores(parts["joint"], raw.query_ids, raw.candidate_ids)
        np.save(output / "ranking.npy", ranking, allow_pickle=False)
        meta = {
            "at": now(),
            "source_epoch": source_epoch,
            "identity_sha256": identity["sha256"],
            "ranking_sha256": digest(output / "ranking.npy"),
            "query_ids": raw.query_ids,
            "candidate_ids": raw.candidate_ids,
            "policy": "Current own model joint score, all TRAIN candidates; positive/self filtering at sampling; no DEV/TEST labels",
        }
        atomic_json(output / "manifest.json", meta)
        ref = {
            "directory": str(output.relative_to(run)),
            "source_epoch": source_epoch,
        }
        print(
            json.dumps(
                {
                    "event": "mining_complete",
                    "source_epoch": source_epoch,
                    "reference": ref,
                }
            ),
            flush=True,
        )
    if world > 1:
        values = [ref]
        dist.broadcast_object_list(values, src=0)
        ref = values[0]
    return ref


def load_mining(run, reference, raw, identity, expected_source_epoch):
    if raw.split != "train" or raw.labels is None:
        raise ValueError("Mining state must belong to TRAIN")
    if reference is None or reference["source_epoch"] != expected_source_epoch:
        raise ValueError("Mining epoch mismatch")
    relative = Path(reference["directory"])
    output = run / relative
    meta = read_json(output / "manifest.json")
    if (
        meta["source_epoch"] != expected_source_epoch
        or meta["identity_sha256"] != identity["sha256"]
        or meta["query_ids"] != raw.query_ids
        or meta["candidate_ids"] != raw.candidate_ids
    ):
        raise ValueError("Mining source/ID identity differs")
    ranking = np.load(output / "ranking.npy", allow_pickle=False)
    if (
        ranking.shape != (len(raw.query_ids), len(raw.candidate_ids) - 1)
        or ranking.dtype.kind not in "iu"
    ):
        raise ValueError("Mining rank shape differs")
    cpos = {c: i for i, c in enumerate(raw.candidate_ids)}
    expected = np.arange(len(raw.candidate_ids))
    for i, q in enumerate(raw.query_ids):
        if not np.array_equal(np.sort(ranking[i]), expected[expected != cpos[q]]):
            raise ValueError("Mining row is not the full nonself candidate permutation")
    return ranking


def save_state(
    path,
    model,
    optimizer,
    rank,
    world,
    identity,
    epoch,
    next_batch,
    steps,
    best,
    stale,
    loss_sum,
    query_count,
    elapsed_seconds,
    positive_links,
    mining,
    completed=False,
):
    all_rng = gather_objects(rng_state(), world)
    all_links = gather_objects(sorted(positive_links), world)
    if rank == 0:
        union = set(tuple(x) for part in all_links for x in part)
        atomic_torch(
            path,
            {
                "version": 1,
                "identity": identity,
                "trainable": trainable_state(model),
                "optimizer": optimizer.state_dict(),
                "rng_by_rank": all_rng,
                "world_size": world,
                "epoch": epoch,
                "next_batch": next_batch,
                "steps": steps,
                "best": best,
                "stale": stale,
                "loss_sum": loss_sum,
                "query_count": query_count,
                "elapsed_seconds": elapsed_seconds,
                "positive_links": sorted(union),
                "mining": mining,
                "training_completed": completed,
                "saved_at": now(),
            },
        )
    barrier()


def verify_positive_coverage(train, positive_links, run, rank, world):
    all_links = gather_objects(sorted(positive_links), world)
    used = set(tuple(x) for part in all_links for x in part)
    expected = {(q, c) for q, v in train.labels.items() for c in v}
    if used != expected:
        raise ValueError("Training ended without complete positive-pair exposure")
    if rank == 0:
        atomic_json(
            run / "positive-coverage.json",
            {
                "all_known_train_positive_pairs": len(expected),
                "sampled_unique_pairs": len(used),
                "complete": True,
                "zero_positive_queries_excluded_from_positive_loss": len(
                    train.query_ids
                )
                - len(train.supervised_query_indices),
            },
        )
    barrier()


def write_prediction_archive(path, prediction_path):
    """Byte-identical archive for unchanged prediction bytes across re-exports."""
    path = Path(path)
    temporary = path.with_suffix(".zip.tmp")
    info = zipfile.ZipInfo("prediction.json", date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(temporary, "w") as archive:
        archive.writestr(
            info,
            Path(prediction_path).read_bytes(),
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        )
    temporary.replace(path)


def export_best(
    model,
    run,
    raw_root,
    cache,
    lexical_dir,
    dev_lex,
    config,
    rank,
    world,
    device,
    identity,
):
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    if best["identity"] != identity:
        raise ValueError("Best checkpoint source/config identity differs")
    # Independently reconstruct the original immutable foundation before replay.
    # Release its training copy from GPU first so publication has no double-base
    # VRAM spike. The caller no longer trains after final export.
    model.cpu()
    torch.cuda.empty_cache()
    model = build_model(config).to(device)
    restore_trainable(model, best["trainable"])
    dev = load_split(raw_root, "dev")
    dstore = ChunkStore(cache / "dev/candidate")
    replay = evaluate(
        model,
        dev,
        dstore,
        dev_lex,
        config,
        rank,
        world,
        device,
        run / "final/dev-replay",
    )
    expected = best["selection"]
    if replay != expected:
        raise ValueError(
            "Selected checkpoint DEV replay did not reproduce metrics/recipe"
        )
    if rank == 0:
        original = read_json(run / f"epochs/epoch-{best['epoch']:03d}/prediction.json")
        reconstructed = read_json(run / "final/dev-replay/prediction.json")
        if original != reconstructed:
            raise ValueError("DEV prediction replay differs despite metric match")
        atomic_json(
            run / "status.json",
            {
                "stage": "test_inference",
                "at": now(),
                "best_epoch": best["epoch"],
                "best_dev": expected,
            },
        )
    test = load_split(raw_root, "test")
    tstore = ChunkStore(cache / "test/candidate")
    validate_identity(test, tstore)
    encode_pool(model, tstore, config, rank, world, device, run / "final/test")
    components = score_pool(
        test, tstore, config, rank, world, device, run / "final/test"
    )
    if rank == 0:
        lexical = TrainBM25.load(lexical_dir / "model").score(
            test.query_texts, test.candidate_texts
        )
        dense = components["joint"]
        ranking = fused_ranking(
            dense,
            lexical,
            test,
            expected["dense_weight"],
            config["retrieval"]["rrf_constant"],
        )
        pred = predictions_from_ranking(ranking, test.query_ids, test.candidate_ids)
        output = run / "final/prediction.json"
        atomic_json(output, pred)
        archive = run / "final/submission.zip"
        write_prediction_archive(archive, output)
        import importlib.util

        verifier = ROOT / "scripts/validate_submission.py"
        spec = importlib.util.spec_from_file_location(
            "p7_independent_verifier", verifier
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        verification = mod.validate_artifact(
            archive, set(test.query_ids), set(test.candidate_ids)
        )
        if verification["self_match_query_ids"]:
            raise ValueError("Self-match violates frozen protocol")
        verification.update(
            {
                "at": now(),
                "selected_epoch": best["epoch"],
                "dev": expected,
                "checkpoint_sha256": digest(run / "best.pt"),
                "identity": identity,
                "dev_predictions_exactly_replayed": True,
                "official_dev_metrics_exactly_replayed": True,
                "scorer": "half global cosine plus half query-mean chunk MaxSim",
            }
        )
        atomic_json(run / "final/verification.json", verification)
        destination = ROOT / "submission.zip"
        if destination.exists() and digest(destination) != digest(archive):
            raise FileExistsError(
                "Do not overwrite a different previously delivered submission"
            )
        shutil.copyfile(archive, destination)
        atomic_json(
            run / "status.json",
            {
                "stage": "completed_verified",
                "at": now(),
                "best_epoch": best["epoch"],
                "best_dev": expected,
                "submission_sha256": digest(destination),
                "submission": str(destination),
            },
        )
        print(
            json.dumps(
                {
                    "event": "completed_verified",
                    "best_epoch": best["epoch"],
                    "best_dev": expected,
                    "submission": str(destination),
                }
            ),
            flush=True,
        )
    barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline.json")
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/run-1")
    parser.add_argument("--export-best", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    run = args.run.resolve()
    manifest_path = Path(config["foundation_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    config["foundation_manifest"] = str(manifest_path.resolve())
    run.mkdir(parents=True, exist_ok=True)
    rank, world, device = setup_distributed()
    tr = config["training"]
    seed_all(tr["seed"])
    if world != 3:
        raise ValueError(
            "Production protocol requires three GPUs; diagnostics use separate tests"
        )
    prepared_path = ROOT / "artifacts/prepared/prepared.json"
    prepared = read_json(prepared_path)
    cache = Path(prepared["chunk_cache"])
    lexical_dir = Path(prepared["lexical_directory"])
    cache_manifest = verify_cache(cache)
    raw_manifest = read_json(ROOT / "data/raw/source-manifest.json")
    expected_sources = {
        x["repository_path"]: x["sha256"] for x in raw_manifest["files"]
    }
    recipe = cache_manifest["recipe"]
    foundation_manifest = read_json(config["foundation_manifest"])
    if (
        recipe["sources"] != expected_sources
        or recipe["tokenizer_sha256"] != foundation_manifest["tokenizer_sha256"]
        or recipe["chunk_length"] != config["representation"]["chunk_length"]
        or recipe["max_chunks"] != config["representation"]["max_chunks"]
        or recipe["special_tokens"] != {"bos_id": 1, "eos": False}
    ):
        raise ValueError(
            "Prepared representation or data identity differs from fixed protocol"
        )
    lexical_manifest = read_json(lexical_dir / "manifest.json")
    if lexical_manifest["raw_manifest_sha256"] != digest(
        ROOT / "data/raw/source-manifest.json"
    ):
        raise ValueError("Lexical data source identity differs")
    for file, expected in lexical_manifest["files"].items():
        if digest(lexical_dir / file) != expected:
            raise ValueError("Lexical provenance differs: " + file)
    identity = source_identity(ROOT, config, prepared_path)
    identity.update(
        {
            "chunk_manifest": digest(cache / "manifest.json"),
            "lexical_manifest": digest(lexical_dir / "manifest.json"),
            "foundation_manifest_sha256": digest(config["foundation_manifest"]),
            "foundation_checkpoint_sha256": foundation_manifest["checkpoint_sha256"],
            "tokenizer_sha256": foundation_manifest["tokenizer_sha256"],
        }
    )
    identity.pop("sha256")
    identity["sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    raw_root = ROOT / "data/raw"
    train = load_split(raw_root, "train")
    dev = load_split(raw_root, "dev")
    store = ChunkStore(cache / "train/candidate")
    dstore = ChunkStore(cache / "dev/candidate")
    validate_identity(train, store)
    validate_identity(dev, dstore)
    train_ids = read_json(lexical_dir / "train-ids.json")
    dev_ids = read_json(lexical_dir / "dev-ids.json")
    for raw, ids in ((train, train_ids), (dev, dev_ids)):
        if ids != {"query_ids": raw.query_ids, "candidate_ids": raw.candidate_ids}:
            raise ValueError("Lexical ID order differs")
    hard = np.load(lexical_dir / "train-ranking.npy", allow_pickle=False)
    dev_lex = np.load(lexical_dir / "dev-scores.npy", allow_pickle=False)
    model = build_model(config).to(device)
    model.train()
    optimizer = build_optimizer(model, config)
    if (run / "identity.json").exists() and read_json(
        run / "identity.json"
    ) != identity:
        raise ValueError("Existing run identity differs; preserve its receipt")
    if rank == 0:
        atomic_json(run / "identity.json", identity)
        atomic_json(
            run / "model.json",
            {
                "foundation": model.foundation_report,
                "adaptation": model.adaptation_report,
                "representation": model.representation_report,
                "optimizer_groups": optimizer.pcr_group_report,
            },
        )
    if args.export_best:
        saved = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
        if not saved["training_completed"] or saved["identity"] != identity:
            raise ValueError(
                "--export-best is final re-export only; training must be complete with matching identity"
            )
        verify_positive_coverage(
            train, set(map(tuple, saved["positive_links"])), run, rank, world
        )
        del saved
        export_best(
            model,
            run,
            raw_root,
            cache,
            lexical_dir,
            dev_lex,
            config,
            rank,
            world,
            device,
            identity,
        )
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    epoch, next_batch, steps, stale = 1, 0, 0, 0
    best = None
    loss_sum = 0.0
    query_count = 0
    previous_elapsed = 0.0
    positive_links = set()
    completed = False
    mining = None
    if (run / "latest.pt").exists():
        saved = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
        if saved["identity"] != identity or saved["world_size"] != world:
            raise ValueError("Resume identity/world mismatch")
        restore_trainable(model, saved["trainable"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng_by_rank"][rank])
        epoch = saved["epoch"]
        next_batch = saved["next_batch"]
        steps = saved["steps"]
        best = saved["best"]
        stale = saved["stale"]
        loss_sum = saved["loss_sum"]
        query_count = saved["query_count"]
        previous_elapsed = saved["elapsed_seconds"]
        positive_links = set(map(tuple, saved["positive_links"]))
        completed = saved["training_completed"]
        mining = saved["mining"]
        del saved
        n_queries = len(train.supervised_query_indices)
        n_batches = math.ceil(n_queries / tr["global_query_batch"])
        if (
            not 1 <= epoch <= tr["max_epochs"] + 1
            or not 0 <= next_batch <= n_batches
            or steps != (epoch - 1) * n_batches + next_batch
            or query_count != min(next_batch * tr["global_query_batch"], n_queries)
            or not math.isfinite(loss_sum)
            or loss_sum < 0
            or not math.isfinite(previous_elapsed)
            or previous_elapsed < 0
            or (best is not None and not 1 <= best["epoch"] < epoch)
        ):
            raise ValueError("Malformed saved training cursor or metrics")
        if completed and (
            next_batch != 0
            or best is None
            or not (
                epoch - 1 >= tr["max_epochs"]
                or (epoch - 1 >= tr["minimum_epochs"] and stale >= tr["patience"])
            )
        ):
            raise ValueError(
                "Saved completion does not satisfy the declared stopping rule"
            )
    elif (run / "best.pt").exists():
        raise ValueError("Best exists without resumable latest state")
    clock_start = time.monotonic()

    def elapsed():
        return previous_elapsed + time.monotonic() - clock_start

    if completed:
        verify_positive_coverage(train, positive_links, run, rank, world)
        export_best(
            model,
            run,
            raw_root,
            cache,
            lexical_dir,
            dev_lex,
            config,
            rank,
            world,
            device,
            identity,
        )
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    if steps == 0 and next_batch == 0 and epoch == 1:
        if rank == 0:
            atomic_json(
                run / "status.json",
                {"stage": "original_foundation_dev_control", "at": now()},
            )
        control = evaluate(
            model,
            dev,
            dstore,
            dev_lex,
            config,
            rank,
            world,
            device,
            run / "control-original",
        )
        if rank == 0:
            print(
                json.dumps(
                    {"event": "original_foundation_control", "selection": control}
                ),
                flush=True,
            )
    eligible = np.asarray(train.supervised_query_indices, dtype=np.int64)
    batches = math.ceil(len(eligible) / tr["global_query_batch"])
    max_steps = batches * tr["max_epochs"]
    warmup = batches * tr["warmup_epochs"]
    while epoch <= tr["max_epochs"]:
        epoch_clock = time.monotonic()
        source_epoch = ((epoch - 1) // tr["mining_interval_epochs"]) * tr[
            "mining_interval_epochs"
        ]
        if mining is None or mining["source_epoch"] != source_epoch:
            if next_batch != 0 or source_epoch != epoch - 1:
                raise ValueError(
                    "Missing or stale mining state outside its refresh boundary"
                )
            mining = mine_negatives(
                model,
                train,
                store,
                config,
                rank,
                world,
                device,
                run,
                identity,
                source_epoch,
            )
        dense_hard = load_mining(run, mining, train, identity, source_epoch)
        order = np.random.default_rng(tr["seed"] + epoch).permutation(eligible).tolist()
        for batch in range(next_batch, batches):
            batch_clock = time.monotonic()
            global_indices = order[
                batch
                * tr["global_query_batch"] : (batch + 1)
                * tr["global_query_batch"]
            ]
            local = global_indices[rank::world]
            progress = min(1.0, max(0.0, (steps - warmup) / max(1, max_steps - warmup)))
            ratio = (
                (steps + 1) / warmup
                if steps < warmup
                else tr["minimum_lr_ratio"]
                + (1 - tr["minimum_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2
            )
            for group in optimizer.param_groups:
                group["lr"] = tr["learning_rate"] * ratio
            optimizer.zero_grad(set_to_none=True)
            value, chunks, tokens, links = train_step(
                model,
                train,
                store,
                local,
                epoch,
                config,
                dense_hard,
                hard,
                device,
                len(global_indices),
            )
            positive_links.update(links)
            summed_gradients(model)
            gradnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                tr["max_gradient_norm"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            steps += 1
            counts = torch.tensor(
                [value, chunks, tokens], device=device, dtype=torch.float64
            )
            if world > 1:
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            loss_sum += float(counts[0])
            query_count += len(global_indices)
            if rank == 0:
                status = {
                    "stage": "training",
                    "at": now(),
                    "epoch": epoch,
                    "max_epochs": tr["max_epochs"],
                    "batch": batch + 1,
                    "batches_per_epoch": batches,
                    "optimizer_steps": steps,
                    "batch_loss": float(counts[0]) / len(global_indices),
                    "epoch_loss": loss_sum / query_count,
                    "gradient_norm_before_clip": float(gradnorm),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "global_queries": len(global_indices),
                    "chunks": int(counts[1]),
                    "input_tokens_one_pass": int(counts[2]),
                    "step_seconds": time.monotonic() - batch_clock,
                    "elapsed_seconds": elapsed(),
                    "best": best,
                    "physical_train_chunks": tr["microbatch_chunks"],
                    "mining_source_epoch": mining["source_epoch"],
                }
                atomic_json(run / "status.json", status)
                print(json.dumps(status), flush=True)
            if steps % tr["checkpoint_steps"] == 0:
                save_state(
                    run / "latest.pt",
                    model,
                    optimizer,
                    rank,
                    world,
                    identity,
                    epoch,
                    batch + 1,
                    steps,
                    best,
                    stale,
                    loss_sum,
                    query_count,
                    elapsed(),
                    positive_links,
                    mining,
                )
        if rank == 0:
            atomic_json(
                run / "status.json",
                {
                    "stage": "dev_evaluation",
                    "at": now(),
                    "epoch": epoch,
                    "best": best,
                    "elapsed_seconds": elapsed(),
                },
            )
        selection = evaluate(
            model,
            dev,
            dstore,
            dev_lex,
            config,
            rank,
            world,
            device,
            run / f"epochs/epoch-{epoch:03d}",
        )
        improvement = (
            best is None
            or selection["metrics"]["tp"] > best["selection"]["metrics"]["tp"]
        )
        better = best is None or (
            selection["metrics"]["tp"],
            selection["dense_weight"],
        ) > (best["selection"]["metrics"]["tp"], best["selection"]["dense_weight"])
        stale = 0 if improvement else stale + 1
        if better:
            best = {"epoch": epoch, "selection": selection}
            if rank == 0:
                atomic_torch(
                    run / "best.pt",
                    {
                        "trainable": trainable_state(model),
                        "identity": identity,
                        "epoch": epoch,
                        "selection": selection,
                        "at": now(),
                    },
                )
        if rank == 0:
            recipes = read_json(run / f"epochs/epoch-{epoch:03d}/selection.json")[
                "recipes"
            ]
            dense = next(x for x in recipes if x["dense_weight"] == 1)
            previous_dense = (
                read_json(run / "best-joint.json")
                if (run / "best-joint.json").exists()
                else None
            )
            if (
                previous_dense is None
                or dense["metrics"]["tp"] > previous_dense["selection"]["metrics"]["tp"]
            ):
                atomic_torch(
                    run / "best-joint.pt",
                    {
                        "trainable": trainable_state(model),
                        "identity": identity,
                        "epoch": epoch,
                        "selection": dense,
                        "at": now(),
                    },
                )
                atomic_json(
                    run / "best-joint.json", {"epoch": epoch, "selection": dense}
                )
            atomic_json(
                run / f"epochs/epoch-{epoch:03d}/training.json",
                {
                    "epoch": epoch,
                    "loss": loss_sum / query_count,
                    "queries": query_count,
                    "seconds": time.monotonic() - epoch_clock,
                    "elapsed_seconds": elapsed(),
                    "steps": steps,
                    "selection": selection,
                    "best": best,
                    "stale_primary_epochs": stale,
                    "mining": mining,
                },
            )
            print(
                json.dumps(
                    {
                        "event": "dev_complete",
                        "epoch": epoch,
                        "selection": selection,
                        "best": best,
                        "stale": stale,
                    }
                ),
                flush=True,
            )
        completed = epoch >= tr["max_epochs"] or (
            epoch >= tr["minimum_epochs"] and stale >= tr["patience"]
        )
        save_state(
            run / "latest.pt",
            model,
            optimizer,
            rank,
            world,
            identity,
            epoch + 1,
            0,
            steps,
            best,
            stale,
            0.0,
            0,
            elapsed(),
            positive_links,
            mining,
            completed=completed,
        )
        if completed:
            break
        epoch += 1
        next_batch = 0
        loss_sum = 0.0
        query_count = 0
    verify_positive_coverage(train, positive_links, run, rank, world)
    export_best(
        model,
        run,
        raw_root,
        cache,
        lexical_dir,
        dev_lex,
        config,
        rank,
        world,
        device,
        identity,
    )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
