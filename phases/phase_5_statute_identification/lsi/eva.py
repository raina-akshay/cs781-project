"""TRAIN-only Explained Variance Adaptation for the native own backbone.

EVA: Paischer et al., NeurIPS 2025. Incremental PCA is vendored unchanged from
Apache-2.0 PEFT v0.17.1; native orchestration explicitly supplies our chunk
content mask instead of pretending case-level labels are language-model labels.
No external model, dataset, learned label text, or checkpoint is downloaded.
"""
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .adaptation import target_linears
from .vendor.eva.incremental_pca import IncrementalPCA


FORMAT = "cs781-p5-s2-eva-initialization-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_digest(value):
    value = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def content_mask(batch):
    """Accept only lossless BOS/content/EOS/PAD chunk schemas, never labels."""
    if not isinstance(batch, Mapping) or {"labels", "gold", "pred", "targets"} & set(batch):
        raise ValueError("EVA batches must contain TRAIN inputs, not case-level label fields")
    ids, lengths = batch.get("input_ids"), batch.get("lengths")
    if (
        not isinstance(ids, torch.Tensor)
        or ids.ndim != 2
        or ids.dtype != torch.long
        or min(ids.shape) < 1
        or not isinstance(lengths, torch.Tensor)
        or lengths.dtype != torch.long
        or lengths.shape != (ids.shape[0],)
        or lengths.device != ids.device
        or ids.shape[1] < 2
        or bool(((lengths < 2) | (lengths > ids.shape[1])).any())
    ):
        raise ValueError("EVA inputs must be long [chunks,tokens] with matching BOS/EOS lengths")
    positions = torch.arange(ids.shape[1], device=ids.device)[None]
    mask = (positions > 0) & (positions < lengths[:, None] - 1)
    padding = positions >= lengths[:, None]
    eos = ids.gather(1, (lengths - 1)[:, None]).squeeze(1)
    invalid = (mask & ((ids <= 2) | (ids >= 32768))) | (padding & (ids != 0))
    if bool(invalid.any() | (ids[:, 0] != 1).any() | (eos != 2).any()):
        raise ValueError(
            "EVA chunks must retain exactly BOS, ordinary content, EOS and trailing PAD"
        )
    if not bool(mask.any()):
        raise ValueError("An EVA fitting batch needs at least one ordinary content token")
    return mask


def _states(inputs, mask):
    value = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
    result = value.detach()[mask].float()
    if not bool(torch.isfinite(result).all()):
        raise ValueError("EVA observed nonfinite TRAIN activations")
    return result


def allocate_ranks(explained_variance_ratios, target_to_statistics, target_caps, budget):
    """Published global explained-variance selection, deterministic tie breaks."""
    options = []
    for name in sorted(target_caps):
        cap = target_caps[name]
        values = explained_variance_ratios[target_to_statistics[name]]
        for index, value in enumerate(values[:cap].detach().cpu().tolist()):
            options.append((-float(value), name, index))
    chosen = sorted(options)[:budget]
    result = {name: 0 for name in target_caps}
    for _, name, _ in chosen:
        result[name] += 1
    return result


def _validated_batch(batch, allowed_ids):
    mask = content_mask(batch)
    if batch.get("split") != "train":
        raise ValueError("Every EVA fitting batch must explicitly declare split=train")
    ids = batch.get("document_ids")
    if (
        not isinstance(ids, (list, tuple))
        or len(ids) != mask.shape[0]
        or any(
            not isinstance(item, str) or not item.strip() or item not in allowed_ids for item in ids
        )
    ):
        raise ValueError("Every EVA chunk needs its original allowed TRAIN document ID")
    return mask


@torch.no_grad()
def compute_eva_initialization(
    backbone,
    batches,
    forward_fn,
    *,
    rank=16,
    alpha=16.0,
    rho=2.0,
    tau=0.99,
    min_batches=4,
    max_batches=256,
    seed=42,
    train_document_ids,
    data_identity,
    foundation_identity,
    progress_callback=None
):
    """Fit the complete native target closure using TRAIN inputs only.

    `batches` is an iterator factory. Each batch has input_ids,
    lengths, split='train', document_ids (one original TRAIN ID per chunk).
    `forward_fn(backbone,batch)` must run one native chunk microbatch so each
    target is visited once. It chooses the recorded autocast policy itself.

    A finite max budget returns converged=False honestly. Production loading
    rejects that artifact. Statistics are never replaced by random vectors.
    """
    targets = dict(target_linears(backbone))
    devices = {layer.weight.device for layer in targets.values()}
    if len(devices) != 1:
        raise ValueError(
            "Compute EVA in one writer process on one device, then distribute the saved basis"
        )
    device = next(iter(devices))
    caps = {
        name: min(round(rank * rho), layer.in_features, layer.out_features)
        for name, layer in targets.items()
    }
    if any(rank > min(layer.in_features, layer.out_features) for layer in targets.values()):
        raise ValueError("Nominal rank exceeds an original projection dimension")
    allowed_ids = set(train_document_ids)
    iterator = iter(batches())
    first = next(iterator)
    first_mask = _validated_batch(first, allowed_ids)

    def remaining_batches():
        yield from iterator
        while True:
            yield from batches()

    stream = remaining_batches()
    training = backbone.training
    handles, probes, calls = [], {}, {}
    groups, owners, target_to_group = {}, {}, {}
    evidence, seen_ids, seen_set = [], [], set()
    fitting = {"mask": first_mask, "states": {}}
    started = time.perf_counter()
    converged = False
    rank_pattern = None
    backbone.eval()
    try:

        def probe_hook(name):
            def hook(module, inputs, output):
                if name in probes:
                    raise ValueError(
                        "The EVA forward callback must visit every projection exactly once per batch"
                    )
                probes[name] = _states(inputs, first_mask).cpu()

            return hook

        for name, layer in targets.items():
            handles.append(layer.register_forward_hook(probe_hook(name)))
        forward_fn(backbone, first)
        if set(probes) != set(targets):
            raise ValueError("The native fitting forward omitted an intended target")
        for handle in handles:
            handle.remove()
        handles.clear()
        # Hashing is followed by exact tensor equality; no unverified numerical
        # coincidence is silently treated as shared input in later minibatches.
        hashes = {}
        for name in targets:
            digest = _tensor_digest(probes[name])
            owner = next(
                (old for old in hashes.get(digest, []) if torch.equal(probes[name], probes[old])),
                None,
            )
            if owner is None:
                owner = name
                hashes.setdefault(digest, []).append(name)
                owners[owner] = []
            owners[owner].append(name)
            target_to_group[name] = owner
        for owner, aliases in owners.items():
            components = max(caps[name] for name in aliases)
            groups[owner] = {
                "pca": IncrementalPCA(
                    n_components=components,
                    copy=True,
                    lowrank=True,
                    lowrank_q=min(2 * components, targets[owner].in_features),
                    lowrank_seed=seed,
                ),
                "cosines": torch.zeros(components, dtype=torch.float32),
                "updates": 0,
                "pending": [],
                "pending_rows": 0,
            }

        def update(owner, states):
            group = groups[owner]
            group["pending"].append(states.detach())
            group["pending_rows"] += len(states)
            pca = group["pca"]
            if group["pending_rows"] < pca.lowrank_q:
                return
            values = (
                torch.cat(group["pending"], dim=0)
                if len(group["pending"]) > 1
                else group["pending"][0]
            )
            group["pending"], group["pending_rows"] = [], 0
            previous = pca.components_.clone() if hasattr(pca, "components_") else None
            # Hooks run inside the encoder's ambient BF16 autocast. Merely
            # converting states to float() does not prevent matmul autocasting
            # in randomized SVD. Keep the complete statistical update outside
            # autocast, while leaving the native encoder's policy unchanged.
            with torch.autocast(device_type=states.device.type, enabled=False):
                pca.partial_fit(values.float())
                group["updates"] += 1
                if previous is not None:
                    group["cosines"] = (
                        F.cosine_similarity(pca.components_, previous, dim=-1).abs().float()
                    )

        def fitting_hook(name):
            def hook(module, inputs, output):
                calls[name] = calls.get(name, 0) + 1
                if calls[name] != 1:
                    raise ValueError("An EVA target was visited twice in a fitting microbatch")
                states = _states(inputs, fitting["mask"])
                owner = target_to_group[name]
                if name == owner:
                    if len(owners[owner]) > 1:
                        fitting["states"][owner] = states
                    update(owner, states)
                elif owner not in fitting["states"] or not torch.equal(
                    states, fitting["states"][owner]
                ):
                    raise ValueError(
                        "Previously shared layer inputs diverged; disable that deduplication explicitly"
                    )

            return hook

        for name, layer in targets.items():
            handles.append(layer.register_forward_hook(fitting_hook(name)))

        def heartbeat(ordinal, mask):
            if progress_callback is None:
                return
            required = (
                None
                if rank_pattern is None
                else {
                    owner: max(rank_pattern[name] for name in aliases)
                    for owner, aliases in owners.items()
                }
            )
            retained_cosines = (
                []
                if required is None
                else [
                    group["cosines"][: required[owner]].detach().cpu()
                    for owner, group in groups.items()
                    if required[owner]
                ]
            )
            minimum_cosine = (
                None
                if not retained_cosines
                else min(float(value.min()) for value in retained_cosines)
            )
            progress_callback(
                {
                    "batch": ordinal,
                    "elapsed_seconds": time.perf_counter() - started,
                    "content_tokens": int(mask.sum().item()),
                    "cumulative_content_tokens": sum(row["content_tokens"] for row in evidence),
                    "converged": bool(converged),
                    "statistics_groups": len(groups),
                    "initialized_groups": sum(
                        hasattr(group["pca"], "components_") for group in groups.values()
                    ),
                    "group_pca_updates": {
                        owner: group["updates"] for owner, group in groups.items()
                    },
                    "minimum_pca_updates": min(group["updates"] for group in groups.values()),
                    "minimum_retained_component_cosine": minimum_cosine,
                    "realized_rank_budget": None
                    if rank_pattern is None
                    else sum(rank_pattern.values()),
                }
            )

        for ordinal in range(1, max_batches + 1):
            if ordinal == 1:
                batch, mask = first, first_mask
                # Reuse the already observed first inputs rather than consume a
                # second, different batch or waste another native encoder pass.
                for owner in owners:
                    update(owner, probes[owner].to(device))
                probes.clear()
            else:
                batch = next(stream)
                mask = _validated_batch(batch, allowed_ids)
                fitting["mask"], fitting["states"] = mask, {}
                calls.clear()
                forward_fn(backbone, batch)
                if set(calls) != set(targets):
                    raise ValueError("A fitting microbatch omitted an intended projection")
                fitting["states"].clear()
            for doc_id in batch["document_ids"]:
                if doc_id not in seen_set:
                    seen_ids.append(doc_id)
                    seen_set.add(doc_id)
            evidence.append(
                {
                    "batch": ordinal,
                    "document_ids": list(batch["document_ids"]),
                    "chunks": int(mask.shape[0]),
                    "content_tokens": int(mask.sum().item()),
                }
            )
            if not all(hasattr(group["pca"], "components_") for group in groups.values()):
                heartbeat(ordinal, mask)
                continue
            ratios = {
                owner: group["pca"].explained_variance_ratio_ for owner, group in groups.items()
            }
            rank_pattern = allocate_ranks(ratios, target_to_group, caps, rank * len(targets))
            required = {
                owner: max(rank_pattern[name] for name in names) for owner, names in owners.items()
            }
            converged = ordinal >= min_batches and all(
                group["updates"] >= min_batches
                and bool((group["cosines"][: required[owner]] >= tau).all())
                for owner, group in groups.items()
            )
            heartbeat(ordinal, mask)
            if converged:
                break
        if rank_pattern is None:
            raise ValueError(
                "The fitting budget did not provide enough ordinary content for all PCA targets"
            )
        statistics = {}
        for owner, group in groups.items():
            pca = group["pca"]
            statistics[owner] = {
                "components_all": pca.components_.detach().cpu().float().clone(),
                "singular_values": pca.singular_values_.detach().cpu().clone(),
                "mean": pca.mean_.detach().cpu().clone(),
                "variance": pca.var_.detach().cpu().clone(),
                "explained_variance_ratio": pca.explained_variance_ratio_.detach().cpu().clone(),
                "n_samples": int(pca.n_samples_seen_.item()),
                "updates": group["updates"],
                "component_cosines": group["cosines"].detach().cpu().clone(),
                "pending_unfitted_content_rows": group["pending_rows"],
            }
        scale = float(alpha) / rank
        result = {
            "format": FORMAT,
            "method": "eva_lora",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "split": "train",
            "rank": rank,
            "alpha": float(alpha),
            "rho": float(rho),
            "tau": float(tau),
            "scaling": scale,
            "min_batches": min_batches,
            "max_batches": max_batches,
            "batches_processed": len(evidence),
            "seed": seed,
            "converged": bool(converged),
            "rank_pattern": rank_pattern,
            "alpha_pattern": {name: value * scale for name, value in rank_pattern.items()},
            "scaling_pattern": {name: scale for name, value in rank_pattern.items() if value},
            "target_shapes": {
                name: {"in_features": layer.in_features, "out_features": layer.out_features}
                for name, layer in targets.items()
            },
            "components": {
                name: statistics[target_to_group[name]]["components_all"][:value].clone()
                for name, value in rank_pattern.items()
            },
            "statistics": statistics,
            "target_to_statistics": target_to_group,
            "train_document_ids": list(train_document_ids),
            "seen_document_ids": seen_ids,
            "batch_evidence": evidence,
            "data_identity": data_identity,
            "foundation_identity": foundation_identity,
            "whiten": False,
            "center_adapter_input_at_runtime": False,
        }
    finally:
        for handle in handles:
            handle.remove()
        backbone.train(training)
    return result


def validate_eva_initialization(state, *, require_converged=True, expected_foundation_sha256=None):
    if state["format"] != FORMAT or state["split"] != "train":
        raise ValueError("Expected a TRAIN-fitted EVA basis")
    if require_converged and not state["converged"]:
        raise ValueError("EVA PCA did not converge within the fitting budget")
    if (
        expected_foundation_sha256 is not None
        and state["foundation_identity"]["checkpoint_sha256"] != expected_foundation_sha256
    ):
        raise ValueError("EVA basis was fitted from a different foundation")
    ranks = state["rank_pattern"]
    if set(ranks) != set(state["components"]) or sum(ranks.values()) != state["rank"] * len(ranks):
        raise ValueError("EVA rank allocation does not match its factors")
    for name, rank in ranks.items():
        basis = state["components"][name]
        shape = (rank, state["target_shapes"][name]["in_features"])
        if basis.shape != shape or not bool(torch.isfinite(basis).all()):
            raise ValueError("Invalid EVA basis: " + name)
    return state


def load_eva_initialization(
    path, expected_sha256=None, expected_foundation_sha256=None, require_converged=True
):
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("EVA initialization file SHA-256 differs")
    state = torch.load(path, map_location="cpu", weights_only=True)
    validate_eva_initialization(
        state,
        require_converged=require_converged,
        expected_foundation_sha256=expected_foundation_sha256,
    )
    return state


def save_eva_initialization(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(state, stream)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "converged": state["converged"],
        "batches_processed": state["batches_processed"],
        "rank_pattern": state["rank_pattern"],
    }
