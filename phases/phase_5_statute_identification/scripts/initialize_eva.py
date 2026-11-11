"""Fit EVA only on a declared, deterministic sample of original TRAIN chunks.

Initialization is a forward-only representation fit. Scored training still sees
every complete TRAIN case. This helper does not construct a task head, consume
case labels, or alter the original foundation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

SOLUTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLUTION))
from lsi.data import CacheBundle
from lsi.eva import compute_eva_initialization, save_eva_initialization
from lsi.foundation import causal_features, load_own_foundation, sha256_file
from lsi.runtime import atomic_json, configure_cuda_runtime, now


class TrainChunkStream:
    """One complete, hash-selected chunk per case in a seeded case permutation."""

    def __init__(self, split, device, micro_chunks, seed=42, chunk_length=2048):
        if split.split != "train" or type(micro_chunks) is not int or micro_chunks < 1:
            raise ValueError("Initialization requires TRAIN and positive physical chunk batches")
        self.split, self.device, self.micro_chunks = split, device, micro_chunks
        self.seed, self.capacity = seed, chunk_length - 2

    def __call__(self):
        order = np.random.default_rng(self.seed).permutation(len(self.split))
        for start in range(0, len(order), self.micro_chunks):
            rows, original_ids = [], []
            for index in order[start : start + self.micro_chunks]:
                index = int(index)
                case_id = self.split.ids[index]
                content = self.split.tokens(index)
                count = max(1, (len(content) + self.capacity - 1) // self.capacity)
                digest = hashlib.sha256(
                    ("cs781-p5-eva-chunk-v1|" + str(self.seed) + "|" + case_id).encode()
                ).digest()
                chunk = int.from_bytes(digest[:8], "big") % count
                part = content[chunk * self.capacity : (chunk + 1) * self.capacity]
                ids = torch.empty(len(part) + 2, dtype=torch.long)
                ids[0], ids[-1] = 1, 2
                if len(part):
                    ids[1:-1] = torch.from_numpy(np.array(part, dtype=np.int64, copy=True))
                rows.append(ids)
                original_ids.append(case_id)
            lengths = torch.tensor([len(row) for row in rows], dtype=torch.long, device=self.device)
            values = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=0).to(
                self.device
            )
            yield {
                "input_ids": values,
                "lengths": lengths,
                "split": "train",
                "document_ids": original_ids,
            }


def run(config_path, fit_path, output):
    config = json.loads(config_path.read_text())
    fit = json.loads(fit_path.read_text())
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    configure_cuda_runtime(device)
    torch.set_num_threads(4)
    started = time.monotonic()
    directory = output.parent
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "status.json"
    base = {
        "pid": os.getpid(),
        "config": str(config_path),
        "fit_config": fit,
        "fit_config_sha256": sha256_file(fit_path),
        "gpus": [0],
        "case_labels_used": False,
        "dev_or_test_used": False,
        "sampling": fit["sampling"],
        "scored_training_coverage": "all original TRAIN cases/content",
    }

    def status(stage, **extra):
        record = {
            **base,
            "state": stage,
            "at_utc": now(),
            "elapsed_seconds": time.monotonic() - started,
            **extra,
        }
        atomic_json(status_path, record)
        print(json.dumps(record, allow_nan=False), flush=True)

    status("verifying_original_assets")
    cache = CacheBundle(config["data_manifest"])
    backbone, settings, report = load_own_foundation(config, load_foundation=True)
    backbone.to(device).requires_grad_(False).eval()
    train = cache.splits["train"]
    stream = TrainChunkStream(train, device, fit["micro_chunks"], fit["seed"])

    def native_forward(net, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return causal_features(net, settings, batch["input_ids"], batch["lengths"])

    def progress(record):
        status(
            "fitting",
            progress=record,
            allocated_bytes=torch.cuda.memory_allocated(device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        )

    status("fitting")
    state = compute_eva_initialization(
        backbone,
        stream,
        native_forward,
        rank=fit["rank"],
        alpha=fit["alpha"],
        rho=fit["rho"],
        tau=fit["tau"],
        min_batches=fit["min_batches"],
        max_batches=fit["max_batches"],
        seed=fit["seed"],
        train_document_ids=train.ids,
        data_identity=cache.identity,
        foundation_identity=report,
        progress_callback=progress,
    )
    reference = save_eva_initialization(state, output)
    atomic_json(directory / "initialization-reference.json", reference)
    status(
        "completed" if state["converged"] else "not_converged",
        artifact=reference,
        fitted_cases=len(state["seen_document_ids"]),
        fitted_content_tokens=sum(row["content_tokens"] for row in state["batch_evidence"]),
    )
    if not state["converged"]:
        raise RuntimeError("EVA did not converge within the fitting budget")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fit-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.config.resolve(), args.fit_config.resolve(), args.output.resolve())
