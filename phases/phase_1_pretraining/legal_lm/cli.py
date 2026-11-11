"""Explicit stages; importing the package never downloads data or starts jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from .config import load_config
from .io import atomic_json, load_json, sha256_file


OFFICIAL_REPO = "Exploration-Lab/CS-781-Capstone"
SOURCES = ("acts", "high_courts", "supreme_court", "tribunals")


def download(args):
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().dataset_info(OFFICIAL_REPO, revision=args.revision, files_metadata=True)
    revision = info.sha
    filenames = [f"Phase_1_Pretraining/Train/train_{s}.json" for s in SOURCES] + [
        "Phase_1_Pretraining/Test/test.json"
    ]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = output / "download-manifest.json"
    if existing.exists() and load_json(existing).get("revision") != revision:
        raise ValueError("Download directory belongs to another revision; choose a new directory")
    records = []
    for filename in filenames:
        path = hf_hub_download(
            OFFICIAL_REPO, filename, repo_type="dataset", revision=revision, local_dir=str(output)
        )
        records.append(
            {"filename": filename, "bytes": Path(path).stat().st_size, "sha256": sha256_file(path)}
        )
    manifest = {"repository": OFFICIAL_REPO, "revision": revision, "files": records}
    atomic_json(existing, manifest)
    return manifest


def prepare(args):
    from .data import prepare_corpus

    base = Path(args.input_dir)
    files = {s: base / "Phase_1_Pretraining" / "Train" / f"train_{s}.json" for s in SOURCES}
    download_manifest = base / "download-manifest.json"
    if not download_manifest.is_file():
        raise ValueError("Use the download stage first: its pinned source manifest is required")
    provenance = load_json(download_manifest)
    for source, path in files.items():
        filename = str(path.relative_to(base))
        record = next((x for x in provenance["files"] if x["filename"] == filename), None)
        if record is None or not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Raw input verification failed for {source}")
    manifest = prepare_corpus(
        files,
        args.output,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        audit_fraction=args.audit_fraction,
        generation_examples_per_source=args.examples_per_source,
    )
    manifest["official_data"] = {
        "repository": provenance["repository"],
        "revision": provenance["revision"],
        "download_manifest_sha256": sha256_file(download_manifest),
    }
    atomic_json(Path(args.output) / "manifest.json", manifest)
    return manifest


def tokenizer(args):
    from .tokenization import train_tokenizer

    return train_tokenizer(
        args.corpus,
        args.output,
        vocab_size=args.vocab_size,
        sample_bytes_per_source=args.sample_mib_per_source * 1024**2,
    )


def tokenize(args):
    from .data import tokenize_corpus

    return tokenize_corpus(
        args.corpus, args.tokenizer, args.output, context_length=args.context_length
    )


def run_training(args):
    from .stage5_train import train_scratch

    return train_scratch(
        load_config(args.config),
        args.view,
        args.run_dir,
        device_name=args.device,
        resume=args.resume,
        max_steps=args.max_steps,
        allow_small_model=args.allow_small_model,
        allow_cpu=args.allow_cpu,
    )


def _generation_common(args, *, development=False):
    import torch
    import torch.distributed as dist
    from .generation import (
        generate_continuation,
        stable_seed,
        validate_records,
        validate_predictions,
        package_submission,
    )
    from .evaluation import evaluate_predictions
    from .tokenization import load_tokenizer_json
    from .train import setup_distributed, choose_precision, amp_context, load_model_checkpoint

    rank, world, device = setup_distributed(args.device)
    try:
        outdir = Path(args.output).resolve()
        errors = [None]
        if rank == 0:
            if outdir.exists() and any(outdir.iterdir()):
                errors[0] = "Output directory is not empty; use a new experiment directory"
            else:
                outdir.mkdir(parents=True, exist_ok=True)
        if world > 1:
            dist.broadcast_object_list(errors, src=0)
        if errors[0]:
            raise FileExistsError(errors[0])
        model, checkpoint = load_model_checkpoint(args.checkpoint, device)
        if checkpoint["run_identity"]["diagnostic_only"] and not args.allow_diagnostic:
            raise ValueError(
                "Diagnostic checkpoint cannot produce a competition artifact; use a qualifying trained model"
            )
        tok = load_tokenizer_json(checkpoint["tokenizer_json"])
        saved_tok_hash = checkpoint["token_manifest"]["tokenizer_sha256"]
        if hashlib.sha256(checkpoint["tokenizer_json"].encode()).hexdigest() != saved_tok_hash:
            raise ValueError("Embedded tokenizer hash does not match the training artifact")
        if development:
            from .data import verify_panel_artifacts

            manifest = checkpoint["token_manifest"]
            corpus = Path(manifest["prepared_dir"])
            panel = verify_panel_artifacts(checkpoint["token_dir"], "dev")
            prompts_path = corpus / panel["prompts"]
            truth_path = corpus / panel["references"]
            truth = load_json(truth_path)
            sources = {row["id"]: row["source"] for row in load_json(corpus / panel["examples"])}
        else:
            prompts_path = Path(args.prompts)
        prompts = load_json(prompts_path)
        validate_records(prompts, "prompts")
        if any(len(record["text"].split()) != 512 for record in prompts.values()):
            raise ValueError("Expected official/held-out 512-word prompts")
        precision, dtype = choose_precision(args.precision, device)
        record = None
        if rank == 0:
            record = {
                "kind": "development" if development else "test_submission",
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "parameter_count": checkpoint["parameter_count"],
                "precision": precision,
                "generation": {
                    "strategy": args.strategy,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                    "max_new_tokens": args.max_new_tokens,
                    "target_words": 128,
                    "max_seconds": args.max_seconds,
                    "repetition_penalty": args.repetition_penalty,
                    "no_repeat_ngram_size": args.no_repeat_ngram_size,
                },
            }
            atomic_json(outdir / "generation.json", record)
        if world > 1:
            dist.barrier()
        predictions, diagnostics = {}, {}
        with amp_context(device, dtype):
            for index, (key, prompt_record) in enumerate(prompts.items()):
                if index % world != rank:
                    continue
                text, detail = generate_continuation(
                    model,
                    tok,
                    prompt_record["text"],
                    target_words=128,
                    max_new_tokens=args.max_new_tokens,
                    strategy=args.strategy,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=stable_seed(args.seed, key),
                    max_seconds=args.max_seconds,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                )
                predictions[key] = {"text": text}
                diagnostics[key] = detail
        local = {"predictions": predictions, "diagnostics": diagnostics}
        all_parts = [None] * world
        if world > 1:
            dist.all_gather_object(all_parts, local)
        else:
            all_parts[0] = local
        merged, all_diagnostics = {}, {}
        for part in all_parts:
            merged.update(part["predictions"])
            all_diagnostics.update(part["diagnostics"])
        validate_predictions(merged, prompts, special_tokens=tok.special_tokens)
        # Free the foundation model before loading the separate metric encoder.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # All shared generation work is complete. Tear down NCCL before the
        # rank-zero-only encoder download/scoring, which may take longer than
        # a distributed collective timeout. torchrun still waits for rank zero
        # and propagates a scoring/packaging failure to the caller.
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
        result = None
        if rank == 0:
            atomic_json(outdir / "diagnostics.json", all_diagnostics)
            if development:
                atomic_json(outdir / "prediction.json", merged)
                metrics = evaluate_predictions(
                    truth,
                    merged,
                    include_bertscore=not args.lexical_only,
                    device=str(device),
                    sources=sources,
                )
                result = {
                    **record,
                    **metrics,
                    "prediction_file": str(outdir / "prediction.json"),
                    "truth_file": str(truth_path.resolve()),
                }
                atomic_json(outdir / "metrics.json", result)
            else:
                result = package_submission(
                    merged, prompts, outdir, special_tokens=tok.special_tokens
                )
                atomic_json(outdir / "submission-manifest.json", {**record, **result})
        return result
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def score(args):
    from .evaluation import evaluate_predictions

    sources = None
    if args.sources:
        raw = load_json(args.sources)
        sources = {row["id"]: row["source"] for row in raw} if isinstance(raw, list) else raw
    result = evaluate_predictions(
        load_json(args.truth),
        load_json(args.prediction),
        include_bertscore=not args.lexical_only,
        device=args.device,
        sources=sources,
    )
    result["truth_sha256"] = sha256_file(args.truth)
    result["prediction_sha256"] = sha256_file(args.prediction)
    atomic_json(args.output, result)
    return result


def _select_development(args, *, temperature_comparison=False):
    reports = [load_json(path) for path in args.reports]
    for report in reports:
        score_value = report["pooled"]["bertscore_f1"]
        if (
            report["kind"] != "development"
            or not math.isfinite(score_value)
            or not report["provenance"]["official_primary_metric_computed"]
        ):
            raise ValueError("Checkpoint selection requires development BERTScore results")
    winner = max(reports, key=lambda report: report["pooled"]["bertscore_f1"])
    result = {
        "comparison": "temperature" if temperature_comparison else "checkpoint",
        "selection_rule": "maximum pooled development BERTScore F1; first report wins an exact tie",
        "checkpoint": winner["checkpoint"],
        "metrics": winner["pooled"],
        "generation": winner["generation"],
        "selected_report": str(Path(args.reports[reports.index(winner)]).resolve()),
    }
    atomic_json(args.output, result)
    return result


def select_checkpoint(args):
    """Select the checkpoint with the highest development BERTScore."""
    return _select_development(args)


def select_temperature(args):
    """Select the higher development BERTScore from two temperature runs."""
    return _select_development(args, temperature_comparison=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download", help="Download only official Phase 1 files at a pinned revision")
    d.add_argument("--output", required=True)
    d.add_argument("--revision", default="main")
    d.set_defaults(func=download)
    d = sub.add_parser("prepare", help="Stream raw documents into globally disjoint splits")
    d.add_argument("--input-dir", required=True)
    d.add_argument("--output", required=True)
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--dev-fraction", type=float, default=0.01)
    d.add_argument("--audit-fraction", type=float, default=0.01)
    d.add_argument("--examples-per-source", type=int, default=128)
    d.set_defaults(func=prepare)
    d = sub.add_parser("tokenizer", help="Fit training-only byte BPE")
    d.add_argument("--corpus", required=True)
    d.add_argument("--output", required=True)
    d.add_argument("--vocab-size", type=int, default=32768)
    d.add_argument("--sample-mib-per-source", type=int, default=25)
    d.set_defaults(func=tokenizer)
    d = sub.add_parser("tokenize", help="Write compact document-aware token windows")
    d.add_argument("--corpus", required=True)
    d.add_argument("--tokenizer", required=True)
    d.add_argument("--output", required=True)
    d.add_argument("--context-length", type=int, default=2048)
    d.set_defaults(func=tokenize)
    d = sub.add_parser(
        "train",
        help="Scratch ordinary deduplicated next-token training or exact resume; supports torchrun",
    )
    d.add_argument("--config", required=True)
    d.add_argument("--view", required=True)
    d.add_argument("--run-dir", required=True)
    d.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    d.add_argument(
        "--resume",
        help="Exactly resume this scratch experiment, schedule, optimizer and source queues",
    )
    d.add_argument(
        "--max-steps",
        type=int,
        help="Bound this invocation without changing the token-based schedule",
    )
    d.add_argument(
        "--allow-small-model",
        action="store_true",
        help="Mark smoke runs diagnostic-only; never competition eligible",
    )
    d.add_argument("--allow-cpu", action="store_true")
    d.set_defaults(func=run_training)
    for name, dev in (("generate", False), ("evaluate-checkpoint", True)):
        d = sub.add_parser(
            name,
            help="Evaluate a dev checkpoint"
            if dev
            else "Generate official test predictions and ZIP",
        )
        d.add_argument("--checkpoint", required=True)
        d.add_argument("--output", required=True)
        if not dev:
            d.add_argument("--prompts", required=True)
        d.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
        d.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
        d.add_argument("--strategy", choices=["greedy", "sampling"], default="greedy")
        d.add_argument("--temperature", type=float, default=0.7)
        d.add_argument("--top-p", type=float, default=0.9)
        d.add_argument("--seed", type=int, default=42)
        d.add_argument(
            "--repetition-penalty",
            type=float,
            default=1.0,
            help="Penalty >=1 applied once per distinct generated token ID; excludes prompt",
        )
        d.add_argument(
            "--no-repeat-ngram-size",
            type=int,
            default=0,
            help="Block repeated generated token n-grams; 0 disables; excludes prompt",
        )
        d.add_argument("--max-new-tokens", type=int, default=512)
        d.add_argument("--max-seconds", type=float)
        d.add_argument("--allow-diagnostic", action="store_true")
        if dev:
            d.add_argument(
                "--lexical-only",
                action="store_true",
                help="Diagnostic; cannot select the final checkpoint",
            )
        d.set_defaults(
            func=lambda args, development=dev: _generation_common(args, development=development)
        )
    d = sub.add_parser("score", help="Score already generated held-out predictions")
    d.add_argument("--truth", required=True)
    d.add_argument("--prediction", required=True)
    d.add_argument("--sources")
    d.add_argument("--output", required=True)
    d.add_argument("--device")
    d.add_argument("--lexical-only", action="store_true")
    d.set_defaults(func=score)
    d = sub.add_parser(
        "select-checkpoint", help="Select by consistent pooled dev BERTScore reports"
    )
    d.add_argument("--reports", nargs="+", required=True)
    d.add_argument("--output", required=True)
    d.set_defaults(func=select_checkpoint)
    d = sub.add_parser(
        "select-temperature",
        help="Compare two sampling temperatures on the same checkpoint and frozen dev panel",
    )
    d.add_argument("--reports", nargs=2, required=True)
    d.add_argument("--output", required=True)
    d.set_defaults(func=select_temperature)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    result = args.func(args)
    if result is not None and int(os.environ.get("RANK", 0)) == 0:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0
