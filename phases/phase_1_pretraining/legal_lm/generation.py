"""Single-prompt cached continuation generation and strict submission packaging.

Only generated token IDs are decoded as the continuation. A word is considered
complete only after a following whitespace character has actually been decoded.
No fallback text, prompt truncation, or automatic filler is used.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Mapping
import zipfile

import torch


def utf8_validation_name(tokenizer: Any) -> str:
    """Record byte-decoding semantics separately from token-sampling settings."""
    return "bytelevel_replace_prefix_v1"


def stable_seed(seed: int, example_id: str) -> int:
    """An ID's sampling seed stays fixed across ordering/GPU partition changes."""
    payload = json.dumps([int(seed), example_id], ensure_ascii=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def completed_words(text: str, target_words: int) -> str | None:
    """Return a normalized prefix only after its final word has a boundary.

    ``len(text.split()) == n`` alone is insufficient: the final BPE token may
    be just a word fragment. Python's Unicode whitespace convention is also
    the convention used by the official evaluator's ``str.split()``.
    """
    words = list(re.finditer(r"\S+", text))
    if len(words) < target_words:
        return None
    final = words[target_words - 1]
    if final.end() == len(text):
        return None
    result = " ".join(word.group() for word in words[:target_words])
    # U+FFFD is valid Unicode, whether literal or introduced by ByteLevel decoding.
    result.encode("utf-8", errors="strict")
    return result


def _apply_repetition_penalty(
    scores: torch.Tensor, seen_tokens: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Penalize each distinct generated token once, independent of frequency.

    This is the sign-aware transform used by Hugging Face's
    RepetitionPenaltyLogitsProcessor. Our caller tracks only the generated
    suffix; prompt tokens are intentionally excluded from the penalty history.
    https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
    """
    penalized = torch.where(scores < 0, scores * penalty, scores / penalty)
    return torch.where(seen_tokens, penalized, scores)


def _choose_token(
    logits: torch.Tensor,
    special_ids: tuple[int, ...],
    strategy: str,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
    seen_tokens: torch.Tensor | None = None,
    repetition_penalty: float = 1.0,
    banned_token_ids: tuple[int, ...] = (),
) -> int:
    scores = logits.float().clone()
    if not torch.isfinite(scores).all():
        raise RuntimeError("Model returned non-finite logits")
    if seen_tokens is not None and repetition_penalty != 1.0:
        scores = _apply_repetition_penalty(scores, seen_tokens, repetition_penalty)
    scores[list(special_ids)] = -torch.inf
    if banned_token_ids:
        scores[list(banned_token_ids)] = -torch.inf
    if not torch.isfinite(scores).any():
        raise RuntimeError("All next-token choices are suppressed")
    if strategy == "greedy":
        return int(scores.argmax().item())
    scores = scores / temperature
    sorted_scores, sorted_ids = scores.sort(descending=True)
    cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
    # Retain the token that crosses the threshold, and always the best token.
    remove = cumulative > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    sorted_scores[remove] = -torch.inf
    probabilities = torch.softmax(sorted_scores, dim=-1)
    selected = torch.multinomial(probabilities, 1, generator=generator)
    return int(sorted_ids[selected].item())


@torch.inference_mode()
def generate_continuation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    target_words: int = 128,
    max_new_tokens: int = 512,
    strategy: str = "greedy",
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    seed: int = 42,
    max_seconds: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate exactly ``target_words`` completed words, or raise.

    Sampling uses an independent seed for each example. Repetition controls
    apply only to the generated suffix. Byte decoding buffers unfinished UTF-8
    characters, and the last word must have a following whitespace boundary.
    """
    encoded = tokenizer.encode(prompt)
    suppressed = tokenizer.special_ids
    prompt_ids = [tokenizer.bos_id, *encoded]
    context_length = model.config.context_length
    capacity = context_length - len(prompt_ids)
    if capacity < 1:
        raise RuntimeError("The full prompt leaves no room for a continuation")
    limit = min(max_new_tokens, capacity)
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    past = None
    generated: list[int] = []
    # Each observed suffix n-gram adds one prefix -> continuation entry.
    # Looking up the current prefix avoids rescanning all earlier tokens.
    ngram_history: dict[tuple[int, ...], set[int]] = {}
    # Maintain one vocabulary-sized device mask instead of copying the growing
    # Python token history to the device at every generation step.
    seen_tokens = (
        torch.zeros(tokenizer.vocab_size, dtype=torch.bool, device=device)
        if repetition_penalty != 1.0
        else None
    )
    decoded = ""
    started = time.monotonic()
    was_training = model.training
    model.eval()
    try:
        for _ in range(limit):
            if max_seconds is not None and time.monotonic() - started > max_seconds:
                raise RuntimeError("Generation time guard reached before the word target")
            output = model(input_ids, past_key_values=past, use_cache=True, last_token_only=True)
            banned_tokens: tuple[int, ...] = ()
            if no_repeat_ngram_size and len(generated) >= no_repeat_ngram_size - 1:
                prefix = tuple(generated[-(no_repeat_ngram_size - 1) :])
                banned_tokens = tuple(sorted(ngram_history.get(prefix, ())))
            next_id = _choose_token(
                output.logits[0, -1],
                suppressed,
                strategy,
                temperature,
                top_p,
                generator,
                seen_tokens,
                repetition_penalty,
                banned_tokens,
            )
            generated.append(next_id)
            if no_repeat_ngram_size and len(generated) >= no_repeat_ngram_size:
                prefix = tuple(generated[-no_repeat_ngram_size:-1])
                ngram_history.setdefault(prefix, set()).add(next_id)
            if seen_tokens is not None:
                seen_tokens[next_id] = True
            # Decode the entire generated suffix, never an individual byte token.
            decoded = tokenizer.decode_prefix(generated)
            result = completed_words(decoded, target_words)
            if result is not None:
                for marker in tokenizer.special_tokens:
                    if marker in result:
                        raise RuntimeError("Continuation contains a literal special-token marker")
                prefix_end = list(re.finditer(r"\S+", decoded))[target_words - 1].end()
                replacements = tokenizer.replacement_counts(generated, prefix_end)
                return result, {
                    "prompt_tokens": len(encoded),
                    "input_tokens_including_bos": len(prompt_ids),
                    "generated_tokens": len(generated),
                    "output_words": target_words,
                    "context_length": context_length,
                    "token_guard": max_new_tokens,
                    "effective_token_limit": limit,
                    "strategy": strategy,
                    "temperature": temperature if strategy == "sampling" else None,
                    "top_p": top_p if strategy == "sampling" else None,
                    "repetition_penalty": repetition_penalty,
                    "repetition_penalty_scope": "generated_suffix",
                    "no_repeat_ngram_size": no_repeat_ngram_size,
                    "no_repeat_ngram_scope": "generated_suffix",
                    "utf8_validation": "bytelevel_replace_prefix_v1",
                    **replacements,
                    "replacement_count_scope": "exported_continuation",
                    "seed": int(seed) if strategy == "sampling" else None,
                    "starts_with_whitespace": bool(decoded and decoded[0].isspace()),
                    "final_word_boundary_observed": True,
                    "elapsed_seconds": time.monotonic() - started,
                }
            past = output.past_key_values
            input_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
        reason = "context capacity" if capacity <= max_new_tokens else "token guard"
        raise RuntimeError(
            f"Reached {reason} after {len(generated)} tokens; decoded "
            f"{len(decoded.split())} word fragments, but not {target_words} completed words"
        )
    finally:
        model.train(was_training)


def strict_json_load(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_records(records: Any, name: str = "records") -> None:
    if not isinstance(records, dict) or not records:
        raise ValueError(f"{name} must be a nonempty JSON object mapping IDs to records")
    for key, record in records.items():
        if (
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(record, dict)
            or not isinstance(record.get("text"), str)
            or not record["text"].strip()
        ):
            raise ValueError(f"Invalid {name} record for ID {key!r}")


def validate_predictions(
    predictions: Mapping[str, Any],
    prompts: Mapping[str, Any],
    *,
    target_words: int = 128,
    special_tokens: tuple[str, ...] = ("<|pad|>", "<|bos|>", "<|eos|>"),
) -> None:
    """Validate export shape/IDs/length; final word completion is decoder-owned."""
    if target_words < 1:
        raise ValueError("target_words must be positive")
    validate_records(predictions, "predictions")
    validate_records(prompts, "prompts")
    missing, extra = set(prompts) - set(predictions), set(predictions) - set(prompts)
    if missing or extra:
        raise ValueError(f"ID mismatch: missing={len(missing)}, extra={len(extra)}")
    for example_id, record in predictions.items():
        if set(record) != {"text"}:
            raise ValueError(f"Prediction {example_id!r} must contain only the text field")
        text = record["text"]
        if len(text.split()) != target_words:
            raise ValueError(f"Prediction {example_id!r} does not contain {target_words} words")
        text.encode("utf-8", errors="strict")
        if any(marker in text for marker in special_tokens):
            raise ValueError(f"Prediction {example_id!r} contains invalid text/control markers")
        # A text-only validator cannot prove token provenance. For official
        # 512-word prompts the 128-word check also precludes prompt prepending.
        prompt = prompts[example_id]["text"].strip()
        if len(prompt.split()) >= target_words and text.lstrip().startswith(prompt):
            raise ValueError(f"Prediction {example_id!r} includes the full prompt")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def package_submission(
    predictions: Mapping[str, Any],
    prompts: Mapping[str, Any],
    output_dir: str | Path,
    *,
    target_words: int = 128,
    special_tokens: tuple[str, ...] = ("<|pad|>", "<|bos|>", "<|eos|>"),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write predictions and a deterministic ZIP containing prediction.json."""
    validate_predictions(
        predictions, prompts, target_words=target_words, special_tokens=special_tokens
    )
    output = Path(output_dir)
    prediction_path, zip_path = output / "prediction.json", output / "submission.zip"
    if not overwrite and (prediction_path.exists() or zip_path.exists()):
        raise FileExistsError("Submission artifacts already exist; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(predictions, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        member = zipfile.ZipInfo("prediction.json", date_time=(1980, 1, 1, 0, 0, 0))
        member.compress_type = zipfile.ZIP_DEFLATED
        member.external_attr = 0o100644 << 16
        archive.writestr(member, payload)
    zip_payload = buffer.getvalue()
    _atomic_bytes(prediction_path, payload)
    _atomic_bytes(zip_path, zip_payload)
    return {
        "examples": len(predictions),
        "target_words": target_words,
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": hashlib.sha256(payload).hexdigest(),
        "zip_path": str(zip_path.resolve()),
        "zip_sha256": hashlib.sha256(zip_payload).hexdigest(),
    }
