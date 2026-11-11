"""Training-only byte-level BPE with explicit, noncolliding control tokens."""

from __future__ import annotations

import codecs
import json
from functools import cached_property
from pathlib import Path
from typing import Iterable

from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers

from .data import (
    _new_directory,
    _write_json,
    file_sha256,
    iter_prepared_documents,
    read_prepared_manifest,
)


PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN)


def _bytelevel_byte_decoder() -> dict[str, int]:
    """Invert the tokenizer's byte alphabet without losing unfinished bytes.

    This is the ByteLevel encoding convention, not a pretrained vocabulary:
    https://github.com/huggingface/tokenizers/blob/v0.22.2/tokenizers/src/pre_tokenizers/byte_level.rs
    """
    byte_values = [*range(0x21, 0x7F), *range(0xA1, 0xAD), *range(0xAE, 0x100)]
    codepoints = byte_values.copy()
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    return {chr(codepoint): value for codepoint, value in zip(codepoints, byte_values)}


class LegalTokenizer:
    """No automatic BOS/EOS insertion; strings are always ordinary text.

    encode_special_tokens=True is intentionally set after loading: it disables
    special-marker recognition in ordinary user/corpus text. Control IDs are
    inserted explicitly by the dataset/generator, never via text substitution.
    """

    def __init__(self, tokenizer: Tokenizer, path: str | Path | None = None):
        self.tokenizer = tokenizer
        self.path = Path(path).resolve() if path is not None else None
        self.tokenizer.encode_special_tokens = True
        self.special_tokens = SPECIAL_TOKENS
        ids = tuple(self.tokenizer.token_to_id(token) for token in SPECIAL_TOKENS)
        if any(value is None for value in ids) or len(set(ids)) != 3:
            raise ValueError("Tokenizer must have three distinct PAD/BOS/EOS token IDs")
        self.pad_id, self.bos_id, self.eos_id = ids
        self.special_ids = ids
        self.vocab_size = self.tokenizer.get_vocab_size(with_added_tokens=True)
        backend = json.loads(self.tokenizer.to_str())
        if backend.get("normalizer") is not None or backend.get("post_processor") is not None:
            raise ValueError("This pipeline requires no normalizer or automatic postprocessor")
        if backend["model"]["type"] != "BPE" or backend["pre_tokenizer"]["type"] != "ByteLevel":
            raise ValueError("Expected a byte-level BPE tokenizer")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=skip_special_tokens)

    @cached_property
    def byte_tokens(self) -> tuple[bytes | None, ...]:
        """Raw ByteLevel bytes per vocabulary ID; control IDs have no text.

        Decoding individual tokens to strings would lose unfinished UTF-8
        bytes. Invert the tokenizer's byte alphabet instead, and fail closed
        if its decoder/vocabulary does not match this encoding convention.
        """
        backend = json.loads(self.tokenizer.to_str())
        decoder = backend.get("decoder")
        if not isinstance(decoder, dict) or decoder.get("type") != "ByteLevel":
            raise ValueError("Raw byte vocabulary requires a ByteLevel decoder")
        model = backend["model"]
        if (
            model.get("type") != "BPE"
            or model.get("byte_fallback")
            or model.get("continuing_subword_prefix")
            or model.get("end_of_word_suffix")
        ):
            raise ValueError("Unsupported BPE byte-vocabulary configuration")
        vocabulary = model["vocab"]
        if len(vocabulary) != self.vocab_size or set(vocabulary.values()) != set(
            range(self.vocab_size)
        ):
            raise ValueError("Byte vocabulary IDs must exactly cover the model vocabulary")
        if any(
            not row.get("special") or row["id"] not in self.special_ids
            for row in backend.get("added_tokens", [])
        ):
            raise ValueError("Unsupported added token in the raw byte vocabulary")
        inverse = _bytelevel_byte_decoder()
        result: list[bytes | None] = [None] * self.vocab_size
        for token, token_id in vocabulary.items():
            if token_id in self.special_ids:
                continue
            if not token:
                raise ValueError("Ordinary byte-level tokens must be nonempty")
            result[token_id] = bytes(inverse[character] for character in token)
        return tuple(result)

    def generated_bytes(self, ids: Iterable[int]) -> bytes:
        """Reconstruct ordinary generated tokens without discarding any bytes."""
        pieces = []
        for token_id in ids:
            raw = self.byte_tokens[int(token_id)]
            if raw is None:
                raise ValueError("A control token cannot be decoded as generated ordinary text")
            pieces.append(raw)
        return b"".join(pieces)

    def decode_prefix(self, ids: Iterable[int]) -> str:
        """Use conventional ByteLevel replacement decoding on a growing prefix.

        As in the tokenizer's ByteLevel decoder, malformed UTF-8 is represented
        by U+FFFD. A valid unfinished trailing character is buffered until later
        tokens complete it, so it cannot prematurely complete the final word.
        Model-generated literal U+FFFD characters also remain unchanged.
        """
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        return decoder.decode(self.generated_bytes(ids), final=False)

    def replacement_counts(self, ids: Iterable[int], prefix_chars: int) -> dict[str, int]:
        """Distinguish literal and decoder-introduced U+FFFD in an output prefix.

        ``prefix_chars`` is an index into the decoded suffix before whitespace
        normalization. Unused lookahead is excluded. This diagnostic is computed
        once after generation, and does not change decoding or token selection.
        """
        raw = self.generated_bytes(ids)
        decoded = codecs.getincrementaldecoder("utf-8")("replace").decode(raw, final=False)
        total = decoded[:prefix_chars].count("\ufffd")
        literal = 0
        offset = raw.find(b"\xef\xbf\xbd")
        while offset >= 0:
            # EF cannot continue an earlier UTF-8 sequence: finalizing this
            # preceding byte prefix gives exactly the literal's character index.
            position = len(raw[:offset].decode("utf-8", errors="replace"))
            if position >= prefix_chars:
                break
            literal += 1
            offset = raw.find(b"\xef\xbf\xbd", offset + 3)
        return {
            "replacement_characters": total,
            "literal_replacement_characters": literal,
            "malformed_byte_replacement_characters": total - literal,
        }


def load_tokenizer(path: str | Path) -> LegalTokenizer:
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "tokenizer.json"
    return LegalTokenizer(Tokenizer.from_file(str(path)), path)


def load_tokenizer_json(serialized_json: str) -> LegalTokenizer:
    """Load the exact tokenizer embedded in a checkpoint, without a disk path."""
    return LegalTokenizer(Tokenizer.from_str(serialized_json))


def train_tokenizer(
    prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    vocab_size: int = 32768,
    sample_bytes_per_source: int = 25 * 1024**2,
    seed: int | None = None,
) -> dict:
    """Fit on a frozen, deterministic, training-only per-source text sample.

    Exactly the requested vocabulary is not guaranteed by BPE on tiny corpora;
    both requested and achieved sizes are recorded. Production model setup must
    require its configured size, while small synthetic diagnostics may use less.
    """
    if not 259 <= vocab_size <= 65536:
        raise ValueError("Vocabulary must contain 256 bytes + three controls and fit uint16")
    if sample_bytes_per_source < 1:
        raise ValueError("sample_bytes_per_source must be positive")
    prepared_dir = Path(prepared_dir).resolve()
    prepared = read_prepared_manifest(prepared_dir)
    seed = prepared["seed"] if seed is None else seed
    if seed != prepared["seed"]:
        raise ValueError("Tokenizer sampling seed must match the frozen preparation seed")
    output = _new_directory(output_dir)
    sample_path = output / "training_sample.jsonl"
    counts = {}
    with sample_path.open("w", encoding="utf-8") as sample_out:
        for source in prepared["sources"]:
            consumed, documents, truncated = 0, 0, 0
            for source_id, doc_hash, text in iter_prepared_documents(
                prepared_dir, "train", source_id=source["source_id"], unique=True, sample_order=True
            ):
                remaining = sample_bytes_per_source - consumed
                if remaining <= 0:
                    break
                encoded = text.encode("utf-8")
                if len(encoded) > remaining:
                    encoded = encoded[:remaining]
                    text = encoded.decode("utf-8", errors="ignore")
                    encoded = text.encode("utf-8")
                    truncated += 1
                if not encoded:
                    break
                sample_out.write(
                    json.dumps(
                        {"source_id": source_id, "doc_hash": doc_hash, "text": text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                consumed += len(encoded)
                documents += 1
                if truncated:
                    break
            counts[source["name"]] = {
                "utf8_bytes": consumed,
                "documents": documents,
                "truncated_final_document": bool(truncated),
                "below_byte_budget": consumed < sample_bytes_per_source,
            }
    if not sum(item["documents"] for item in counts.values()):
        raise ValueError("No training text available to fit tokenizer")
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.encode_special_tokens = True
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[
            AddedToken(token, special=True, normalized=False) for token in SPECIAL_TOKENS
        ],
    )

    def training_texts():
        with sample_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)["text"]

    tokenizer.train_from_iterator(
        training_texts(), trainer=trainer, length=sum(item["documents"] for item in counts.values())
    )
    tokenizer_path = output / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    wrapped = load_tokenizer(tokenizer_path)
    manifest = {
        "schema_version": 1,
        "kind": "legal_byte_bpe",
        "tokenizer_file": "tokenizer.json",
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "requested_vocab_size": vocab_size,
        "vocab_size": wrapped.vocab_size,
        "special_tokens": list(SPECIAL_TOKENS),
        "special_ids": {"pad": wrapped.pad_id, "bos": wrapped.bos_id, "eos": wrapped.eos_id},
        "seed": seed,
        "sampling_split": "train",
        "sampling_policy": "Hash-ordered unique texts within each source; truncate final sampled document on UTF-8 boundary",
        "sample_bytes_per_source": sample_bytes_per_source,
        "sample_per_source": counts,
        "sample_file": sample_path.name,
        "sample_sha256": file_sha256(sample_path),
        "prepared_dir": str(prepared_dir),
        "prepared_manifest_sha256": file_sha256(prepared_dir / "manifest.json"),
        "normalization": "none",
        "automatic_special_tokens": False,
        "ordinary_special_marker_recognition": False,
    }

    _write_json(output / "manifest.json", manifest)
    return manifest
