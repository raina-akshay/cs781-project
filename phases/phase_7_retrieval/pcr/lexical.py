"""BM25 diagnostic with vocabulary, IDF and average length fitted on TRAIN only.

Candidate/query TF at inference is a deterministic text transform; no held-out
corpus statistics, labels, pretrained text models, stopword lists or stemming.
Query terms contribute once (binary query weighting), avoiding query-document
length repetition dominating a long-case retrieval score.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np
from scipy import sparse

from .data import read_json, write_json

WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.lower())


@dataclass
class TrainBM25:
    vocabulary: dict[str, int]
    idf: np.ndarray
    average_length: float
    train_document_count: int
    k1: float = 1.2
    b: float = 0.75

    @classmethod
    def fit(cls, train_texts: list[str], k1: float = 1.2, b: float = 0.75):
        if not train_texts or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 corpus or parameters")
        frequencies = Counter()
        total = 0
        for text in train_texts:
            terms = words(text)
            total += len(terms)
            frequencies.update(set(terms))
        if not total:
            raise ValueError("Empty lexical TRAIN corpus")
        vocabulary = {term: i for i, term in enumerate(sorted(frequencies))}
        n = len(train_texts)
        idf = np.asarray(
            [
                math.log1p((n - frequencies[t] + 0.5) / (frequencies[t] + 0.5))
                for t in vocabulary
            ],
            dtype=np.float32,
        )
        return cls(vocabulary, idf, total / n, n, k1, b)

    def matrix(self, texts: list[str], queries: bool = False):
        data, indices, indptr = [], [], [0]
        for text in texts:
            tokens = words(text)
            counts = Counter(tokens)
            normalization = self.k1 * (
                1.0 - self.b + self.b * len(tokens) / self.average_length
            )
            row = []
            for token, count in counts.items():
                i = self.vocabulary.get(token)
                if i is None:
                    continue
                value = (
                    1.0
                    if queries
                    else self.idf[i] * count * (self.k1 + 1.0) / (count + normalization)
                )
                row.append((i, value))
            for i, value in sorted(row):
                indices.append(i)
                data.append(value)
            indptr.append(len(data))
        return sparse.csr_matrix(
            (
                np.asarray(data, dtype=np.float32),
                np.asarray(indices, dtype=np.int32),
                np.asarray(indptr, dtype=np.int64),
            ),
            shape=(len(texts), len(self.vocabulary)),
        )

    def score(self, query_texts: list[str], candidate_texts: list[str]) -> np.ndarray:
        return (
            self.matrix(query_texts, queries=True) @ self.matrix(candidate_texts).T
        ).toarray()

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        write_json(
            directory / "bm25.json",
            {
                "vocabulary": list(self.vocabulary),
                "average_length": self.average_length,
                "train_document_count": self.train_document_count,
                "k1": self.k1,
                "b": self.b,
                "query_term_weight": "binary",
                "fit_scope": "TRAIN candidate texts only",
                "tokenizer": "unicode_word_lowercase",
                "version": 1,
            },
        )
        np.save(directory / "idf.npy", self.idf, allow_pickle=False)

    @classmethod
    def load(cls, directory: Path):
        meta = read_json(Path(directory) / "bm25.json")
        obj = cls(
            {t: i for i, t in enumerate(meta["vocabulary"])},
            np.load(Path(directory) / "idf.npy", allow_pickle=False),
            meta["average_length"],
            meta["train_document_count"],
            meta["k1"],
            meta["b"],
        )
        if len(obj.vocabulary) != len(obj.idf) or not np.isfinite(obj.idf).all():
            raise ValueError("Corrupt lexical model")
        return obj
