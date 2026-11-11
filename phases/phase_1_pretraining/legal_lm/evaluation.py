"""Official Phase 1 metric semantics, with explicitly optional BERTScore.

ROUGE-L and BLEU reproduce the supplied official-evaluate_phase1.py. BERTScore
uses its English defaults, batch size 8, no IDF or baseline rescaling. The metric
encoder is an evaluation dependency only, never part of foundation training.
"""

from __future__ import annotations

from collections import Counter
from importlib import metadata
import math
from typing import Any, Mapping, Sequence

from .generation import validate_records


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    masks: dict[str, int] = {}
    for index, token in enumerate(a):
        masks[token] = masks.get(token, 0) | (1 << index)
    state = 0
    for token in b:
        matched = masks.get(token, 0) | state
        shifted = (state << 1) | 1
        state = matched & ~(matched - shifted)
    return state.bit_count()


def rouge_l_f1(prediction: str, reference: str) -> float:
    predicted, expected = prediction.split(), reference.split()
    if not predicted or not expected:
        return 0.0
    common = lcs_length(predicted, expected)
    if not common:
        return 0.0
    precision, recall = common / len(predicted), common / len(expected)
    return 2 * precision * recall / (precision + recall)


def ngrams(tokens: Sequence[str], order: int) -> Counter:
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def bleu4(predictions: Sequence[Sequence[str]], references: Sequence[Sequence[str]]) -> float:
    """Case-sensitive, unsmoothed corpus BLEU-4; not mean sentence BLEU."""
    if len(predictions) != len(references):
        raise ValueError("Prediction and reference counts differ")
    clipped, totals = [0] * 4, [0] * 4
    predicted_length = reference_length = 0
    for prediction, reference in zip(predictions, references):
        predicted_length += len(prediction)
        reference_length += len(reference)
        for order in range(1, 5):
            predicted = ngrams(prediction, order)
            expected = ngrams(reference, order)
            totals[order - 1] += sum(predicted.values())
            clipped[order - 1] += sum(
                min(count, expected.get(gram, 0)) for gram, count in predicted.items()
            )
    precision = [clipped[index] / totals[index] if totals[index] else 0.0 for index in range(4)]
    if any(value <= 0 for value in precision) or predicted_length == 0:
        return 0.0
    penalty = (
        1.0
        if predicted_length > reference_length
        else math.exp(1.0 - reference_length / predicted_length)
    )
    return penalty * math.exp(sum(math.log(value) for value in precision) / 4)


def _version(package: str) -> str | None:
    return metadata.version(package)


def evaluate_predictions(
    truth: Mapping[str, Any],
    predictions: Mapping[str, Any],
    *,
    include_bertscore: bool = True,
    device: str | None = None,
    sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate aligned records, optionally with explicit per-source groups.

    ``include_bertscore=False`` is a cheap diagnostic, not the official primary
    metric and not sufficient for checkpoint selection. No audit split is read
    implicitly. BERTScore is imported and its encoder loaded only if requested.
    """
    validate_records(truth, "truth")
    validate_records(predictions, "predictions")
    missing, extra = set(truth) - set(predictions), set(predictions) - set(truth)
    if missing or extra:
        raise ValueError(f"ID mismatch: missing={len(missing)}, extra={len(extra)}")
    ids = list(truth)
    if sources is not None:
        if set(sources) != set(ids) or any(
            not isinstance(source, str) or not source.strip() for source in sources.values()
        ):
            raise ValueError(
                "Sources must map every evaluated ID exactly once to a nonempty source name"
            )
    predicted = [predictions[key]["text"].strip() for key in ids]
    references = [truth[key]["text"].strip() for key in ids]
    rouge = [
        rouge_l_f1(prediction, reference) for prediction, reference in zip(predicted, references)
    ]
    f1 = None
    bert_metadata: dict[str, Any] = {"enabled": include_bertscore}
    if include_bertscore:
        from bert_score import BERTScorer

        # BERTScorer exposes the resolved encoder configuration for provenance.
        # These settings equal bert_score.score(lang='en', batch_size=8), as
        # used by the supplied evaluator. Pass batch_size to score as well.
        scorer = BERTScorer(lang="en", batch_size=8, device=device)
        _, _, f1 = scorer.score(predicted, references, batch_size=8, verbose=False)
        f1 = f1.detach().cpu()
        if f1.numel() != len(ids) or not all(math.isfinite(float(value)) for value in f1):
            raise RuntimeError("BERTScore returned invalid per-example scores")
        model_config = getattr(getattr(scorer, "_model", None), "config", None)
        revision = getattr(model_config, "_commit_hash", None)
        bert_metadata.update(
            {
                "lang": "en",
                "batch_size": 8,
                "hash": scorer.hash,
                "model_type": scorer.model_type,
                "num_layers": scorer.num_layers,
                "model_revision": revision,
                "model_revision_status": "resolved_from_loaded_config"
                if revision
                else "unavailable_in_loaded_config",
                "idf": scorer.idf,
                "rescale_with_baseline": scorer.rescale_with_baseline,
                "use_fast_tokenizer": scorer.use_fast_tokenizer,
                "device": str(scorer.device),
            }
        )

    def summarize(indices: list[int]) -> dict[str, Any]:
        # Tensor mean matches the official F.mean().item() aggregation, including
        # its floating-point precision. BLEU is recomputed on each whole group.
        return {
            "examples": len(indices),
            "bertscore_f1": float(f1[indices].mean().item()) if f1 is not None else None,
            "rouge_l_f1": sum(rouge[index] for index in indices) / len(indices),
            "bleu4": bleu4(
                [predicted[index].split() for index in indices],
                [references[index].split() for index in indices],
            ),
        }

    grouped: dict[str, list[int]] = {}
    if sources is not None:
        for index, key in enumerate(ids):
            grouped.setdefault(sources[key], []).append(index)
    return {
        "pooled": summarize(list(range(len(ids)))),
        "per_source": {source: summarize(indices) for source, indices in sorted(grouped.items())},
        "provenance": {
            "official_primary_metric_computed": include_bertscore,
            "tokenization": "Python str.split(); case-sensitive; no BLEU smoothing",
            "versions": {
                name: _version(name)
                for name in ("torch", "bert-score", "transformers", "tokenizers")
            },
            "bertscore": bert_metadata,
        },
    }
