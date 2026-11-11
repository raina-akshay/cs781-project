"""TRAIN-fitted lexical MMR over disjoint original-token passages."""
import numpy as np


def merge_spans(spans):
    result = []
    for a, b in sorted(spans):
        if a >= b:
            continue
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def serialize_spans(tokens, spans, separator):
    spans = merge_spans(spans)
    result = []
    for i, (a, b) in enumerate(spans):
        if i:
            result.extend(separator)
        result.extend(tokens[a:b])
    return result, spans


def select_evidence(tokens, tokenizer, vectorizer, budget=1536, passage=192, relevance_weight=0.7):
    if not tokens or budget < 320:
        raise ValueError("Need nonempty source and anchor budget")
    if len(tokens) <= budget:
        return list(tokens), [(0, len(tokens))]
    separator = tokenizer.encode("\n[...]\n")
    anchors = [(0, 128), (len(tokens) - 192, len(tokens))]
    candidates = [
        (a, min(a + passage, len(tokens) - 192)) for a in range(128, len(tokens) - 192, passage)
    ]
    texts = [tokenizer.decode(tokens[a:b]) for a, b in candidates + anchors]
    matrix = vectorizer.transform(texts).tocsr()
    centroid = vectorizer.transform([tokenizer.decode(tokens)])
    count = len(candidates)
    relevance = (matrix[:count] @ centroid.T).toarray().ravel()
    redundancy = (matrix[:count] @ matrix[count:].T).toarray().max(1)
    chosen = list(anchors)
    available = np.ones(count, dtype=bool)
    while available.any():
        score = relevance_weight * relevance - (1 - relevance_weight) * redundancy
        score[~available] = -np.inf
        j = int(score.argmax())
        available[j] = False
        a, b = candidates[j]
        result, spans = serialize_spans(tokens, chosen + [(a, b)], separator)
        while len(result) > budget:
            b -= len(result) - budget
            if b <= a:
                break
            result, spans = serialize_spans(tokens, chosen + [(a, b)], separator)
        if b <= a:
            continue
        chosen = spans
        redundancy = np.maximum(redundancy, (matrix[:count] @ matrix[j].T).toarray().ravel())
        if len(result) == budget:
            break
    result, spans = serialize_spans(tokens, chosen, separator)
    return result, spans
