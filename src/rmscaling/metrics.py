"""Small, dependency-free binary ranking metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence


def binary_auc(labels: Sequence[float], scores: Sequence[float]) -> float | None:
    """Return ROC AUC with average ranks for ties, or None for one-class data."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    n_pos = sum(float(y) > 0.5 for y in labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        mean_rank = ((index + 1) + end) / 2.0
        rank_sum += mean_rank * sum(float(y) > 0.5 for _, y in ordered[index:end])
        index = end
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def binary_logloss(labels: Sequence[float], probabilities: Sequence[float]) -> float:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must have equal nonzero length")
    eps = 1e-15
    return -sum(
        float(y) * math.log(min(max(float(p), eps), 1.0 - eps))
        + (1.0 - float(y)) * math.log(1.0 - min(max(float(p), eps), 1.0 - eps))
        for y, p in zip(labels, probabilities)
    ) / len(labels)


def ndcg_at_k(
    labels: Sequence[float],
    scores: Sequence[float],
    group_ids: Sequence[int],
    k: int = 10,
) -> tuple[float | None, int]:
    """Mean NDCG over observed groups with >=2 items and >=1 click."""
    if not (len(labels) == len(scores) == len(group_ids)):
        raise ValueError("labels, scores and group_ids must have equal length")
    if k < 1:
        raise ValueError("k must be positive")
    groups: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for group, score, label in zip(group_ids, scores, labels):
        groups[int(group)].append((float(score), float(label)))
    values: list[float] = []
    for pairs in groups.values():
        if len(pairs) < 2 or not any(label > 0 for _, label in pairs):
            continue
        ranked = sorted(pairs, key=lambda pair: pair[0], reverse=True)[:k]
        ideal = sorted((label for _, label in pairs), reverse=True)[:k]
        dcg = sum((2.0**label - 1.0) / math.log2(position + 2) for position, (_, label) in enumerate(ranked))
        idcg = sum((2.0**label - 1.0) / math.log2(position + 2) for position, label in enumerate(ideal))
        if idcg > 0:
            values.append(dcg / idcg)
    return (sum(values) / len(values), len(values)) if values else (None, 0)
