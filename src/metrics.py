"""Stage-level metric helpers: precision/recall/F1, accuracy, and DAG validity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from sklearn.metrics import precision_recall_fscore_support


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    support: int

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "support": self.support,
        }


def set_prf(predicted: Iterable, gold: Iterable) -> PRF:
    """Precision/recall/F1 over two unordered collections of comparable items."""
    pred_set = list(predicted)
    gold_set = list(gold)

    pred_counts = _multiset(pred_set)
    gold_counts = _multiset(gold_set)

    tp = 0
    for key, count in pred_counts.items():
        tp += min(count, gold_counts.get(key, 0))

    precision = tp / sum(pred_counts.values()) if pred_counts else 0.0
    recall = tp / sum(gold_counts.values()) if gold_counts else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return PRF(precision=precision, recall=recall, f1=f1, support=sum(gold_counts.values()))


def accuracy(predicted: Sequence, gold: Sequence) -> float:
    """Position-aligned accuracy. Uneven lengths are penalized."""
    if not gold:
        return 0.0
    aligned = min(len(predicted), len(gold))
    correct = sum(1 for i in range(aligned) if predicted[i] == gold[i])
    return correct / len(gold)


def labelled_prf(
    predicted: Sequence[str],
    gold: Sequence[str],
    average: str = "macro",
) -> PRF:
    """Macro/micro PRF over position-aligned categorical predictions (sklearn)."""
    aligned = min(len(predicted), len(gold))
    if aligned == 0:
        return PRF(0.0, 0.0, 0.0, 0)
    pred = list(predicted[:aligned])
    truth = list(gold[:aligned])
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        pred,
        average=average,
        zero_division=0,
    )
    return PRF(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        support=aligned,
    )


def dag_validity_rate(validations: Sequence[dict]) -> float:
    """Fraction of examples whose DAG passes validation."""
    if not validations:
        return 0.0
    valid = sum(1 for v in validations if v.get("ok"))
    return valid / len(validations)


def _multiset(items: Iterable) -> dict:
    counts: dict = {}
    for item in items:
        key = _hashable(item)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _hashable(item):
    if isinstance(item, dict):
        return tuple(sorted(item.items()))
    if isinstance(item, list):
        return tuple(_hashable(v) for v in item)
    return item


__all__ = [
    "PRF",
    "set_prf",
    "labelled_prf",
    "accuracy",
    "dag_validity_rate",
]
