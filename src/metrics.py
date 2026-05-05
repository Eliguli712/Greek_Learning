"""Stage-level metric helpers: precision/recall/F1, accuracy, and DAG validity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from sklearn.metrics import precision_recall_fscore_support


def _safe_div(numer: float, denom: float) -> float:
    """Avoid ZeroDivisionError; return 0.0 when denom == 0."""
    return numer / denom if denom != 0 else 0.0


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


@dataclass(frozen=True)
class Metrics:
    """
    Metrics for a single semantic section (or any single evaluation unit).

    Parameters
    ----------
    correct : int
        Count of correct predicted items (true positives).
    predicted : int
        Count of predicted items (true positives + false positives).
    gold : int
        Count of gold/ground-truth items (true positives + false negatives).
    """

    correct: int
    predicted: int
    gold: int

    def __post_init__(self) -> None:
        for name, value in (("correct", self.correct), ("predicted", self.predicted), ("gold", self.gold)):
            if not isinstance(value, int):
                raise TypeError(f"{name} must be int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.correct > self.predicted:
            raise ValueError(f"correct ({self.correct}) cannot exceed predicted ({self.predicted})")
        if self.correct > self.gold:
            raise ValueError(f"correct ({self.correct}) cannot exceed gold ({self.gold})")

    def precision(self) -> float:
        """Precision = correct / predicted."""
        return _safe_div(self.correct, self.predicted)

    def recall(self) -> float:
        """Recall = correct / gold."""
        return _safe_div(self.correct, self.gold)

    def f_score(self, beta: float = 1.0) -> float:
        """
        F-beta score.

        F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
        Defaults to F1 when beta=1.
        """
        if beta <= 0:
            raise ValueError("beta must be > 0")
        p = self.precision()
        r = self.recall()
        beta2 = beta * beta
        return _safe_div((1.0 + beta2) * p * r, (beta2 * p + r))

    def F_score(self) -> float:
        """Backwards-compatible alias for F1."""
        return self.f_score(beta=1.0)

    def as_dict(self) -> Dict[str, float]:
        """Convenience for logging/printing."""
        return {"precision": self.precision(), "recall": self.recall(), "f1": self.F_score()}


class BlockMetrics:
    """
    Aggregates Metrics across multiple semantic sections ("blocks").

    Supports:
      - macro averaging (average of per-block scores)
      - micro averaging (sum counts then compute)
      - weighted macro averaging (weights = gold counts, by default)

    Example
    -------
    bm = BlockMetrics()
    bm.add("entities", correct=10, predicted=12, gold=15)
    bm.add("relations", correct=7, predicted=9, gold=9)
    print(bm.micro().as_dict())
    print(bm.macro())
    """

    def __init__(self) -> None:
        self._blocks: Dict[str, Metrics] = {}

    def add(self, name: str, correct: int, predicted: int, gold: int) -> None:
        """Add/replace a named block's Metrics."""
        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")
        self._blocks[name] = Metrics(correct=correct, predicted=predicted, gold=gold)

    def names(self) -> List[str]:
        """Return block names in insertion order (Python 3.7+ preserves dict order)."""
        return list(self._blocks.keys())

    def get(self, name: str) -> Metrics:
        """Get Metrics for a specific block."""
        if name not in self._blocks:
            raise KeyError(f"Unknown block: {name}")
        return self._blocks[name]

    def per_block(self) -> Dict[str, Dict[str, float]]:
        """Return per-block metrics as simple dicts (good for JSON logs)."""
        return {k: v.as_dict() for k, v in self._blocks.items()}

    def micro(self) -> Metrics:
        """Micro-average: sum counts, then compute Metrics."""
        correct = sum(m.correct for m in self._blocks.values())
        predicted = sum(m.predicted for m in self._blocks.values())
        gold = sum(m.gold for m in self._blocks.values())
        return Metrics(correct=correct, predicted=predicted, gold=gold)

    def macro(self) -> Dict[str, float]:
        """Macro-average: average of per-block precision/recall/F1 (unweighted)."""
        if not self._blocks:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        ps = [m.precision() for m in self._blocks.values()]
        rs = [m.recall() for m in self._blocks.values()]
        fs = [m.F_score() for m in self._blocks.values()]
        return {
            "precision": sum(ps) / len(ps),
            "recall": sum(rs) / len(rs),
            "f1": sum(fs) / len(fs),
        }

    def weighted_macro(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Weighted macro-average of per-block scores.

        Default weights: each block's gold count.
        You may pass explicit weights as a dict {block_name: weight}.
        Blocks missing from weights get weight 0.
        """
        if not self._blocks:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        if weights is None:
            weights = {k: float(m.gold) for k, m in self._blocks.items()}

        total_w = sum(float(weights.get(k, 0.0)) for k in self._blocks.keys())
        if total_w == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        p = sum(float(weights.get(k, 0.0)) * self._blocks[k].precision() for k in self._blocks) / total_w
        r = sum(float(weights.get(k, 0.0)) * self._blocks[k].recall() for k in self._blocks) / total_w
        f = sum(float(weights.get(k, 0.0)) * self._blocks[k].F_score() for k in self._blocks) / total_w
        return {"precision": p, "recall": r, "f1": f}

    def summary(self) -> Dict[str, object]:
        """One-stop summary including per-block, micro, macro, weighted-macro."""
        return {
            "per_block": self.per_block(),
            "micro": self.micro().as_dict(),
            "macro": self.macro(),
            "weighted_macro": self.weighted_macro(),
        }


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
    "Metrics",
    "BlockMetrics",
    "set_prf",
    "labelled_prf",
    "accuracy",
    "dag_validity_rate",
]
