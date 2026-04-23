"""
Ambiguity-resolution fallback for the lexeme layer.

When the rule-based lemma lookup cannot find a direct hit, this module ranks
known lemma candidates by surface similarity against the target surface form.
Two standard, non-neural similarity signals are combined:

  * character n-gram TF-IDF cosine similarity (scikit-learn), and
  * ratcliff/obershelp ratio (difflib from the Python standard library).

The implementation is deliberately transparent: the caller receives the full
ranked list with per-signal scores, so downstream code can decide whether to
accept a candidate or mark the token as unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _surface_key(text: str) -> str:
    stripped = _strip_accents(text).lower()
    return "".join(ch for ch in stripped if ch.isalpha())


@dataclass
class SimilarityCandidate:
    key: str
    lemma: str
    entry: Mapping[str, Any]
    cosine: float
    ratio: float
    score: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "lemma": self.lemma,
            "cosine": round(self.cosine, 4),
            "ratio": round(self.ratio, 4),
            "score": round(self.score, 4),
        }


class SimilarityComparator:
    """
    Rank lemma candidates by surface similarity.

    Parameters
    ----------
    candidates:
        Mapping from a lookup key (accent-stripped surface form) to a
        dictionary with at least a ``lemma`` field. Extra fields are preserved
        and returned with the ranked candidates.
    ngram_range:
        Character n-gram range for the TF-IDF vectorizer.
    min_score:
        Candidates scoring below this blended score are dropped.
    cosine_weight:
        Weight of the TF-IDF cosine signal in the blended score. The
        difflib ratio uses ``1 - cosine_weight``.
    """

    def __init__(
        self,
        candidates: Mapping[str, Mapping[str, Any]],
        ngram_range: Tuple[int, int] = (2, 4),
        min_score: float = 0.4,
        cosine_weight: float = 0.6,
    ) -> None:
        self._candidates: Dict[str, Dict[str, Any]] = {
            key: dict(entry) for key, entry in candidates.items() if key
        }
        self.ngram_range = ngram_range
        self.min_score = float(min_score)
        self.cosine_weight = float(cosine_weight)

        self._keys: List[str] = list(self._candidates.keys())
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None

        if self._keys:
            self._vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=self.ngram_range,
                lowercase=False,
            )
            self._matrix = self._vectorizer.fit_transform(self._keys)

    def __len__(self) -> int:
        return len(self._candidates)

    def rank(self, surface: str, top_k: int = 5) -> List[SimilarityCandidate]:
        """Return up to ``top_k`` similarity candidates for ``surface``."""
        if not self._keys or self._vectorizer is None or self._matrix is None:
            return []

        query_key = _surface_key(surface)
        if not query_key:
            return []

        query_vec = self._vectorizer.transform([query_key])
        cosines = cosine_similarity(query_vec, self._matrix).ravel()

        ranked: List[SimilarityCandidate] = []
        for idx, key in enumerate(self._keys):
            cosine = float(cosines[idx])
            ratio = SequenceMatcher(None, query_key, key).ratio()
            score = self.cosine_weight * cosine + (1.0 - self.cosine_weight) * ratio
            if score < self.min_score:
                continue
            entry = self._candidates[key]
            ranked.append(
                SimilarityCandidate(
                    key=key,
                    lemma=str(entry.get("lemma", key)),
                    entry=entry,
                    cosine=cosine,
                    ratio=ratio,
                    score=score,
                )
            )

        ranked.sort(key=lambda c: c.score, reverse=True)
        return ranked[: max(0, top_k)]

    def best(self, surface: str) -> Optional[SimilarityCandidate]:
        """Return the single best candidate, or ``None`` if nothing qualifies."""
        ranked = self.rank(surface, top_k=1)
        return ranked[0] if ranked else None

    @classmethod
    def from_components(
        cls,
        components: Mapping[str, Mapping[str, Any]],
        **kwargs: Any,
    ) -> "SimilarityComparator":
        """Build a comparator from a ``components`` lexicon (same shape as
        ``data/lexica/greek_lemmas.json#components``)."""
        normalized = {
            _surface_key(key): dict(entry)
            for key, entry in components.items()
            if _surface_key(key)
        }
        return cls(candidates=normalized, **kwargs)


__all__ = ["SimilarityCandidate", "SimilarityComparator"]
