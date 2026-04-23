"""Shared text and resource normalization helpers used across the pipeline."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping

import nltk
from nltk.tokenize import RegexpTokenizer


_WORD_TOKENIZER = RegexpTokenizer(r"\w+")
# Unicode-aware word tokenizer; \w already matches Greek letters in Python's re.


def strip_accents(text: str) -> str:
    """Remove combining diacritical marks, keep base letters."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def surface_key(text: str) -> str:
    """Return an accent-stripped, lowercase, alphabetic key for lookup."""
    stripped = strip_accents(text).lower()
    return "".join(ch for ch in stripped if ch.isalpha())


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base without mutating either argument."""
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_json_resource(path: Path) -> Dict[str, Any]:
    """Load a JSON file, returning {} if the file is missing or empty."""
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return dict(data) if isinstance(data, Mapping) else {}


def sentence_split(text: str) -> List[str]:
    """Split text into sentences using NLTK punkt where available, regex fallback."""
    text = text.strip()
    if not text:
        return []
    try:
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        except Exception:
            pass
    parts: List[str] = []
    buffer: List[str] = []
    for ch in text:
        buffer.append(ch)
        if ch in ".!?;·":
            sentence = "".join(buffer).strip()
            if sentence:
                parts.append(sentence)
            buffer = []
    tail = "".join(buffer).strip()
    if tail:
        parts.append(tail)
    return parts


def word_tokens(text: str) -> List[str]:
    """Return alphabetic word tokens, accent-preserving, Greek-friendly."""
    if not text:
        return []
    return _WORD_TOKENIZER.tokenize(text)


__all__ = [
    "strip_accents",
    "surface_key",
    "deep_merge",
    "load_json_resource",
    "sentence_split",
    "word_tokens",
]
