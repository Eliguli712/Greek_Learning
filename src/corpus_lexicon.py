"""Corpus-backed lexical hints for the raw semantic compiler.

The hand-written lexicon in this project is intentionally small. This module
bootstraps a transparent fallback lexicon from local CoNLL-U files so the raw
pipeline can use real Ancient Greek lemmas, UPOS tags, and common morphology
without depending on a network service or an opaque model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

from src.normalize import surface_key


@dataclass(frozen=True)
class CorpusLexiconEntry:
    form: str
    lemma: str
    upos: str
    feats: Dict[str, str] = field(default_factory=dict)
    count: int = 0
    source: str = "corpus_exact"
    lemma_remove: int = 0
    lemma_append: str = ""


class CorpusLexicon:
    """Lookup table from normalized surface forms to UD-derived analyses."""

    def __init__(
        self,
        entries: Mapping[str, CorpusLexiconEntry],
        suffix_entries: Mapping[str, CorpusLexiconEntry],
    ) -> None:
        self.entries = dict(entries)
        self.suffix_entries = dict(suffix_entries)

    def get(self, token: str) -> Optional[CorpusLexiconEntry]:
        key = surface_key(token)
        exact = self.entries.get(key)
        if exact is not None:
            return exact

        for suffix_length in range(min(7, len(key)), 0, -1):
            suffix = key[-suffix_length:]
            suffix_entry = self.suffix_entries.get(suffix)
            if suffix_entry is None:
                continue
            lemma = _apply_lemma_rule(key, suffix_entry.lemma_remove, suffix_entry.lemma_append)
            return CorpusLexiconEntry(
                form=token,
                lemma=lemma,
                upos=suffix_entry.upos,
                feats=dict(suffix_entry.feats),
                count=suffix_entry.count,
                source=f"suffix_guess:{suffix}",
                lemma_remove=suffix_entry.lemma_remove,
                lemma_append=suffix_entry.lemma_append,
            )
        return None

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def from_conllu_files(cls, paths: Iterable[Path]) -> "CorpusLexicon":
        choices: Dict[str, Counter[Tuple[str, str, str, str]]] = defaultdict(Counter)
        suffix_choices: Dict[str, Counter[Tuple[str, str, int, str]]] = defaultdict(Counter)

        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    if not raw_line or raw_line.startswith("#"):
                        continue
                    fields = raw_line.rstrip("\n").split("\t")
                    if len(fields) != 10:
                        continue
                    token_id = fields[0]
                    if "-" in token_id or "." in token_id:
                        continue

                    form = fields[1]
                    upos = fields[3]
                    key = surface_key(form)
                    if not key or upos == "PUNCT":
                        continue

                    lemma = fields[2] if fields[2] and fields[2] != "_" else form
                    feats_raw = fields[5] if fields[5] else "_"
                    choices[key][(form, lemma, upos, feats_raw)] += 1
                    lemma_key = surface_key(lemma)
                    remove, append = _lemma_rule(key, lemma_key)
                    for suffix_length in range(1, min(7, len(key)) + 1):
                        suffix = key[-suffix_length:]
                        suffix_choices[suffix][(upos, feats_raw, remove, append)] += 1

        entries: Dict[str, CorpusLexiconEntry] = {}
        for key, counter in choices.items():
            form, lemma, upos, feats_raw = _best_analysis(counter)
            entries[key] = CorpusLexiconEntry(
                form=form,
                lemma=lemma,
                upos=upos,
                feats=_parse_feats(feats_raw),
                count=counter[(form, lemma, upos, feats_raw)],
            )

        suffix_entries: Dict[str, CorpusLexiconEntry] = {}
        for suffix, counter in suffix_choices.items():
            upos, feats_raw, remove, append = _best_suffix_analysis(counter)
            suffix_entries[suffix] = CorpusLexiconEntry(
                form=suffix,
                lemma="",
                upos=upos,
                feats=_parse_feats(feats_raw),
                count=counter[(upos, feats_raw, remove, append)],
                source="suffix_pattern",
                lemma_remove=remove,
                lemma_append=append,
            )

        return cls(entries, suffix_entries)


def corpus_lexicon_from_project(project_root: Path) -> CorpusLexicon:
    return _cached_corpus_lexicon(str(project_root.resolve()))


@lru_cache(maxsize=8)
def _cached_corpus_lexicon(project_root: str) -> CorpusLexicon:
    root = Path(project_root)
    train_paths = sorted((root / "data" / "ud_treebanks").glob("*-train.conllu"))
    if train_paths:
        return CorpusLexicon.from_conllu_files(train_paths)

    real_eval = root / "data" / "real_eval"
    return CorpusLexicon.from_conllu_files(sorted(real_eval.glob("*.conllu")))


def _best_analysis(counter: Counter[Tuple[str, str, str, str]]) -> Tuple[str, str, str, str]:
    def sort_key(item: Tuple[Tuple[str, str, str, str], int]) -> Tuple[int, int, str]:
        (form, lemma, upos, feats_raw), count = item
        has_features = 1 if feats_raw and feats_raw != "_" else 0
        return (count, has_features, f"{upos}:{lemma}:{form}")

    return max(counter.items(), key=sort_key)[0]


def _best_suffix_analysis(counter: Counter[Tuple[str, str, int, str]]) -> Tuple[str, str, int, str]:
    def sort_key(item: Tuple[Tuple[str, str, int, str], int]) -> Tuple[int, int, str]:
        (upos, feats_raw, remove, append), count = item
        has_features = 1 if feats_raw and feats_raw != "_" else 0
        return (count, has_features, f"{upos}:{feats_raw}:{remove}:{append}")

    return max(counter.items(), key=sort_key)[0]


def _lemma_rule(form_key: str, lemma_key: str) -> Tuple[int, str]:
    if not form_key or not lemma_key:
        return (0, "")

    prefix = 0
    limit = min(len(form_key), len(lemma_key))
    while prefix < limit and form_key[prefix] == lemma_key[prefix]:
        prefix += 1
    return (len(form_key) - prefix, lemma_key[prefix:])


def _apply_lemma_rule(form_key: str, remove: int, append: str) -> str:
    if remove <= 0:
        return form_key + append
    if remove >= len(form_key):
        return append or form_key
    return form_key[:-remove] + append


def _parse_feats(raw: str) -> Dict[str, str]:
    if not raw or raw == "_":
        return {}

    feats: Dict[str, str] = {}
    for item in raw.split("|"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        feats[key] = value
    return feats


__all__ = ["CorpusLexicon", "CorpusLexiconEntry", "corpus_lexicon_from_project"]
