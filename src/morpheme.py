from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import unicodedata


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _surface_key(text: str) -> str:
    stripped = _strip_accents(text).lower()
    return "".join(ch for ch in stripped if ch.isalpha())


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_json_resource(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, Mapping):
        return dict(data)
    return {}


DEFAULT_MORPH_RULES: Dict[str, Any] = {
    "whole_word": {
        "μεγαλοδους": [
            {
                "form": "μεγαλο-",
                "type": "combining_form",
                "gloss": "large",
                "lemma": "μέγας",
            },
            {
                "form": "οδους",
                "type": "root",
                "gloss": "tooth",
                "lemma": "ὀδούς",
                "pos": "NOUN",
            },
        ]
    },
    "combining_forms": {
        "μεγαλο": {
            "form": "μεγαλο-",
            "type": "combining_form",
            "gloss": "large",
            "lemma": "μέγας",
        },
        "φιλο": {
            "form": "φιλο-",
            "type": "combining_form",
            "gloss": "loving / fond of",
            "lemma": "φίλος",
        },
        "θεο": {
            "form": "θεο-",
            "type": "combining_form",
            "gloss": "god / divine",
            "lemma": "θεός",
        },
        "μικρο": {
            "form": "μικρο-",
            "type": "combining_form",
            "gloss": "small",
            "lemma": "μικρός",
        },
    },
    "roots": {
        "οδους": {
            "form": "οδους",
            "type": "root",
            "gloss": "tooth",
            "lemma": "ὀδούς",
            "pos": "NOUN",
        },
        "λογος": {
            "form": "λογος",
            "type": "root",
            "gloss": "word / account / reason",
            "lemma": "λόγος",
            "pos": "NOUN",
        },
        "θεος": {
            "form": "θεος",
            "type": "root",
            "gloss": "god",
            "lemma": "θεός",
            "pos": "NOUN",
        },
        "ανθρωπος": {
            "form": "ανθρωπος",
            "type": "root",
            "gloss": "human / person",
            "lemma": "ἄνθρωπος",
            "pos": "NOUN",
        },
        "γραφ": {
            "form": "γραφ",
            "type": "root",
            "gloss": "write",
            "lemma": "γράφω",
            "pos": "VERB",
        },
    },
    "endings": {
        "ος": {
            "form": "-ος",
            "type": "inflectional_ending",
            "gloss": "nom.sg.masc",
        },
        "ους": {
            "form": "-ους",
            "type": "inflectional_ending",
            "gloss": "nom.sg / acc.pl",
        },
        "ον": {
            "form": "-ον",
            "type": "inflectional_ending",
            "gloss": "nom/acc.sg.neut",
        },
        "η": {
            "form": "-η",
            "type": "inflectional_ending",
            "gloss": "nom.sg.fem",
        },
        "αι": {
            "form": "-αι",
            "type": "inflectional_ending",
            "gloss": "nom.pl.fem",
        },
        "ς": {
            "form": "-ς",
            "type": "inflectional_ending",
            "gloss": "word-final sigma",
        },
    },
}


@dataclass
class Morpheme:
    form: str
    type: str
    gloss: str
    lemma: Optional[str] = None
    source: str = "rule"
    features: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "form": self.form,
            "type": self.type,
            "gloss": self.gloss,
        }
        if self.lemma:
            payload["lemma"] = self.lemma
        if self.source:
            payload["source"] = self.source
        if self.features:
            payload["features"] = self.features
        if self.confidence != 1.0:
            payload["confidence"] = round(self.confidence, 3)
        return payload


@dataclass
class MorphemeAnalysis:
    token: str
    normalized_token: str
    morphemes: List[Morpheme]
    strategy: str
    candidate_score: float
    notes: List[str] = field(default_factory=list)
    syllables: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "token": self.token,
            "normalized_token": self.normalized_token,
            "morphemes": [item.as_dict() for item in self.morphemes],
            "strategy": self.strategy,
            "candidate_score": round(self.candidate_score, 3),
        }
        if self.notes:
            payload["notes"] = self.notes
        if self.syllables:
            payload["syllables"] = self.syllables
        return payload


class MorphemeSegmenter:
    """
    Rule-based morpheme segmenter for the project's compiler pipeline.

    The design follows the specification in `ancient_greek_translator.docx`:
    lexicon-first matching, longest valid match, rule fallback, and simple
    legality checks over candidate templates.
    """

    def __init__(
        self,
        resources: Optional[Mapping[str, Any]] = None,
        project_root: Optional[Path] = None,
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])

        resource_rules: Dict[str, Any] = {}
        if resources and isinstance(resources.get("morph_rules"), Mapping):
            resource_rules = dict(resources["morph_rules"])
        else:
            resource_rules = _load_json_resource(self.project_root / "data/lexica/morph_rules.json")

        merged_rules = _deep_merge(DEFAULT_MORPH_RULES, resource_rules)
        self.whole_word = dict(merged_rules.get("whole_word", {}))
        self.combining_forms = dict(merged_rules.get("combining_forms", {}))
        self.roots = dict(merged_rules.get("roots", {}))
        self.endings = dict(merged_rules.get("endings", {}))

    def analyze(
        self,
        token: str,
        syllables: Optional[Sequence[str]] = None,
        phonemes: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> MorphemeAnalysis:
        normalized_token = unicodedata.normalize("NFC", token.strip())
        key = _surface_key(normalized_token)
        if not key:
            raise ValueError("token must contain at least one alphabetic character")

        candidates: List[Tuple[float, str, List[Morpheme], List[str]]] = []

        whole_word = self._whole_word_candidate(key)
        if whole_word is not None:
            candidates.append(whole_word)

        candidates.extend(self._compound_candidates(key))
        candidates.extend(self._root_and_ending_candidates(key))

        direct_root = self._single_root_candidate(key)
        if direct_root is not None:
            candidates.append(direct_root)

        if not candidates:
            candidates.append(self._fallback_candidate(normalized_token))

        score, strategy, morphemes, notes = max(candidates, key=lambda item: item[0])

        if phonemes:
            notes = list(notes) + [f"received {len(phonemes)} phoneme segments from prior layer"]

        return MorphemeAnalysis(
            token=normalized_token,
            normalized_token=normalized_token,
            morphemes=morphemes,
            strategy=strategy,
            candidate_score=score,
            notes=notes,
            syllables=list(syllables or []),
        )

    def segment(
        self,
        token: str,
        syllables: Optional[Sequence[str]] = None,
        phonemes: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> MorphemeAnalysis:
        return self.analyze(token, syllables=syllables, phonemes=phonemes)

    def _whole_word_candidate(
        self,
        key: str,
    ) -> Optional[Tuple[float, str, List[Morpheme], List[str]]]:
        if key not in self.whole_word:
            return None

        morphemes = [
            self._morpheme_from_entry(entry, source="whole_word_lexicon", confidence=0.99)
            for entry in self.whole_word[key]
        ]
        return (
            0.99,
            "whole_word_lexicon",
            morphemes,
            ["used explicit segmentation override from morph lexicon"],
        )

    def _compound_candidates(self, key: str) -> List[Tuple[float, str, List[Morpheme], List[str]]]:
        candidates: List[Tuple[float, str, List[Morpheme], List[str]]] = []
        ordered_forms = sorted(self.combining_forms.items(), key=lambda item: len(item[0]), reverse=True)

        for form_key, form_entry in ordered_forms:
            if not key.startswith(form_key) or len(key) == len(form_key):
                continue

            remainder = key[len(form_key) :]
            root_entry, restored = self._resolve_root(remainder)
            if root_entry is None:
                continue

            morphemes = [
                self._morpheme_from_entry(form_entry, source="combining_form_lexicon", confidence=0.95),
                self._morpheme_from_entry(root_entry, source="root_lexicon", confidence=0.94),
            ]
            if not self._template_is_legal(morphemes):
                continue

            notes = ["selected longest valid combining-form + root candidate"]
            if restored:
                notes.append("restored an initial root vowel for compound normalization")

            score = 0.93 if not restored else 0.91
            candidates.append((score, "compound_match", morphemes, notes))

        return candidates

    def _root_and_ending_candidates(
        self,
        key: str,
    ) -> List[Tuple[float, str, List[Morpheme], List[str]]]:
        candidates: List[Tuple[float, str, List[Morpheme], List[str]]] = []
        ordered_endings = sorted(self.endings.items(), key=lambda item: len(item[0]), reverse=True)

        for ending_key, ending_entry in ordered_endings:
            if not key.endswith(ending_key) or len(key) == len(ending_key):
                continue

            stem = key[: -len(ending_key)]
            root_entry, restored = self._resolve_root(stem)
            if root_entry is None:
                continue

            morphemes = [
                self._morpheme_from_entry(root_entry, source="root_lexicon", confidence=0.85),
                self._morpheme_from_entry(ending_entry, source="ending_rules", confidence=0.8),
            ]
            if not self._template_is_legal(morphemes):
                continue

            notes = ["matched lexical stem plus inflectional ending"]
            if restored:
                notes.append("restored an initial root vowel before ending analysis")

            candidates.append((0.82, "root_plus_ending", morphemes, notes))

        return candidates

    def _single_root_candidate(
        self,
        key: str,
    ) -> Optional[Tuple[float, str, List[Morpheme], List[str]]]:
        root_entry, _ = self._resolve_root(key)
        if root_entry is None:
            return None

        morphemes = [self._morpheme_from_entry(root_entry, source="root_lexicon", confidence=0.88)]
        return (0.88, "root_match", morphemes, ["token resolved directly to a lexical root"])

    def _fallback_candidate(self, token: str) -> Tuple[float, str, List[Morpheme], List[str]]:
        fallback = Morpheme(
            form=_surface_key(token) or token,
            type="root",
            gloss="unknown",
            source="fallback",
            confidence=0.3,
        )
        return (
            0.3,
            "fallback",
            [fallback],
            ["no lexicon or rule candidate matched; emitted opaque root fallback"],
        )

    def _resolve_root(self, key: str) -> Tuple[Optional[Mapping[str, Any]], bool]:
        if key in self.roots:
            return self.roots[key], False

        for vowel in ("ο", "ε", "α", "ι", "υ"):
            restored_key = vowel + key
            if restored_key in self.roots:
                return self.roots[restored_key], True

        return None, False

    def _template_is_legal(self, morphemes: Sequence[Morpheme]) -> bool:
        template = tuple(item.type for item in morphemes)
        legal_templates = {
            ("root",),
            ("combining_form", "root"),
            ("prefix", "root"),
            ("root", "inflectional_ending"),
        }
        return template in legal_templates

    def _morpheme_from_entry(
        self,
        entry: Mapping[str, Any],
        source: str,
        confidence: float,
    ) -> Morpheme:
        features = {}
        for feature_name in ("pos", "features"):
            if feature_name in entry:
                features[feature_name] = entry[feature_name]

        return Morpheme(
            form=str(entry.get("form", "")),
            type=str(entry.get("type", "root")),
            gloss=str(entry.get("gloss", "unknown")),
            lemma=entry.get("lemma"),
            source=source,
            features=features,
            confidence=confidence,
        )


__all__ = ["Morpheme", "MorphemeAnalysis", "MorphemeSegmenter"]
