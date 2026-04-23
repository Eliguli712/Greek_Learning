from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
import unicodedata

from src.morpheme import MorphemeAnalysis, MorphemeSegmenter
from src.similarity_comparator import SimilarityComparator


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


DEFAULT_GREEK_LEMMAS: Dict[str, Any] = {
    "whole_word": {
        "μεγαλοδους": {
            "lemma": "ὀδούς",
            "lexeme": "μεγαλόδους",
            "canonical_form": "megalodous",
            "pos": "NOUN",
            "source_components": ["μεγαλο-", "οδους"],
        }
    },
    "components": {
        "μεγαλο": {
            "lemma": "μέγας",
            "pos": "ADJ",
            "gloss": "large",
        },
        "οδους": {
            "lemma": "ὀδούς",
            "pos": "NOUN",
            "gloss": "tooth",
        },
        "λογος": {
            "lemma": "λόγος",
            "pos": "NOUN",
            "gloss": "word / account / reason",
        },
        "θεος": {
            "lemma": "θεός",
            "pos": "NOUN",
            "gloss": "god",
        },
        "ανθρωπος": {
            "lemma": "ἄνθρωπος",
            "pos": "NOUN",
            "gloss": "human / person",
        },
        "γραφ": {
            "lemma": "γράφω",
            "pos": "VERB",
            "gloss": "write",
        },
    },
}

DEFAULT_TRANSLITERATION: Dict[str, str] = {
    "αι": "ai",
    "ει": "ei",
    "οι": "oi",
    "ου": "ou",
    "αυ": "au",
    "ευ": "eu",
    "ηυ": "eu",
    "γγ": "ng",
    "γκ": "gk",
    "γχ": "nch",
    "β": "b",
    "γ": "g",
    "δ": "d",
    "ζ": "z",
    "θ": "th",
    "κ": "k",
    "λ": "l",
    "μ": "m",
    "ν": "n",
    "ξ": "x",
    "π": "p",
    "ρ": "r",
    "σ": "s",
    "ς": "s",
    "τ": "t",
    "φ": "ph",
    "χ": "ch",
    "ψ": "ps",
    "α": "a",
    "ε": "e",
    "η": "e",
    "ι": "i",
    "ο": "o",
    "υ": "u",
    "ω": "o",
}


@dataclass
class LexemeAnalysis:
    token: str
    lemma: str
    lexeme: str
    canonical_form: str
    pos: str
    transliteration: str
    source_components: List[str] = field(default_factory=list)
    component_lemmas: List[str] = field(default_factory=list)
    strategy: str = "rule"
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "lemma": self.lemma,
            "lexeme": self.lexeme,
            "canonical_form": self.canonical_form,
            "pos": self.pos,
            "transliteration": self.transliteration,
            "strategy": self.strategy,
            "confidence": round(self.confidence, 3),
        }
        if self.source_components:
            payload["source_components"] = self.source_components
        if self.component_lemmas:
            payload["component_lemmas"] = self.component_lemmas
        if self.notes:
            payload["notes"] = self.notes
        return payload


class LexemeNormalizer:
    """
    Normalize a morpheme bundle to a lemma-centered lexical representation.

    This follows the document specification: lemma lookup, variant
    normalization, terminology standardization, and transliteration storage.
    """

    def __init__(
        self,
        resources: Optional[Mapping[str, Any]] = None,
        project_root: Optional[Path] = None,
        segmenter: Optional[MorphemeSegmenter] = None,
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        self.segmenter = segmenter or MorphemeSegmenter(resources=resources, project_root=self.project_root)

        greek_resource = {}
        translit_resource = {}

        if resources and isinstance(resources.get("greek_lemmas"), Mapping):
            greek_resource = dict(resources["greek_lemmas"])
        else:
            greek_resource = _load_json_resource(self.project_root / "data/lexica/greek_lemmas.json")

        if resources and isinstance(resources.get("greek_to_latin"), Mapping):
            translit_resource = dict(resources["greek_to_latin"])
        else:
            translit_resource = _load_json_resource(self.project_root / "data/translit/greek_to_latin.json")

        merged_lemmas = _deep_merge(DEFAULT_GREEK_LEMMAS, greek_resource)
        merged_translit = _deep_merge(DEFAULT_TRANSLITERATION, translit_resource)

        self.whole_word = dict(merged_lemmas.get("whole_word", {}))
        self.components = dict(merged_lemmas.get("components", {}))
        self.transliteration = dict(merged_translit)
        self.similarity = SimilarityComparator.from_components(self.components)

    def analyze(
        self,
        token: str,
        morpheme_analysis: Optional[MorphemeAnalysis] = None,
        language: str = "ancient_greek",
    ) -> LexemeAnalysis:
        return self.normalize(token, morpheme_analysis=morpheme_analysis, language=language)

    def normalize(
        self,
        token: str,
        morpheme_analysis: Optional[MorphemeAnalysis] = None,
        language: str = "ancient_greek",
    ) -> LexemeAnalysis:
        normalized_token = unicodedata.normalize("NFC", token.strip())
        key = _surface_key(normalized_token)
        if not key:
            raise ValueError("token must contain at least one alphabetic character")

        analysis = morpheme_analysis or self.segmenter.analyze(normalized_token)
        if key in self.whole_word:
            return self._from_whole_word(normalized_token, analysis, key)
        return self._from_components(normalized_token, analysis, language=language)

    def _from_whole_word(
        self,
        token: str,
        analysis: MorphemeAnalysis,
        key: str,
    ) -> LexemeAnalysis:
        record = self.whole_word[key]
        source_components = list(record.get("source_components", [item.form for item in analysis.morphemes]))
        component_lemmas = [item.lemma for item in analysis.morphemes if item.lemma]

        transliteration = str(record.get("canonical_form", self._transliterate(token)))
        return LexemeAnalysis(
            token=token,
            lemma=str(record.get("lemma", token)),
            lexeme=str(record.get("lexeme", token)),
            canonical_form=str(record.get("canonical_form", transliteration)),
            pos=str(record.get("pos", self._infer_pos(analysis))),
            transliteration=transliteration,
            source_components=source_components,
            component_lemmas=component_lemmas,
            strategy="whole_word_lexicon",
            confidence=0.99,
            notes=[
                "resolved via direct lexeme override",
                "preserved internal lexical source components for downstream semantic typing",
            ],
        )

    def _from_components(
        self,
        token: str,
        analysis: MorphemeAnalysis,
        language: str,
    ) -> LexemeAnalysis:
        component_forms = [item.form for item in analysis.morphemes]
        component_lemmas: List[str] = []
        notes = ["derived lemma from segmented morpheme bundle"]

        head_entry: Optional[Mapping[str, Any]] = None
        head_lemma: Optional[str] = None
        for morpheme in reversed(analysis.morphemes):
            key = _surface_key(morpheme.form.rstrip("-"))
            if key in self.components:
                head_entry = self.components[key]
                head_lemma = str(head_entry.get("lemma", morpheme.lemma or token))
                break
            if morpheme.lemma:
                head_lemma = morpheme.lemma
                break

        for morpheme in analysis.morphemes:
            morpheme_key = _surface_key(morpheme.form.rstrip("-"))
            if morpheme_key in self.components:
                component_lemmas.append(str(self.components[morpheme_key].get("lemma", morpheme.form)))
            elif morpheme.lemma:
                component_lemmas.append(morpheme.lemma)

        strategy = "component_lookup"
        confidence = 0.84 if head_entry or head_lemma else 0.55

        # Ambiguity-resolution fallback: when neither the lexicon nor any
        # segmented morpheme supplies a lemma, rank the token against known
        # component keys by character n-gram TF-IDF cosine + difflib ratio.
        if head_entry is None and head_lemma is None and len(self.similarity) > 0:
            best = self.similarity.best(token)
            if best is not None:
                head_entry = best.entry
                head_lemma = best.lemma
                strategy = "similarity_fallback"
                confidence = round(0.45 + 0.4 * best.score, 3)
                notes.append(
                    f"similarity fallback matched '{best.key}' (score={round(best.score, 3)})"
                )

        lemma = head_lemma or token
        pos = str(head_entry.get("pos", self._infer_pos(analysis))) if head_entry else self._infer_pos(analysis)
        transliteration = self._transliterate(token, language=language)

        if component_forms:
            notes.append("retained source components alongside the normalized head lemma")

        return LexemeAnalysis(
            token=token,
            lemma=lemma,
            lexeme=token,
            canonical_form=transliteration,
            pos=pos,
            transliteration=transliteration,
            source_components=component_forms,
            component_lemmas=component_lemmas,
            strategy="component_lookup",
            confidence=confidence,
            notes=notes,
        )

    def _infer_pos(self, analysis: MorphemeAnalysis) -> str:
        for morpheme in reversed(analysis.morphemes):
            pos = morpheme.features.get("pos")
            if isinstance(pos, str) and pos:
                return pos

        if analysis.morphemes and analysis.morphemes[-1].type == "inflectional_ending":
            return "NOUN"

        if any(item.type == "combining_form" for item in analysis.morphemes):
            return "NOUN"

        return "X"

    def _transliterate(self, token: str, language: str = "ancient_greek") -> str:
        if language != "ancient_greek":
            return _surface_key(token)

        text = _strip_accents(token).lower()
        output: List[str] = []
        index = 0
        while index < len(text):
            pair = text[index : index + 2]
            if pair in self.transliteration:
                output.append(self.transliteration[pair])
                index += 2
                continue

            char = text[index]
            output.append(self.transliteration.get(char, char))
            index += 1

        return "".join(piece for piece in output if piece.isascii())


__all__ = ["LexemeAnalysis", "LexemeNormalizer"]
