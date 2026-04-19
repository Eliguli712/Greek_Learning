from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
import unicodedata

from src.lexeme import LexemeAnalysis, LexemeNormalizer
from src.morpheme import MorphemeAnalysis, MorphemeSegmenter


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
    return dict(data) if isinstance(data, Mapping) else {}


DEFAULT_SEMANTIC_RULES: Dict[str, Any] = {
    "pos_semantics": {
        "NOUN": {
            "semantic_type": "entity",
            "semantic_role": "referent",
            "features": {
                "interpretable": True,
                "denotation": "thing",
            },
        },
        "VERB": {
            "semantic_type": "event",
            "semantic_role": "predicate",
            "features": {
                "interpretable": True,
                "denotation": "eventuality",
                "requires_arguments": True,
            },
        },
        "ADJ": {
            "semantic_type": "property",
            "semantic_role": "modifier",
            "features": {
                "interpretable": True,
                "denotation": "attribute",
            },
        },
        "ADV": {
            "semantic_type": "manner",
            "semantic_role": "modifier",
            "features": {
                "interpretable": True,
                "denotation": "event_modifier",
            },
        },
        "PRON": {
            "semantic_type": "indexical",
            "semantic_role": "referent",
            "features": {
                "interpretable": True,
                "denotation": "participant",
            },
        },
        "ADP": {
            "semantic_type": "relation",
            "semantic_role": "linker",
            "features": {
                "interpretable": True,
                "denotation": "relational",
            },
        },
        "CONJ": {
            "semantic_type": "connector",
            "semantic_role": "linker",
            "features": {
                "interpretable": True,
                "denotation": "coordination",
            },
        },
        "PART": {
            "semantic_type": "operator",
            "semantic_role": "scope_marker",
            "features": {
                "interpretable": True,
                "denotation": "scope",
            },
        },
        "X": {
            "semantic_type": "unknown",
            "semantic_role": "unknown",
            "features": {
                "interpretable": False,
            },
        },
    },
    "lemma_semantics": {
        "θεός": {
            "semantic_type": "entity",
            "semantic_role": "referent",
            "semantic_class": "deity",
            "features": {
                "animacy": "animate",
                "agency_bias": 0.95,
            },
        },
        "ἄνθρωπος": {
            "semantic_type": "entity",
            "semantic_role": "referent",
            "semantic_class": "human",
            "features": {
                "animacy": "animate",
                "agency_bias": 0.95,
            },
        },
        "λόγος": {
            "semantic_type": "entity",
            "semantic_role": "referent",
            "semantic_class": "discourse",
            "features": {
                "animacy": "inanimate",
                "abstract": True,
            },
        },
        "γράφω": {
            "semantic_type": "event",
            "semantic_role": "predicate",
            "semantic_class": "communication",
            "features": {
                "requires_agent": True,
                "requires_theme": True,
            },
        },
    },
    "morpheme_type_semantics": {
        "prefix": {
            "semantic_role": "modifier",
            "features": {
                "bound": True,
            },
        },
        "root": {
            "semantic_role": "semantic_core",
            "features": {
                "bound": False,
            },
        },
        "inflectional_ending": {
            "semantic_role": "grammatical_marker",
            "features": {
                "bound": True,
            },
        },
        "suffix": {
            "semantic_role": "derivational_marker",
            "features": {
                "bound": True,
            },
        },
    },
}


@dataclass
class SemanticToken:
    token: str
    lemma: str
    lexeme: str
    canonical_form: str
    pos: str
    transliteration: str
    semantic_type: str
    semantic_role: str
    semantic_class: Optional[str] = None
    features: Dict[str, Any] = field(default_factory=dict)
    source_components: List[str] = field(default_factory=list)
    component_lemmas: List[str] = field(default_factory=list)
    morpheme_roles: List[Dict[str, Any]] = field(default_factory=list)
    strategy: str = "rule"
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "token": self.token,
            "lemma": self.lemma,
            "lexeme": self.lexeme,
            "canonical_form": self.canonical_form,
            "pos": self.pos,
            "transliteration": self.transliteration,
            "semantic_type": self.semantic_type,
            "semantic_role": self.semantic_role,
            "strategy": self.strategy,
            "confidence": round(self.confidence, 3),
        }

        if self.semantic_class:
            payload["semantic_class"] = self.semantic_class
        if self.features:
            payload["features"] = self.features
        if self.source_components:
            payload["source_components"] = self.source_components
        if self.component_lemmas:
            payload["component_lemmas"] = self.component_lemmas
        if self.morpheme_roles:
            payload["morpheme_roles"] = self.morpheme_roles
        if self.notes:
            payload["notes"] = self.notes

        return payload


class SemanticTokenizer:
    """
    Map normalized lexemes into semantic tokens.

    This module is intentionally rule-based and transparent:
    token -> morpheme analysis -> lexeme analysis -> semantic token
    """

    def __init__(
        self,
        resources: Optional[Mapping[str, Any]] = None,
        project_root: Optional[Path] = None,
        segmenter: Optional[MorphemeSegmenter] = None,
        lexeme_normalizer: Optional[LexemeNormalizer] = None,
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        self.segmenter = segmenter or MorphemeSegmenter(
            resources=resources,
            project_root=self.project_root,
        )
        self.lexeme_normalizer = lexeme_normalizer or LexemeNormalizer(
            resources=resources,
            project_root=self.project_root,
            segmenter=self.segmenter,
        )

        semantic_resource: Dict[str, Any] = {}
        if resources and isinstance(resources.get("semantic_rules"), Mapping):
            semantic_resource = dict(resources["semantic_rules"])
        else:
            semantic_resource = _load_json_resource(
                self.project_root / "data/semantic/semantic_rules.json"
            )

        merged_rules = _deep_merge(DEFAULT_SEMANTIC_RULES, semantic_resource)
        self.pos_semantics = dict(merged_rules.get("pos_semantics", {}))
        self.lemma_semantics = dict(merged_rules.get("lemma_semantics", {}))
        self.morpheme_type_semantics = dict(merged_rules.get("morpheme_type_semantics", {}))

    def analyze(
        self,
        token: str,
        lexeme_analysis: Optional[LexemeAnalysis] = None,
        morpheme_analysis: Optional[MorphemeAnalysis] = None,
        language: str = "ancient_greek",
    ) -> SemanticToken:
        return self.tokenize(
            token=token,
            lexeme_analysis=lexeme_analysis,
            morpheme_analysis=morpheme_analysis,
            language=language,
        )

    def tokenize(
        self,
        token: str,
        lexeme_analysis: Optional[LexemeAnalysis] = None,
        morpheme_analysis: Optional[MorphemeAnalysis] = None,
        language: str = "ancient_greek",
    ) -> SemanticToken:
        normalized_token = unicodedata.normalize("NFC", token.strip())
        key = _surface_key(normalized_token)
        if not key:
            raise ValueError("token must contain at least one alphabetic character")

        morph = morpheme_analysis or self.segmenter.analyze(normalized_token)
        lex = lexeme_analysis or self.lexeme_normalizer.analyze(
            normalized_token,
            morpheme_analysis=morph,
            language=language,
        )

        pos_rule = dict(self.pos_semantics.get(lex.pos, self.pos_semantics.get("X", {})))
        lemma_rule = dict(self.lemma_semantics.get(lex.lemma, {}))

        semantic_type = str(
            lemma_rule.get("semantic_type", pos_rule.get("semantic_type", "unknown"))
        )
        semantic_role = str(
            lemma_rule.get("semantic_role", pos_rule.get("semantic_role", "unknown"))
        )
        semantic_class = lemma_rule.get("semantic_class")

        features = _deep_merge(
            dict(pos_rule.get("features", {})),
            dict(lemma_rule.get("features", {})),
        )

        morpheme_roles = self._annotate_morphemes(morph)
        confidence = self._estimate_confidence(lex=lex, morph=morph, lemma_rule=lemma_rule)
        notes = self._build_notes(lex=lex, morph=morph, lemma_rule=lemma_rule)

        return SemanticToken(
            token=normalized_token,
            lemma=lex.lemma,
            lexeme=lex.lexeme,
            canonical_form=lex.canonical_form,
            pos=lex.pos,
            transliteration=lex.transliteration,
            semantic_type=semantic_type,
            semantic_role=semantic_role,
            semantic_class=str(semantic_class) if semantic_class is not None else None,
            features=features,
            source_components=list(lex.source_components),
            component_lemmas=list(lex.component_lemmas),
            morpheme_roles=morpheme_roles,
            strategy=f"semantic::{lex.strategy}",
            confidence=confidence,
            notes=notes,
        )

    def _annotate_morphemes(self, analysis: MorphemeAnalysis) -> List[Dict[str, Any]]:
        annotations: List[Dict[str, Any]] = []

        for morpheme in analysis.morphemes:
            rule = self.morpheme_type_semantics.get(morpheme.type, {})
            entry: Dict[str, Any] = {
                "form": morpheme.form,
                "type": morpheme.type,
                "gloss": morpheme.gloss,
                "semantic_role": rule.get("semantic_role", "unspecified"),
            }

            merged_features = _deep_merge(
                dict(rule.get("features", {})),
                dict(morpheme.features),
            )
            if morpheme.lemma:
                entry["lemma"] = morpheme.lemma
            if merged_features:
                entry["features"] = merged_features
            if morpheme.source:
                entry["source"] = morpheme.source
            entry["confidence"] = round(morpheme.confidence, 3)

            annotations.append(entry)

        return annotations

    def _estimate_confidence(
        self,
        lex: LexemeAnalysis,
        morph: MorphemeAnalysis,
        lemma_rule: Mapping[str, Any],
    ) -> float:
        score = 0.45 * lex.confidence + 0.35 * morph.candidate_score + 0.20
        if lemma_rule:
            score += 0.10
        return max(0.0, min(score, 0.99))

    def _build_notes(
        self,
        lex: LexemeAnalysis,
        morph: MorphemeAnalysis,
        lemma_rule: Mapping[str, Any],
    ) -> List[str]:
        notes: List[str] = []
        notes.extend(lex.notes)
        notes.extend(morph.notes)

        if lemma_rule:
            notes.append("semantic label derived from lemma-specific rule")
        else:
            notes.append("semantic label derived from POS-level default rule")

        if len(morph.morphemes) > 1:
            notes.append("semantic token preserves internal morpheme structure")

        return notes

    def tokenize_to_dict(
        self,
        token: str,
        lexeme_analysis: Optional[LexemeAnalysis] = None,
        morpheme_analysis: Optional[MorphemeAnalysis] = None,
        language: str = "ancient_greek",
    ) -> Dict[str, Any]:
        return self.tokenize(
            token=token,
            lexeme_analysis=lexeme_analysis,
            morpheme_analysis=morpheme_analysis,
            language=language,
        ).as_dict()


if __name__ == "__main__":
    tokenizer = SemanticTokenizer()
    sample_tokens = ["λόγος", "θεός", "ἄνθρωπος", "γράφω"]

    for token in sample_tokens:
        try:
            result = tokenizer.tokenize_to_dict(token)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"{token}: {exc}")
