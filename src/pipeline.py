"""
End-to-end semantic compiler.

Stages (each layer outputs structured JSON consumed by the next):

    raw text
      -> phoneme normalization (syllabification + phoneme split)
      -> morpheme segmentation
      -> lexeme normalization
      -> semantic tokenization
      -> relation prediction
      -> DAG construction + validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.dag import DAGBuilder, SemanticDAG
from src.lexeme import LexemeNormalizer
from src.morpheme import MorphemeSegmenter
from src.normalize import sentence_split, surface_key, word_tokens
from src.phoneme import split_to_phonemes, syllabify_word
from src.semantic_tokenizer import SemanticTokenizer
from src.syntax_logic import RelationPredictor


@dataclass
class CompilerResult:
    text: str
    language: str
    sentences: List[str]
    tokens: List[str]
    phoneme_layer: List[Dict[str, Any]] = field(default_factory=list)
    morpheme_layer: List[Dict[str, Any]] = field(default_factory=list)
    lexeme_layer: List[Dict[str, Any]] = field(default_factory=list)
    semantic_tokens: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    dag: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "sentences": self.sentences,
            "tokens": self.tokens,
            "phoneme_layer": self.phoneme_layer,
            "morpheme_layer": self.morpheme_layer,
            "lexeme_layer": self.lexeme_layer,
            "semantic_tokens": self.semantic_tokens,
            "relations": self.relations,
            "dag": self.dag,
            "validation": self.validation,
            "notes": self.notes,
        }


class SemanticCompiler:
    """Top-level orchestrator binding all six pipeline stages together."""

    def __init__(
        self,
        resources: Optional[Mapping[str, Any]] = None,
        project_root: Optional[Path] = None,
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1])
        self.segmenter = MorphemeSegmenter(resources=resources, project_root=self.project_root)
        self.lexeme_normalizer = LexemeNormalizer(
            resources=resources,
            project_root=self.project_root,
            segmenter=self.segmenter,
        )
        self.semantic_tokenizer = SemanticTokenizer(
            resources=resources,
            project_root=self.project_root,
            segmenter=self.segmenter,
            lexeme_normalizer=self.lexeme_normalizer,
        )
        self.relation_predictor = RelationPredictor()
        self.dag_builder = DAGBuilder()

    def analyze(self, text: str, language: str = "ancient_greek") -> CompilerResult:
        sentences = sentence_split(text)
        tokens = self._tokenize(sentences if sentences else [text])

        result = CompilerResult(
            text=text,
            language=language,
            sentences=sentences,
            tokens=tokens,
        )

        for token in tokens:
            phoneme_entry = self._phoneme_entry(token)
            result.phoneme_layer.append(phoneme_entry)

            try:
                morph = self.segmenter.analyze(
                    token,
                    syllables=phoneme_entry.get("syllables"),
                    phonemes=[{"symbol": p} for p in phoneme_entry.get("phonemes", [])],
                )
            except Exception as exc:
                result.morpheme_layer.append({"token": token, "error": str(exc)})
                result.lexeme_layer.append({"token": token, "error": str(exc)})
                result.semantic_tokens.append({"token": token, "error": str(exc)})
                result.notes.append(f"morpheme stage failed for {token!r}: {exc}")
                continue

            result.morpheme_layer.append(morph.as_dict())

            try:
                lex = self.lexeme_normalizer.normalize(
                    token,
                    morpheme_analysis=morph,
                    language=language,
                )
            except Exception as exc:
                result.lexeme_layer.append({"token": token, "error": str(exc)})
                result.semantic_tokens.append({"token": token, "error": str(exc)})
                result.notes.append(f"lexeme stage failed for {token!r}: {exc}")
                continue

            result.lexeme_layer.append(lex.as_dict())

            try:
                sem = self.semantic_tokenizer.tokenize(
                    token,
                    lexeme_analysis=lex,
                    morpheme_analysis=morph,
                    language=language,
                )
            except Exception as exc:
                result.semantic_tokens.append({"token": token, "error": str(exc)})
                result.notes.append(f"semantic stage failed for {token!r}: {exc}")
                continue

            result.semantic_tokens.append(sem.as_dict())

        usable_semantic = [tok for tok in result.semantic_tokens if "error" not in tok]
        index_map = [i for i, tok in enumerate(result.semantic_tokens) if "error" not in tok]

        relations = self.relation_predictor.predict(usable_semantic)

        # Re-map relation indices back to the original token positions so that
        # the DAG nodes line up with `result.tokens`.
        remapped: List = []
        from src.syntax_logic import Relation
        for rel in relations:
            remapped.append(
                Relation(
                    src=index_map[rel.src],
                    dst=index_map[rel.dst],
                    label=rel.label,
                    confidence=rel.confidence,
                    rule=rel.rule,
                )
            )

        # Build a DAG using all semantic token slots so node ids align with
        # token positions; failed tokens become "unknown" placeholder nodes.
        node_inputs: List[Dict[str, Any]] = []
        for tok in result.semantic_tokens:
            if "error" in tok:
                node_inputs.append(
                    {
                        "token": tok.get("token"),
                        "lemma": tok.get("token"),
                        "pos": "X",
                        "semantic_type": "unknown",
                        "semantic_role": "unknown",
                        "semantic_class": None,
                        "transliteration": surface_key(tok.get("token", "")),
                        "features": {},
                        "confidence": 0.0,
                    }
                )
            else:
                node_inputs.append(tok)

        dag: SemanticDAG = self.dag_builder.build(node_inputs, remapped)
        result.relations = [edge for edge in dag.edges]
        result.dag = dag.to_dict()
        result.validation = dag.validation
        return result

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #

    def _tokenize(self, sentences: Sequence[str]) -> List[str]:
        out: List[str] = []
        for sentence in sentences:
            for token in word_tokens(sentence):
                if any(ch.isalpha() for ch in token):
                    out.append(token)
        return out

    def _phoneme_entry(self, token: str) -> Dict[str, Any]:
        try:
            syllables = syllabify_word(token)
        except Exception:
            syllables = []
        try:
            phonemes = split_to_phonemes(token)
        except Exception:
            phonemes = []
        return {
            "word": token,
            "syllables": syllables,
            "phonemes": phonemes,
        }


__all__ = ["SemanticCompiler", "CompilerResult"]
