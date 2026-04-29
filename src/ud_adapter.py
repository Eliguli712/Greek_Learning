"""Universal Dependencies adapter for building semantic DAGs from real Greek data.

The rule-based raw-text pipeline in this repo is intentionally small. This
module provides a stronger evaluation path: use UD CoNLL-U annotations as the
upstream linguistic analysis, convert them into this project's semantic-token
shape, then reuse the existing DAGBuilder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from src.dag import DAGBuilder, SemanticDAG
from src.normalize import surface_key
from src.syntax_logic import Relation


UPOS_SEMANTICS: Dict[str, Dict[str, Any]] = {
    "NOUN": {
        "semantic_type": "entity",
        "semantic_role": "referent",
        "semantic_class": "nominal",
    },
    "PROPN": {
        "semantic_type": "entity",
        "semantic_role": "referent",
        "semantic_class": "proper_name",
    },
    "PRON": {
        "semantic_type": "indexical",
        "semantic_role": "referent",
        "semantic_class": "participant",
    },
    "DET": {
        "semantic_type": "indexical",
        "semantic_role": "referent",
        "semantic_class": "determiner",
    },
    "NUM": {
        "semantic_type": "entity",
        "semantic_role": "referent",
        "semantic_class": "quantity",
    },
    "VERB": {
        "semantic_type": "event",
        "semantic_role": "predicate",
        "semantic_class": "eventuality",
    },
    "AUX": {
        "semantic_type": "event",
        "semantic_role": "predicate",
        "semantic_class": "auxiliary",
    },
    "ADJ": {
        "semantic_type": "property",
        "semantic_role": "modifier",
        "semantic_class": "attribute",
    },
    "ADV": {
        "semantic_type": "manner",
        "semantic_role": "modifier",
        "semantic_class": "event_modifier",
    },
    "ADP": {
        "semantic_type": "relation",
        "semantic_role": "linker",
        "semantic_class": "adposition",
    },
    "CCONJ": {
        "semantic_type": "connector",
        "semantic_role": "linker",
        "semantic_class": "coordination",
    },
    "SCONJ": {
        "semantic_type": "connector",
        "semantic_role": "linker",
        "semantic_class": "subordination",
    },
    "PART": {
        "semantic_type": "operator",
        "semantic_role": "scope_marker",
        "semantic_class": "particle",
    },
}


UD_TO_DAG_LABEL: Dict[str, str] = {
    "nsubj": "AGENT",
    "csubj": "AGENT",
    "nsubj:pass": "THEME",
    "csubj:pass": "THEME",
    "obj": "THEME",
    "iobj": "THEME",
    "obl:arg": "THEME",
    "xcomp": "COMPLEMENT",
    "ccomp": "COMPLEMENT",
    "acl": "COMPLEMENT",
    "advcl": "COMPLEMENT",
    "advcl:cmp": "COMPLEMENT",
    "obl": "COMPLEMENT",
    "amod": "MODIFIER",
    "nmod": "MODIFIER",
    "nummod": "MODIFIER",
    "advmod": "MODIFIER",
    "appos": "MODIFIER",
    "det": "MODIFIER",
    "obl:agent": "AGENT",
    "conj": "COORD",
}


GREEK_TO_LATIN: Dict[str, str] = {
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


@dataclass(frozen=True)
class UDToken:
    index: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: Dict[str, str]
    head: int
    deprel: str
    misc: str = "_"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "form": self.form,
            "lemma": self.lemma,
            "upos": self.upos,
            "xpos": self.xpos,
            "feats": self.feats,
            "head": self.head,
            "deprel": self.deprel,
            "misc": self.misc,
        }


@dataclass(frozen=True)
class UDSentence:
    sent_id: str
    text: str
    tokens: List[UDToken]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sent_id": self.sent_id,
            "text": self.text,
            "tokens": [token.as_dict() for token in self.tokens],
        }


@dataclass
class UDDAGResult:
    sentence: UDSentence
    semantic_tokens: List[Dict[str, Any]]
    relations: List[Relation]
    dag: SemanticDAG
    unmapped_dependencies: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.sentence.sent_id,
            "text": self.sentence.text,
            "tokens": [token.form for token in self.sentence.tokens],
            "ud_tokens": [token.as_dict() for token in self.sentence.tokens],
            "semantic_tokens": self.semantic_tokens,
            "relations": [relation.as_dict() for relation in self.relations],
            "dag": self.dag.to_dict(),
            "validation": self.dag.validation,
            "unmapped_dependencies": self.unmapped_dependencies,
        }


def read_conllu(path: Path) -> List[UDSentence]:
    """Read a CoNLL-U file into sentence objects, skipping punctuation tokens."""
    sentences: List[UDSentence] = []
    metadata: Dict[str, str] = {}
    tokens: List[UDToken] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                if tokens:
                    sentences.append(_sentence_from_parts(metadata, tokens))
                metadata = {}
                tokens = []
                continue

            if line.startswith("#"):
                if "=" in line:
                    key, value = line[1:].split("=", 1)
                    metadata[key.strip()] = value.strip()
                continue

            token = _token_from_conllu_line(line)
            if token is not None:
                tokens.append(token)

    if tokens:
        sentences.append(_sentence_from_parts(metadata, tokens))

    return sentences


def iter_conllu(paths: Sequence[Path], max_sentences: Optional[int] = None) -> Iterable[UDSentence]:
    """Yield sentences from one or more CoNLL-U files."""
    yielded = 0
    for path in paths:
        for sentence in read_conllu(path):
            if max_sentences is not None and yielded >= max_sentences:
                return
            yield sentence
            yielded += 1


def semantic_tokens_from_ud(sentence: UDSentence, relations: Sequence[Relation]) -> List[Dict[str, Any]]:
    """Convert UD tokens to the semantic-token dictionaries expected by DAGBuilder."""
    argument_heads = {
        relation.src
        for relation in relations
        if relation.label in {"AGENT", "THEME", "EXPERIENCER"}
    }

    semantic_tokens: List[Dict[str, Any]] = []
    for position, token in enumerate(sentence.tokens):
        rule = UPOS_SEMANTICS.get(
            token.upos,
            {
                "semantic_type": "unknown",
                "semantic_role": "unknown",
                "semantic_class": None,
            },
        )
        features: Dict[str, Any] = {
            "upos": token.upos,
            "xpos": token.xpos,
            "ud_features": token.feats,
            "ud_head": token.head,
            "ud_deprel": token.deprel,
            "source": "universal_dependencies",
            "interpretable": rule.get("semantic_type") != "unknown",
        }
        if position in argument_heads:
            features["requires_arguments"] = True

        semantic_tokens.append(
            {
                "token": token.form,
                "lemma": token.lemma,
                "lexeme": token.form,
                "canonical_form": transliterate_greek(token.lemma),
                "pos": token.upos,
                "transliteration": transliterate_greek(token.form),
                "semantic_type": rule.get("semantic_type", "unknown"),
                "semantic_role": rule.get("semantic_role", "unknown"),
                "semantic_class": rule.get("semantic_class"),
                "features": features,
                "source_components": [token.form],
                "component_lemmas": [token.lemma],
                "morpheme_roles": [],
                "strategy": "ud_gold_adapter",
                "confidence": 1.0,
                "notes": [
                    "semantic token derived from Universal Dependencies gold annotation"
                ],
            }
        )

    return semantic_tokens


def relations_from_ud(sentence: UDSentence) -> Tuple[List[Relation], List[Dict[str, Any]]]:
    """Map selected UD dependencies into this repo's DAG relation inventory."""
    index_to_position = {token.index: position for position, token in enumerate(sentence.tokens)}
    relations: List[Relation] = []
    unmapped: List[Dict[str, Any]] = []

    for token in sentence.tokens:
        if token.head == 0:
            continue

        src = index_to_position.get(token.head)
        dst = index_to_position.get(token.index)
        if src is None or dst is None:
            continue

        label = UD_TO_DAG_LABEL.get(token.deprel)
        if label is None:
            unmapped.append(
                {
                    "dependent": token.form,
                    "dependent_index": token.index,
                    "head": token.head,
                    "deprel": token.deprel,
                }
            )
            continue

        relations.append(
            Relation(
                src=src,
                dst=dst,
                label=label,
                confidence=1.0,
                rule=f"ud::{token.deprel}",
            )
        )

    return relations, unmapped


def dag_from_ud(sentence: UDSentence) -> UDDAGResult:
    """Build a semantic DAG from one UD-annotated sentence."""
    relations, unmapped = relations_from_ud(sentence)
    semantic_tokens = semantic_tokens_from_ud(sentence, relations)
    dag = DAGBuilder().build(semantic_tokens, relations)
    return UDDAGResult(
        sentence=sentence,
        semantic_tokens=semantic_tokens,
        relations=relations,
        dag=dag,
        unmapped_dependencies=unmapped,
    )


def transliterate_greek(text: str) -> str:
    key = surface_key(text)
    output: List[str] = []
    index = 0
    while index < len(key):
        pair = key[index : index + 2]
        if pair in GREEK_TO_LATIN:
            output.append(GREEK_TO_LATIN[pair])
            index += 2
            continue
        char = key[index]
        output.append(GREEK_TO_LATIN.get(char, char))
        index += 1
    return "".join(piece for piece in output if piece.isascii())


def _sentence_from_parts(metadata: Mapping[str, str], tokens: List[UDToken]) -> UDSentence:
    text = metadata.get("text") or " ".join(token.form for token in tokens)
    sent_id = metadata.get("sent_id") or metadata.get("newdoc id") or f"sent_{len(tokens)}"
    return UDSentence(sent_id=sent_id, text=text, tokens=list(tokens))


def _token_from_conllu_line(line: str) -> Optional[UDToken]:
    fields = line.split("\t")
    if len(fields) != 10:
        return None

    token_id = fields[0]
    if "-" in token_id or "." in token_id:
        return None

    upos = fields[3]
    form = fields[1]
    if upos == "PUNCT" or not surface_key(form):
        return None

    try:
        index = int(token_id)
        head = int(fields[6]) if fields[6].isdigit() else 0
    except ValueError:
        return None

    return UDToken(
        index=index,
        form=form,
        lemma=fields[2],
        upos=upos,
        xpos=fields[4],
        feats=_parse_feats(fields[5]),
        head=head,
        deprel=fields[7],
        misc=fields[9],
    )


def _parse_feats(raw: str) -> Dict[str, str]:
    if not raw or raw == "_":
        return {}

    features: Dict[str, str] = {}
    for item in raw.split("|"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        features[key] = value
    return features


__all__ = [
    "UDDAGResult",
    "UDSentence",
    "UDToken",
    "UD_TO_DAG_LABEL",
    "UPOS_SEMANTICS",
    "dag_from_ud",
    "iter_conllu",
    "read_conllu",
    "relations_from_ud",
    "semantic_tokens_from_ud",
    "transliterate_greek",
]
