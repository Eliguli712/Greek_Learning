"""
Rule-based relation predictor over semantic tokens.

This is intentionally local and inspectable: each edge carries the rule that
produced it. Edges are expressed as (src_index, dst_index, label, confidence).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.normalize import surface_key


@dataclass
class Relation:
    src: int
    dst: int
    label: str
    confidence: float
    rule: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "rule": self.rule,
        }


class RelationPredictor:
    """
    Local, rule-based relation inference.

    Heuristics (Ancient Greek is relatively free-order, so the rules are
    deliberately conservative):
      - For each predicate (semantic_type=event), pick:
          * AGENT: nearest preceding entity if available, else nearest
            following entity that is not already taken as theme.
          * THEME: nearest following entity if available, else any remaining
            entity not already taken as agent.
      - Modifiers (semantic_role=modifier): attach to the nearest entity
        (preferring the following one), label MODIFIER.
      - Linkers (semantic_role=linker): COORD between immediate neighbors
        when both neighbors exist.
      - Operators (semantic_role=scope_marker): SCOPE over the nearest
        following predicate or entity.
    """

    def predict(self, semantic_tokens: Sequence[Dict[str, Any]]) -> List[Relation]:
        relations: List[Relation] = []
        n = len(semantic_tokens)
        if n == 0:
            return relations

        for i, token in enumerate(semantic_tokens):
            stype = token.get("semantic_type")
            srole = token.get("semantic_role")

            if stype == "event":
                relations.extend(self._predicate_arguments(i, semantic_tokens))
            elif srole == "modifier" and stype not in {"event"}:
                edge = self._modifier_attachment(i, semantic_tokens)
                if edge is not None:
                    relations.append(edge)
            elif srole == "linker" and stype == "connector":
                edge = self._coordination(i, semantic_tokens)
                if edge is not None:
                    relations.append(edge)
            elif srole == "scope_marker":
                edge = self._scope(i, semantic_tokens)
                if edge is not None:
                    relations.append(edge)

        return relations

    # ------------------------------------------------------------------ #
    # individual rules                                                   #
    # ------------------------------------------------------------------ #

    def _predicate_arguments(
        self,
        predicate_index: int,
        tokens: Sequence[Dict[str, Any]],
    ) -> List[Relation]:
        edges: List[Relation] = []
        used: set = set()

        agent_index = self._find_case_entity(tokens, predicate_index, case="nom", exclude=used)
        if agent_index is None:
            agent_index = self._find_entity(tokens, predicate_index, direction=-1, exclude=used)
        if agent_index is None:
            agent_index = self._find_entity(tokens, predicate_index, direction=+1, exclude=used)
        if agent_index is not None:
            used.add(agent_index)
            edges.append(
                Relation(
                    src=predicate_index,
                    dst=agent_index,
                    label="AGENT",
                    confidence=0.7,
                    rule="predicate_agent_local",
                )
            )

        theme_index = self._find_case_entity(tokens, predicate_index, case="acc", exclude=used)
        if theme_index is None:
            theme_index = self._find_entity(tokens, predicate_index, direction=+1, exclude=used)
        if theme_index is None:
            theme_index = self._find_entity(tokens, predicate_index, direction=-1, exclude=used)
        if theme_index is not None:
            used.add(theme_index)
            edges.append(
                Relation(
                    src=predicate_index,
                    dst=theme_index,
                    label="THEME",
                    confidence=0.65,
                    rule="predicate_theme_local",
                )
            )

        return edges

    def _modifier_attachment(
        self,
        modifier_index: int,
        tokens: Sequence[Dict[str, Any]],
    ) -> Optional[Relation]:
        target = self._find_entity(tokens, modifier_index, direction=+1, exclude=set())
        if target is None:
            target = self._find_entity(tokens, modifier_index, direction=-1, exclude=set())
        if target is None:
            return None
        return Relation(
            src=target,
            dst=modifier_index,
            label="MODIFIER",
            confidence=0.6,
            rule="modifier_nearest_entity",
        )

    def _coordination(
        self,
        connector_index: int,
        tokens: Sequence[Dict[str, Any]],
    ) -> Optional[Relation]:
        left = connector_index - 1
        right = connector_index + 1
        if left < 0 or right >= len(tokens):
            return None
        return Relation(
            src=left,
            dst=right,
            label="COORD",
            confidence=0.6,
            rule="connector_neighbours",
        )

    def _scope(
        self,
        operator_index: int,
        tokens: Sequence[Dict[str, Any]],
    ) -> Optional[Relation]:
        target = self._find_any(tokens, operator_index, direction=+1)
        if target is None:
            target = self._find_any(tokens, operator_index, direction=-1)
        if target is None:
            return None
        return Relation(
            src=operator_index,
            dst=target,
            label="SCOPE",
            confidence=0.55,
            rule="operator_scope_neighbour",
        )

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #

    def _find_case_entity(
        self,
        tokens: Sequence[Dict[str, Any]],
        origin: int,
        case: str,
        exclude: set,
    ) -> Optional[int]:
        candidates: List[int] = []
        for index, token in enumerate(tokens):
            if index == origin or index in exclude:
                continue
            if token.get("semantic_type") not in {"entity", "indexical"}:
                continue
            if self._case_hint(token) == case:
                candidates.append(index)
        if not candidates:
            return None
        return min(candidates, key=lambda index: abs(index - origin))

    def _case_hint(self, token: Dict[str, Any]) -> Optional[str]:
        features = token.get("features") or {}
        ud_features = features.get("ud_features") or {}
        explicit_case = ud_features.get("Case") or features.get("case")
        if isinstance(explicit_case, str):
            value = explicit_case.lower()
            if value.startswith("nom"):
                return "nom"
            if value.startswith("acc"):
                return "acc"

        key = surface_key(str(token.get("token") or token.get("lemma") or ""))
        if key.endswith(("ον", "αν", "ην", "ους")):
            return "acc"
        if key.endswith(("ος", "ης", "ας", "οι", "αι")):
            return "nom"
        return None

    def _find_entity(
        self,
        tokens: Sequence[Dict[str, Any]],
        origin: int,
        direction: int,
        exclude: set,
    ) -> Optional[int]:
        index = origin + direction
        while 0 <= index < len(tokens):
            if index not in exclude:
                tok = tokens[index]
                if tok.get("semantic_type") in {"entity", "indexical"}:
                    return index
            index += direction
        return None

    def _find_any(
        self,
        tokens: Sequence[Dict[str, Any]],
        origin: int,
        direction: int,
    ) -> Optional[int]:
        index = origin + direction
        while 0 <= index < len(tokens):
            if tokens[index].get("semantic_type") not in {"unknown"}:
                return index
            index += direction
        return None


__all__ = ["Relation", "RelationPredictor"]
