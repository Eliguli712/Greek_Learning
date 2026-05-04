"""
Rule-based relation predictor over semantic tokens.

This is intentionally local and inspectable: each edge carries the rule that
produced it. Edges are expressed as (src_index, dst_index, label, confidence).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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

    DISCOURSE_PARTICLE_KEYS = {
        "\u03b1\u03bb\u03bb",  # all'
        "\u03b1\u03bb\u03bb\u03b1",  # alla
        "\u03b1\u03bd",  # an
        "\u03b3\u03b1\u03c1",  # gar
        "\u03b4\u03b5",  # de
        "\u03b4\u03b7",  # de
        "\u03ba\u03b1\u03b9",  # kai
        "\u03bc\u03b5\u03bd",  # men
    }
    def __init__(
        self,
        project_root: Optional[Path] = None,
        use_classifier: bool = False,
        filter_discourse_particles: bool = False,
        infer_baseline_complements: bool = False,
        suppress_nonfinite_agents: bool = False,
        enhance_baseline_relations: bool = False,
    ) -> None:
        self.classifier = None
        self.filter_discourse_particles = filter_discourse_particles
        self.infer_baseline_complements = infer_baseline_complements
        self.suppress_nonfinite_agents = suppress_nonfinite_agents
        self.enhance_baseline_relations = enhance_baseline_relations
        if use_classifier:
            from src.relation_classifier import relation_classifier_from_project

            root = Path(project_root or Path(__file__).resolve().parents[1])
            self.classifier = relation_classifier_from_project(root)

    def predict(self, semantic_tokens: Sequence[Dict[str, Any]]) -> List[Relation]:
        if self.classifier is not None:
            classified = self.classifier.predict(semantic_tokens)
            if classified:
                return [
                    Relation(
                        src=item.src,
                        dst=item.dst,
                        label=item.label,
                        confidence=item.confidence,
                        rule="relation_classifier",
                    )
                    for item in classified
                ]

        relations: List[Relation] = []
        n = len(semantic_tokens)
        if n == 0:
            return relations

        for i, token in enumerate(semantic_tokens):
            stype = token.get("semantic_type")
            srole = token.get("semantic_role")

            if (
                stype == "event"
                and not self._is_function_word(token)
                and not self._is_auxiliary_predicate(token)
            ):
                relations.extend(self._predicate_arguments(i, semantic_tokens))
                if self.infer_baseline_complements:
                    relations.extend(self._predicate_complements(i, semantic_tokens))
            elif srole == "modifier" and stype not in {"event"} and not self._is_function_word(token):
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

        if self.enhance_baseline_relations:
            relations = self._enhance_baseline_relations(relations, semantic_tokens)

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

        passive_like = self._is_passive_like(tokens[predicate_index])
        strict_case = self._has_case_hints(tokens)

        if passive_like:
            theme_index = self._find_case_entity(tokens, predicate_index, case="nom", exclude=used)
            if theme_index is not None:
                used.add(theme_index)
                edges.append(
                    Relation(
                        src=predicate_index,
                        dst=theme_index,
                        label="THEME",
                        confidence=0.72,
                        rule="predicate_theme_passive_nom",
                    )
                )
        else:
            allow_agent = not (
                self.suppress_nonfinite_agents
                and self._verb_form(tokens[predicate_index]) != "Fin"
            )
            agent_index = None
            if allow_agent:
                agent_index = self._find_case_entity(tokens, predicate_index, case="nom", exclude=used)
            if allow_agent and agent_index is None and not strict_case:
                agent_index = self._find_entity(tokens, predicate_index, direction=-1, exclude=used)
            if allow_agent and agent_index is None and not strict_case:
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
        if theme_index is None and not strict_case:
            theme_index = self._find_entity(tokens, predicate_index, direction=+1, exclude=used)
        if theme_index is None and not strict_case:
            theme_index = self._find_entity(tokens, predicate_index, direction=-1, exclude=used)
        if theme_index is None and strict_case and self._allow_short_clause_theme_fallback(tokens, used):
            theme_index = self._find_entity(tokens, predicate_index, direction=-1, exclude=used)
            if theme_index is None:
                theme_index = self._find_entity(tokens, predicate_index, direction=+1, exclude=used)
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

    def _predicate_complements(
        self,
        predicate_index: int,
        tokens: Sequence[Dict[str, Any]],
    ) -> List[Relation]:
        edges: List[Relation] = []
        predicate = tokens[predicate_index]

        if self._verb_form(predicate) == "Fin":
            for index, token in enumerate(tokens):
                if index == predicate_index or abs(index - predicate_index) > 15:
                    continue
                if not self._is_event_complement_candidate(token):
                    continue
                edges.append(
                    Relation(
                        src=predicate_index,
                        dst=index,
                        label="COMPLEMENT",
                        confidence=0.5,
                        rule="predicate_complement_local_event",
                    )
                )

        for index, token in enumerate(tokens):
            if index == predicate_index or abs(index - predicate_index) > 3:
                continue
            if token.get("semantic_type") not in {"entity", "indexical"}:
                continue
            if self._is_function_word(token):
                continue
            if index <= 0 or tokens[index - 1].get("pos") != "ADP":
                continue
            edges.append(
                Relation(
                    src=predicate_index,
                    dst=index,
                    label="COMPLEMENT",
                    confidence=0.48,
                    rule="predicate_complement_local_adp_oblique",
                )
            )

        return edges

    def _modifier_attachment(
        self,
        modifier_index: int,
        tokens: Sequence[Dict[str, Any]],
    ) -> Optional[Relation]:
        modifier = tokens[modifier_index]
        if modifier.get("semantic_type") == "manner":
            target = self._find_event(tokens, modifier_index, direction=-1, exclude=set())
            if target is None:
                target = self._find_event(tokens, modifier_index, direction=+1, exclude=set())
        else:
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
        if self.enhance_baseline_relations and tokens[connector_index].get("pos") != "CCONJ":
            return None
        left = self._find_content(tokens, connector_index, direction=-1)
        right = self._find_content(tokens, connector_index, direction=+1)
        if left is None or right is None:
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

    def _enhance_baseline_relations(
        self,
        relations: Sequence[Relation],
        tokens: Sequence[Dict[str, Any]],
    ) -> List[Relation]:
        enhanced: List[Relation] = []
        for relation in relations:
            if (
                relation.label == "AGENT"
                and self._is_content_event(tokens[relation.src])
                and self._verb_form(tokens[relation.src]) == "Fin"
                and self._is_bad_baseline_agent_target(tokens, relation.dst)
            ):
                continue
            if (
                relation.label == "THEME"
                and self._is_content_event(tokens[relation.src])
                and self._is_quantity_like_nominal(tokens, relation.dst)
            ):
                continue
            if relation.label == "MODIFIER" and tokens[relation.dst].get("semantic_type") == "manner":
                target = self._forward_manner_target(tokens, relation.dst)
                if target is not None and relation.src != target:
                    continue
            if (
                relation.label == "MODIFIER"
                and tokens[relation.dst].get("semantic_type") == "property"
                and self._is_nominal_token(tokens[relation.src])
            ):
                src_case = self._case_value(tokens[relation.src])
                dst_case = self._case_value(tokens[relation.dst])
                if src_case and dst_case and src_case != dst_case:
                    continue
            enhanced.append(relation)

        existing_agents = {relation.src for relation in enhanced if relation.label == "AGENT"}
        for predicate_index, token in enumerate(tokens):
            if (
                self._is_content_event(token)
                and self._verb_form(token) == "Fin"
                and not self._is_passive_like(token)
                and predicate_index not in existing_agents
            ):
                agent_index = self._best_baseline_agent(tokens, predicate_index)
                if agent_index is not None:
                    self._add_unique_relation(
                        enhanced,
                        predicate_index,
                        agent_index,
                        "AGENT",
                        0.62,
                        "baseline_agent_nominal_quality",
                    )
            if (
                self._is_content_event(token)
                and self._verb_form(token) == "Part"
                and self._case_value(token) == "Gen"
            ):
                agent_index = self._nearest_case_value(
                    tokens,
                    predicate_index,
                    {"Gen"},
                    max_distance=3,
                )
                if agent_index is not None:
                    self._add_unique_relation(
                        enhanced,
                        predicate_index,
                        agent_index,
                        "AGENT",
                        0.58,
                        "baseline_agent_genitive_participle",
                    )
            if (
                self._is_content_event(token)
                and self._verb_form(token) == "Part"
                and self._is_passive_like(token)
            ):
                for index in range(predicate_index + 1, min(len(tokens), predicate_index + 5)):
                    if (
                        self._case_value(tokens[index]) == "Gen"
                        and self._is_nominal_token(tokens[index])
                        and self._is_prepositional_object(tokens, index)
                    ):
                        self._add_unique_relation(
                            enhanced,
                            predicate_index,
                            index,
                            "AGENT",
                            0.58,
                            "baseline_agent_passive_oblique",
                        )
                        break

        existing_themes = {relation.src for relation in enhanced if relation.label == "THEME"}
        for predicate_index, token in enumerate(tokens):
            if not self._is_content_event(token):
                continue
            if self._verb_form(token) != "Inf":
                theme_index = self._nearest_case_value(
                    tokens,
                    predicate_index,
                    {"Dat"},
                    allow_property=True,
                    max_distance=5,
                    skip_pronouns=True,
                )
                if theme_index is not None:
                    self._add_unique_relation(
                        enhanced,
                        predicate_index,
                        theme_index,
                        "THEME",
                        0.54,
                        "baseline_theme_dative_argument",
                    )
            if predicate_index not in existing_themes and self._allows_genitive_theme(tokens, predicate_index):
                theme_index = self._nearest_case_value(
                    tokens,
                    predicate_index,
                    {"Gen"},
                    max_distance=5,
                    skip_pronouns=True,
                )
                if theme_index is not None:
                    self._add_unique_relation(
                        enhanced,
                        predicate_index,
                        theme_index,
                        "THEME",
                        0.53,
                        "baseline_theme_genitive_argument_general",
                    )

        if any(token.get("pos") == "AUX" and self._verb_form(token) == "Fin" for token in tokens):
            for predicate_index, token in enumerate(tokens):
                if (
                    token.get("pos") in {"NOUN", "ADJ"}
                    and self._case_value(token) == "Nom"
                    and token.get("semantic_type") in {"entity", "property"}
                ):
                    candidates: List[int] = []
                    for index, candidate in enumerate(tokens):
                        if index == predicate_index or abs(index - predicate_index) > 5:
                            continue
                        if (
                            self._case_value(candidate) == "Nom"
                            and self._is_nominal_token(candidate)
                            and candidate.get("pos") != "DET"
                        ):
                            candidates.append(index)
                    if candidates:
                        theme_index = min(candidates, key=lambda index: abs(index - predicate_index))
                        self._add_unique_relation(
                            enhanced,
                            predicate_index,
                            theme_index,
                            "THEME",
                            0.52,
                            "baseline_theme_nominal_predicate",
                        )
                    for index, candidate in enumerate(tokens):
                        if (
                            abs(index - predicate_index) <= 6
                            and self._is_nominal_token(candidate)
                            and self._is_prepositional_object(tokens, index)
                        ):
                            self._add_unique_relation(
                                enhanced,
                                predicate_index,
                                index,
                                "COMPLEMENT",
                                0.5,
                                "baseline_complement_nominal_oblique",
                            )

        for predicate_index, token in enumerate(tokens):
            if not self._is_content_event(token):
                continue
            if self._verb_form(token) in {"Inf", "Part"}:
                complement_index = self._nearest_case_value(
                    tokens,
                    predicate_index,
                    {"Dat"},
                    max_distance=4,
                    skip_prepositional=True,
                )
                if complement_index is not None:
                    self._add_unique_relation(
                        enhanced,
                        predicate_index,
                        complement_index,
                        "COMPLEMENT",
                        0.5,
                        "baseline_complement_nonfinite_dative",
                    )
            for index, candidate in enumerate(tokens):
                if (
                    index != predicate_index
                    and self._is_nominal_token(candidate)
                    and self._is_prepositional_object(tokens, index)
                ):
                    max_distance = 8 if self._verb_form(token) in {"Part", "Inf"} else 4
                    if 0 < index - predicate_index <= max_distance:
                        self._add_unique_relation(
                            enhanced,
                            predicate_index,
                            index,
                            "COMPLEMENT",
                            0.49,
                            "baseline_complement_adp_object",
                        )
            if self._verb_form(token) == "Inf":
                for index, candidate in enumerate(tokens):
                    if (
                        index > predicate_index
                        and index - predicate_index <= 12
                        and self._is_content_event(candidate)
                        and self._mood(candidate) in {"Sub", "Opt"}
                    ):
                        self._add_unique_relation(
                            enhanced,
                            predicate_index,
                            index,
                            "COMPLEMENT",
                            0.5,
                            "baseline_complement_inf_subordinate_general",
                        )
                        break
            for index, candidate in enumerate(tokens):
                if (
                    abs(index - predicate_index) <= 6
                    and self._is_quantity_like_nominal(tokens, index)
                ):
                    self._add_unique_relation(
                        enhanced,
                        predicate_index,
                        index,
                        "COMPLEMENT",
                        0.49,
                        "baseline_complement_quantity",
                    )
            if self._verb_form(token) == "Part":
                for index in range(predicate_index + 1, min(len(tokens), predicate_index + 5)):
                    if self._is_content_event(tokens[index]) and self._verb_form(tokens[index]) == "Inf":
                        self._add_unique_relation(
                            enhanced,
                            predicate_index,
                            index,
                            "COMPLEMENT",
                            0.51,
                            "baseline_complement_participle_inf",
                        )
                        break
            if self._verb_form(token) == "Fin":
                for index, candidate in enumerate(tokens):
                    if (
                        index != predicate_index
                        and abs(index - predicate_index) <= 4
                        and self._is_content_event(candidate)
                        and self._verb_form(candidate) == "Fin"
                        and self._is_passive_like(candidate)
                    ):
                        self._add_unique_relation(
                            enhanced,
                            predicate_index,
                            index,
                            "COMPLEMENT",
                            0.5,
                            "baseline_complement_passive_clause",
                        )

        for index, token in enumerate(tokens):
            if token.get("semantic_type") == "manner" and not self._is_function_word(token):
                target = self._forward_manner_target(tokens, index)
                if target is not None:
                    self._add_unique_relation(
                        enhanced,
                        target,
                        index,
                        "MODIFIER",
                        0.58,
                        "baseline_modifier_forward_manner",
                    )
        for index, token in enumerate(tokens[:2]):
            if token.get("semantic_type") == "manner" and not self._is_function_word(token):
                for predicate_index in range(index + 1, min(len(tokens), index + 9)):
                    if (
                        self._is_content_event(tokens[predicate_index])
                        and self._verb_form(tokens[predicate_index]) == "Fin"
                    ):
                        self._add_unique_relation(
                            enhanced,
                            predicate_index,
                            index,
                            "COMPLEMENT",
                            0.47,
                            "baseline_complement_initial_adverb",
                        )
                        break

        for index, token in enumerate(tokens):
            if token.get("pos") == "NUM":
                if index > 0 and tokens[index - 1].get("pos") == "CCONJ":
                    continue
                head_index = None
                for candidate_index in range(index - 1, max(-1, index - 5), -1):
                    if self._is_quantity_like_nominal(tokens, candidate_index):
                        head_index = candidate_index
                        break
                if head_index is None:
                    for candidate_index in range(index - 1, max(-1, index - 4), -1):
                        if (
                            self._is_nominal_token(tokens[candidate_index])
                            and tokens[candidate_index].get("pos") not in {"DET", "NUM"}
                        ):
                            head_index = candidate_index
                            break
                if head_index is not None:
                    self._add_unique_relation(
                        enhanced,
                        head_index,
                        index,
                        "MODIFIER",
                        0.55,
                        "baseline_modifier_number",
                    )
            if (
                self._case_value(token) == "Gen"
                and self._is_nominal_token(token)
                and token.get("pos") != "DET"
            ):
                head_index = None
                for candidate_index in range(index + 1, min(len(tokens), index + 4)):
                    if (
                        tokens[candidate_index].get("pos") in {"NOUN", "PROPN"}
                        and self._is_nominal_token(tokens[candidate_index])
                        and self._case_value(tokens[candidate_index]) != "Gen"
                    ):
                        head_index = candidate_index
                        break
                if head_index is None:
                    for candidate_index in range(index - 1, max(-1, index - 4), -1):
                        if (
                            tokens[candidate_index].get("pos") in {"NOUN", "PROPN", "ADJ"}
                            and self._is_nominal_token(tokens[candidate_index], allow_property=True)
                            and self._case_value(tokens[candidate_index]) != "Gen"
                        ):
                            head_index = candidate_index
                            break
                if head_index is not None:
                    self._add_unique_relation(
                        enhanced,
                        head_index,
                        index,
                        "MODIFIER",
                        0.52,
                        "baseline_modifier_genitive",
                    )
            if self._is_nominal_token(token, allow_property=True) and token.get("semantic_type") == "property":
                if self._is_prepositional_object(tokens, index) or (
                    index > 0 and tokens[index - 1].get("pos") in {"CCONJ", "DET", "ADP"}
                ):
                    for candidate_index in range(index + 1, min(len(tokens), index + 6)):
                        if (
                            tokens[candidate_index].get("pos") == "NOUN"
                            and self._is_nominal_token(tokens[candidate_index])
                            and self._case_value(tokens[candidate_index]) == self._case_value(token)
                        ):
                            self._add_unique_relation(
                                enhanced,
                                candidate_index,
                                index,
                                "MODIFIER",
                                0.5,
                                "baseline_modifier_adp_property",
                            )
                            break
                if (
                    index > 0
                    and self._is_nominal_token(tokens[index - 1])
                    and tokens[index - 1].get("pos") != "NUM"
                    and self._same_case_gender_number(token, tokens[index - 1])
                ):
                    self._add_unique_relation(
                        enhanced,
                        index - 1,
                        index,
                        "MODIFIER",
                        0.52,
                        "baseline_modifier_postposed_property",
                    )
            if token.get("pos") in {"NOUN", "ADJ"} and self._case_value(token) and index > 1:
                if tokens[index - 1].get("pos") == "DET":
                    for candidate_index in range(index - 2, max(-1, index - 5), -1):
                        if (
                            tokens[candidate_index].get("pos") == "PROPN"
                            and self._case_value(tokens[candidate_index]) == self._case_value(token)
                        ):
                            self._add_unique_relation(
                                enhanced,
                                candidate_index,
                                index,
                                "MODIFIER",
                                0.51,
                                "baseline_modifier_appositive_article",
                            )
                            break

        return enhanced

    def _add_unique_relation(
        self,
        relations: List[Relation],
        src: int,
        dst: int,
        label: str,
        confidence: float,
        rule: str,
    ) -> None:
        if src == dst:
            return
        if any(relation.src == src and relation.dst == dst and relation.label == label for relation in relations):
            return
        relations.append(Relation(src=src, dst=dst, label=label, confidence=confidence, rule=rule))

    def _case_value(self, token: Dict[str, Any]) -> str:
        features = token.get("features") or {}
        ud_features = features.get("ud_features") or {}
        return str(ud_features.get("Case") or features.get("case") or "")

    def _gender_value(self, token: Dict[str, Any]) -> str:
        features = token.get("features") or {}
        ud_features = features.get("ud_features") or {}
        return str(ud_features.get("Gender") or features.get("gender") or "")

    def _number_value(self, token: Dict[str, Any]) -> str:
        features = token.get("features") or {}
        ud_features = features.get("ud_features") or {}
        return str(ud_features.get("Number") or features.get("number") or "")

    def _lemma(self, token: Dict[str, Any]) -> str:
        return str(token.get("lemma") or "")

    def _is_content_event(self, token: Dict[str, Any]) -> bool:
        return (
            token.get("semantic_type") == "event"
            and not self._is_function_word(token)
            and not self._is_auxiliary_predicate(token)
        )

    def _is_nominal_token(self, token: Dict[str, Any], allow_property: bool = False) -> bool:
        semantic_types = {"entity", "indexical"}
        if allow_property:
            semantic_types.add("property")
        return token.get("semantic_type") in semantic_types and not self._is_function_word(token)

    def _has_previous_article(self, tokens: Sequence[Dict[str, Any]], index: int) -> bool:
        return (
            index > 0
            and tokens[index - 1].get("pos") == "DET"
            and self._case_value(tokens[index - 1]) == self._case_value(tokens[index])
        )

    def _is_prepositional_object(self, tokens: Sequence[Dict[str, Any]], index: int) -> bool:
        return (
            index > 0
            and tokens[index - 1].get("pos") == "ADP"
        ) or (
            index > 1
            and tokens[index - 2].get("pos") == "ADP"
            and tokens[index - 1].get("pos") == "DET"
        )

    def _is_quantity_like_nominal(self, tokens: Sequence[Dict[str, Any]], index: int) -> bool:
        token = tokens[index]
        if not self._is_nominal_token(token) or token.get("pos") == "PRON":
            return False
        if token.get("pos") == "NUM":
            return True
        if token.get("pos") not in {"NOUN", "PROPN"}:
            return False
        for candidate_index in range(max(0, index - 3), min(len(tokens), index + 4)):
            if candidate_index != index and tokens[candidate_index].get("pos") == "NUM":
                return True
        return False

    def _allows_genitive_theme(
        self,
        tokens: Sequence[Dict[str, Any]],
        predicate_index: int,
    ) -> bool:
        predicate = tokens[predicate_index]
        if self._verb_form(predicate) == "Inf":
            return False
        if self._is_passive_like(predicate):
            return False
        for index, token in enumerate(tokens):
            if index == predicate_index or abs(index - predicate_index) > 5:
                continue
            if self._is_prepositional_object(tokens, index):
                continue
            if not self._is_nominal_token(token, allow_property=True):
                continue
            if self._case_value(token) in {"Acc", "Dat"} and token.get("pos") != "DET":
                return False
        return True

    def _is_bad_baseline_agent_target(self, tokens: Sequence[Dict[str, Any]], index: int) -> bool:
        token = tokens[index]
        if token.get("pos") == "DET":
            return True
        if token.get("pos") == "PRON" and "Neut" in self._gender_value(token):
            return True
        if self._has_previous_article(tokens, index):
            return True
        if index > 0:
            for event_index in range(max(0, index - 2), min(len(tokens), index + 3)):
                if (
                    event_index != index
                    and self._is_content_event(tokens[event_index])
                    and abs(event_index - index) <= 1
                ):
                    return True
        return False

    def _baseline_agent_score(
        self,
        tokens: Sequence[Dict[str, Any]],
        predicate_index: int,
        candidate_index: int,
    ) -> Optional[int]:
        candidate = tokens[candidate_index]
        if self._case_value(candidate) != "Nom":
            return None
        if not self._is_nominal_token(candidate, allow_property=True):
            return None
        if candidate.get("pos") == "DET":
            return None
        if candidate.get("pos") == "PRON" and "Neut" in self._gender_value(candidate):
            return None
        if self._has_previous_article(tokens, candidate_index):
            return None
        if candidate_index > 0 and candidate.get("semantic_type") != "property":
            for event_index in range(max(0, candidate_index - 2), min(len(tokens), candidate_index + 3)):
                if (
                    event_index not in {candidate_index, predicate_index}
                    and self._is_content_event(tokens[event_index])
                    and abs(event_index - candidate_index) <= 1
                ):
                    return None
        distance = abs(candidate_index - predicate_index)
        if distance > 15:
            return None
        score = distance
        if candidate_index == 0:
            score -= 3
        if candidate.get("semantic_type") == "property":
            score -= 1
        if candidate.get("pos") == "PRON":
            score += 2
        return score

    def _best_baseline_agent(
        self,
        tokens: Sequence[Dict[str, Any]],
        predicate_index: int,
    ) -> Optional[int]:
        candidates: List[tuple[int, int]] = []
        for index in range(len(tokens)):
            if index == predicate_index:
                continue
            score = self._baseline_agent_score(tokens, predicate_index, index)
            if score is not None:
                candidates.append((score, index))
        if not candidates:
            return None
        return min(candidates)[1]

    def _nearest_case_value(
        self,
        tokens: Sequence[Dict[str, Any]],
        origin: int,
        cases: set[str],
        allow_property: bool = False,
        max_distance: int = 5,
        skip_pronouns: bool = False,
        skip_prepositional: bool = True,
    ) -> Optional[int]:
        candidates: List[tuple[int, int]] = []
        for index, token in enumerate(tokens):
            if index == origin or abs(index - origin) > max_distance:
                continue
            if self._case_value(token) not in cases:
                continue
            if not self._is_nominal_token(token, allow_property=allow_property):
                continue
            if token.get("pos") == "DET":
                continue
            if skip_pronouns and token.get("pos") == "PRON":
                continue
            if skip_prepositional and self._is_prepositional_object(tokens, index):
                continue
            candidates.append((abs(index - origin), index))
        if not candidates:
            return None
        return min(candidates)[1]

    def _same_case_gender_number(self, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        if self._case_value(left) != self._case_value(right):
            return False
        left_gender = self._gender_value(left)
        right_gender = self._gender_value(right)
        if left_gender and right_gender and left_gender != right_gender:
            return False
        left_number = self._number_value(left)
        right_number = self._number_value(right)
        if left_number and right_number and left_number != right_number:
            return False
        return True

    def _forward_manner_target(
        self,
        tokens: Sequence[Dict[str, Any]],
        modifier_index: int,
    ) -> Optional[int]:
        if (
            modifier_index + 2 < len(tokens)
            and self._is_content_event(tokens[modifier_index + 1])
            and self._is_content_event(tokens[modifier_index + 2])
            and self._verb_form(tokens[modifier_index + 2]) == "Inf"
        ):
            return modifier_index + 2
        if modifier_index + 1 < len(tokens) and self._is_content_event(tokens[modifier_index + 1]):
            return modifier_index + 1
        return None

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
            if self._is_function_word(token):
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
                if tok.get("semantic_type") in {"entity", "indexical"} and not self._is_function_word(tok):
                    return index
            index += direction
        return None

    def _find_event(
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
                if tok.get("semantic_type") == "event" and not self._is_function_word(tok):
                    return index
            index += direction
        return None

    def _find_content(
        self,
        tokens: Sequence[Dict[str, Any]],
        origin: int,
        direction: int,
    ) -> Optional[int]:
        index = origin + direction
        while 0 <= index < len(tokens):
            tok = tokens[index]
            if tok.get("semantic_type") not in {"unknown", "operator", "relation", "connector"}:
                if not self._is_function_word(tok):
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

    def _is_function_word(self, token: Dict[str, Any]) -> bool:
        features = token.get("features") or {}
        if features.get("function_word") is True:
            return True
        if self.filter_discourse_particles and self._is_discourse_particle(token):
            return True
        return token.get("pos") in {"DET", "ADP", "CCONJ", "SCONJ", "PART"}

    def _is_event_complement_candidate(self, token: Dict[str, Any]) -> bool:
        if token.get("semantic_type") != "event":
            return False
        if self._is_function_word(token) or self._is_auxiliary_predicate(token):
            return False
        return (
            self._verb_form(token) in {"Inf", "Part"}
            or self._mood(token) in {"Sub", "Opt"}
        )

    def _is_auxiliary_predicate(self, token: Dict[str, Any]) -> bool:
        return token.get("pos") == "AUX"

    def _is_discourse_particle(self, token: Dict[str, Any]) -> bool:
        key = surface_key(str(token.get("lemma") or token.get("token") or ""))
        return key in self.DISCOURSE_PARTICLE_KEYS

    def _is_passive_like(self, token: Dict[str, Any]) -> bool:
        features = token.get("features") or {}
        ud_features = features.get("ud_features") or {}
        voice = str(ud_features.get("Voice") or features.get("voice") or "").lower()
        return "pass" in voice

    def _verb_form(self, token: Dict[str, Any]) -> str:
        features = token.get("features") or {}
        ud_features = features.get("ud_features") or {}
        return str(ud_features.get("VerbForm") or features.get("verb_form") or "")

    def _mood(self, token: Dict[str, Any]) -> str:
        features = token.get("features") or {}
        ud_features = features.get("ud_features") or {}
        return str(ud_features.get("Mood") or features.get("mood") or "")

    def _has_case_hints(self, tokens: Sequence[Dict[str, Any]]) -> bool:
        return any(self._case_hint(token) is not None for token in tokens)

    def _allow_short_clause_theme_fallback(
        self,
        tokens: Sequence[Dict[str, Any]],
        used: set,
    ) -> bool:
        if len(tokens) > 4 or not used:
            return False
        if any(self._case_hint(token) == "acc" for token in tokens):
            return False
        entity_count = sum(
            1
            for token in tokens
            if token.get("semantic_type") in {"entity", "indexical"}
            and not self._is_function_word(token)
        )
        return entity_count <= 2


__all__ = ["Relation", "RelationPredictor"]
