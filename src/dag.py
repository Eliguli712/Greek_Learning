"""
DAG construction over semantic tokens and predicted relations.

The output is a JSON-serializable graph plus a structural validation report.
We use NetworkX (a standard NLP/graph academic library) for cycle detection
and graph manipulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import networkx as nx

from src.syntax_logic import Relation
from src.validate import validate_graph, ALLOWED_EDGE_LABELS


@dataclass
class SemanticDAG:
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "validation": self.validation,
        }

    def to_networkx(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for node in self.nodes:
            graph.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
        for edge in self.edges:
            graph.add_edge(edge["src"], edge["dst"], **{k: v for k, v in edge.items() if k not in {"src", "dst"}})
        return graph


class DAGBuilder:
    """Compose a typed semantic DAG and validate it."""

    def build(
        self,
        semantic_tokens: Sequence[Dict[str, Any]],
        relations: Sequence[Relation],
    ) -> SemanticDAG:
        nodes = [self._node_from_token(i, tok) for i, tok in enumerate(semantic_tokens)]

        edges: List[Dict[str, Any]] = []
        seen = set()
        for rel in relations:
            if not (0 <= rel.src < len(nodes) and 0 <= rel.dst < len(nodes)):
                continue
            if rel.src == rel.dst:
                continue
            key = (nodes[rel.src]["id"], nodes[rel.dst]["id"], rel.label)
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                {
                    "src": nodes[rel.src]["id"],
                    "dst": nodes[rel.dst]["id"],
                    "label": rel.label,
                    "confidence": round(rel.confidence, 3),
                    "rule": rel.rule,
                }
            )

        edges = self._break_cycles(nodes, edges)
        validation = validate_graph(nodes, edges, allowed_labels=ALLOWED_EDGE_LABELS)
        return SemanticDAG(nodes=nodes, edges=edges, validation=validation)

    def _node_from_token(self, index: int, token: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": f"n{index}",
            "index": index,
            "token": token.get("token"),
            "lemma": token.get("lemma"),
            "pos": token.get("pos"),
            "semantic_type": token.get("semantic_type"),
            "semantic_role": token.get("semantic_role"),
            "semantic_class": token.get("semantic_class"),
            "transliteration": token.get("transliteration"),
            "features": token.get("features", {}),
            "confidence": token.get("confidence"),
        }

    def _break_cycles(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        graph = nx.DiGraph()
        for node in nodes:
            graph.add_node(node["id"])
        for index, edge in enumerate(edges):
            graph.add_edge(edge["src"], edge["dst"], _eid=index)

        kept = list(edges)
        while True:
            try:
                cycle = nx.find_cycle(graph, orientation="original")
            except nx.NetworkXNoCycle:
                break
            # Drop the lowest-confidence edge in the cycle.
            cycle_edges = []
            for u, v, _direction in cycle:
                eid = graph[u][v]["_eid"]
                cycle_edges.append((eid, kept[eid]["confidence"]))
            drop_eid = min(cycle_edges, key=lambda item: item[1])[0]
            dropped = kept[drop_eid]
            graph.remove_edge(dropped["src"], dropped["dst"])
            kept[drop_eid] = None  # mark as dropped

        return [edge for edge in kept if edge is not None]


__all__ = ["SemanticDAG", "DAGBuilder"]
