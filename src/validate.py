"""Validation rules for semantic DAGs produced by the pipeline."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set

import networkx as nx


ALLOWED_EDGE_LABELS: Set[str] = {
    "AGENT",
    "THEME",
    "EXPERIENCER",
    "MODIFIER",
    "COMPLEMENT",
    "COORD",
    "SCOPE",
}


def validate_graph(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    allowed_labels: Iterable[str] = ALLOWED_EDGE_LABELS,
) -> Dict[str, Any]:
    """
    Run structural validation on a semantic DAG.

    Checks:
      - acyclicity (must be a DAG)
      - edge labels are in the allowed inventory
      - every predicate that requires arguments has at least one outgoing
        AGENT, THEME, or EXPERIENCER edge
      - every edge endpoint references a known node id
    """
    allowed = set(allowed_labels)
    report: Dict[str, Any] = {
        "ok": True,
        "acyclic": True,
        "illegal_edges": [],
        "dangling_edges": [],
        "missing_required_args": [],
        "node_count": len(nodes),
        "edge_count": len(edges),
    }

    node_index = {node["id"]: node for node in nodes}

    graph = nx.DiGraph()
    for node in nodes:
        graph.add_node(node["id"])

    for edge in edges:
        src = edge.get("src")
        dst = edge.get("dst")
        label = edge.get("label", "")

        if src not in node_index or dst not in node_index:
            report["dangling_edges"].append(edge)
            report["ok"] = False
            continue

        if label not in allowed:
            report["illegal_edges"].append(edge)
            report["ok"] = False
            continue

        graph.add_edge(src, dst, label=label)

    if not nx.is_directed_acyclic_graph(graph):
        report["acyclic"] = False
        report["ok"] = False
        try:
            report["cycle"] = list(nx.find_cycle(graph, orientation="original"))
        except nx.NetworkXNoCycle:
            report["cycle"] = []

    for node in nodes:
        features = node.get("features") or {}
        if features.get("requires_arguments"):
            outgoing = [
                edge for edge in edges
                if edge.get("src") == node["id"]
                and edge.get("label") in {"AGENT", "THEME", "EXPERIENCER"}
            ]
            if not outgoing:
                report["missing_required_args"].append(node["id"])
                report["ok"] = False

    return report


__all__ = ["validate_graph", "ALLOWED_EDGE_LABELS"]
