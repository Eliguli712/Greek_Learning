"""Simple UD-backed baseline for semantic DAG construction.

This baseline uses the same UD tokens, lemmas, POS tags, and morphology as the
main compiler, but it intentionally ignores UD dependency edges. Instead, it
predicts semantic relations using only local word-order, POS, and case
heuristics via the existing RelationPredictor. The baseline is therefore a
useful comparison point for measuring how much the dependency-to-semantic
mapping contributes beyond a shallow linear heuristic.
"""

from __future__ import annotations

from src.dag import DAGBuilder
from src.syntax_logic import RelationPredictor
from src.ud_adapter import UDDAGResult, UDSentence, semantic_tokens_from_ud


def dag_from_ud_baseline(sentence: UDSentence) -> UDDAGResult:
    """Build a semantic DAG baseline from one UD-annotated sentence."""
    semantic_tokens = semantic_tokens_from_ud(sentence, relations=[])
    relations = RelationPredictor(
        filter_discourse_particles=True,
        infer_baseline_complements=True,
        suppress_nonfinite_agents=True,
        enhance_baseline_relations=True,
    ).predict(semantic_tokens)
    dag = DAGBuilder().build(semantic_tokens, relations)
    return UDDAGResult(
        sentence=sentence,
        semantic_tokens=semantic_tokens,
        relations=relations,
        dag=dag,
        unmapped_dependencies=[],
    )


__all__ = ["dag_from_ud_baseline"]
