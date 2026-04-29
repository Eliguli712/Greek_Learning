"""
Build this project's semantic DAGs from real Ancient Greek UD CoNLL-U data.

This bypasses the fragile raw-text analyzer and treats Universal Dependencies
as the upstream linguistic annotation source. The output tests the interesting
project question: can real Greek annotations be represented as the repo's
semantic DAG format?

Usage:
    python scripts/run_ud_dag.py --conllu data/real_eval/grc_perseus-ud-test.conllu
    python scripts/run_ud_dag.py --conllu data/real_eval/*.conllu --max-sentences 100
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.ud_adapter import UD_TO_DAG_LABEL, dag_from_ud, iter_conllu


FUNCTION_MARKER_DEPRELS = {
    "aux",
    "aux:pass",
    "case",
    "cc",
    "cop",
    "discourse",
    "fixed",
    "flat:name",
    "mark",
}

PREDICATE_ARGUMENT_DEPRELS = {
    "csubj",
    "csubj:pass",
    "iobj",
    "nsubj",
    "nsubj:outer",
    "nsubj:pass",
    "obj",
    "obl:agent",
    "obl:arg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conllu",
        type=Path,
        nargs="+",
        required=True,
        help="One or more Ancient Greek UD CoNLL-U files.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=200,
        help="Maximum number of sentences to convert.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "ud_dag_results.jsonl",
        help="JSONL output path for converted DAGs.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional JSON summary path. Defaults to <output>.summary.json.",
    )
    return parser.parse_args()


def build_ud_dags(
    conllu_paths: Sequence[Path],
    max_sentences: int,
    output_path: Path,
    summary_path: Path,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    label_counts: Counter[str] = Counter()
    deprel_counts: Counter[str] = Counter()
    unmapped_deprel_counts: Counter[str] = Counter()
    invalid: List[Dict[str, Any]] = []

    sentence_count = 0
    token_count = 0
    edge_count = 0
    mapped_dependency_count = 0
    unmapped_dependency_count = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for sentence in iter_conllu(conllu_paths, max_sentences=max_sentences):
            result = dag_from_ud(sentence)
            payload = result.to_dict()
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

            sentence_count += 1
            token_count += len(sentence.tokens)
            edge_count += len(result.dag.edges)
            mapped_dependency_count += len(result.relations)
            unmapped_dependency_count += len(result.unmapped_dependencies)

            for relation in result.relations:
                label_counts[relation.label] += 1
                deprel_counts[relation.rule.replace("ud::", "", 1)] += 1
            for dependency in result.unmapped_dependencies:
                unmapped_deprel_counts[str(dependency.get("deprel", ""))] += 1

            if not result.dag.validation.get("ok", False):
                invalid.append(
                    {
                        "id": sentence.sent_id,
                        "validation": result.dag.validation,
                    }
                )

    total_considered_dependencies = mapped_dependency_count + unmapped_dependency_count
    content_mapped = sum(deprel_counts.values())
    content_unmapped = sum(
        count
        for deprel, count in unmapped_deprel_counts.items()
        if deprel not in FUNCTION_MARKER_DEPRELS
    )
    predicate_argument_mapped = sum(
        count
        for deprel, count in deprel_counts.items()
        if deprel in PREDICATE_ARGUMENT_DEPRELS
    )
    predicate_argument_unmapped = sum(
        count
        for deprel, count in unmapped_deprel_counts.items()
        if deprel in PREDICATE_ARGUMENT_DEPRELS
    )
    summary = {
        "inputs": [str(path) for path in conllu_paths],
        "output": str(output_path),
        "sentences": sentence_count,
        "tokens": token_count,
        "edges": edge_count,
        "node_preservation_rate": 1.0 if token_count else 0.0,
        "lemma_pos_feature_retention_rate": 1.0 if token_count else 0.0,
        "mapped_dependencies": mapped_dependency_count,
        "unmapped_dependencies": unmapped_dependency_count,
        "mapped_dependency_coverage": _safe_div(
            mapped_dependency_count,
            total_considered_dependencies,
        ),
        "overall_dependency_prf": _conversion_prf(
            mapped_dependency_count,
            total_considered_dependencies,
        ),
        "content_dependency_coverage": _safe_div(
            content_mapped,
            content_mapped + content_unmapped,
        ),
        "content_dependency_prf": _conversion_prf(
            content_mapped,
            content_mapped + content_unmapped,
        ),
        "predicate_argument_coverage": _safe_div(
            predicate_argument_mapped,
            predicate_argument_mapped + predicate_argument_unmapped,
        ),
        "predicate_argument_prf": _conversion_prf(
            predicate_argument_mapped,
            predicate_argument_mapped + predicate_argument_unmapped,
        ),
        "dag_validity_rate": _safe_div(sentence_count - len(invalid), sentence_count),
        "label_counts": dict(sorted(label_counts.items())),
        "mapped_ud_deprel_counts": dict(sorted(deprel_counts.items())),
        "unmapped_ud_deprel_counts": dict(unmapped_deprel_counts.most_common()),
        "invalid_dags": invalid,
        "mapping": dict(sorted(UD_TO_DAG_LABEL.items())),
        "notes": [
            "This uses UD gold annotations as upstream input; it is not raw-text parsing.",
            "Coverage measures the share of non-root, non-punctuation UD dependencies mapped into this repo's DAG label inventory.",
            "Content dependency coverage excludes UD function-marker dependencies such as case, cc, mark, cop, discourse, aux, fixed, and flat:name.",
            "Predicate-argument coverage measures subject/object/argument-style UD dependencies mapped into AGENT/THEME-style DAG edges.",
            "Node preservation and lemma/POS/feature retention are 1.0 by construction because the adapter copies UD annotations into DAG nodes.",
            "DAG validity measures structural graph validity after conversion, not linguistic prediction accuracy.",
        ],
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def _safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _conversion_prf(mapped: int, target: int) -> Dict[str, float | int]:
    precision = 1.0 if mapped else 0.0
    recall = mapped / target if target else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": target,
    }


def main() -> None:
    args = parse_args()
    summary_path = args.summary or Path(str(args.output) + ".summary.json")
    summary = build_ud_dags(
        conllu_paths=args.conllu,
        max_sentences=args.max_sentences,
        output_path=args.output,
        summary_path=summary_path,
    )

    print("=" * 72)
    print("UD -> semantic DAG conversion")
    print("=" * 72)
    print(f"  sentences converted         : {summary['sentences']}")
    print(f"  tokens                      : {summary['tokens']}")
    print(f"  DAG edges                   : {summary['edges']}")
    print(f"  mapped dependency coverage  : {summary['mapped_dependency_coverage']}")
    print(f"  content dependency coverage : {summary['content_dependency_coverage']}")
    print(f"  predicate-argument coverage : {summary['predicate_argument_coverage']}")
    print(f"  DAG validity rate           : {summary['dag_validity_rate']}")
    print(f"  output                      : {args.output}")
    print(f"  summary                     : {summary_path}")


if __name__ == "__main__":
    main()
