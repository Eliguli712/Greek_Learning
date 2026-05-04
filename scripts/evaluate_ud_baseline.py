"""
Evaluate UD-backed baseline/compiler DAGs against UD-derived proxy relations.

This is a large-scale sanity check, not the manual historical semantic gold
benchmark. Gold relations are created by mapping UD dependency labels into this
project's semantic labels. The baseline system receives UD tokens, lemmas, POS,
and morphology, but it does not receive UD head/deprel edges.

Usage:
    python scripts/evaluate_ud_baseline.py --conllu data/real_eval/grc_perseus-ud-test.conllu
    python scripts/evaluate_ud_baseline.py --system compiler --conllu data/real_eval/*.conllu
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.metrics import set_prf  # noqa: E402
from src.ud_adapter import dag_from_ud, read_conllu, relations_from_ud  # noqa: E402
from src.ud_baseline import dag_from_ud_baseline  # noqa: E402


DEFAULT_LABELS = ("AGENT", "THEME", "MODIFIER", "COMPLEMENT", "COORD")
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "ud_baseline_proxy_eval.json"
RelationTuple = Tuple[str, str, str, str]


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
        "--system",
        choices=("baseline", "compiler"),
        default="baseline",
        help="baseline ignores UD dependency edges; compiler maps UD dependency edges.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=500,
        help="Maximum number of sentences to evaluate across all input files.",
    )
    parser.add_argument(
        "--skip-sentences",
        type=int,
        default=0,
        help="Skip this many sentences across the concatenated inputs before evaluating.",
    )
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def evaluate(
    conllu_paths: Sequence[Path],
    system: str,
    labels: Sequence[str],
    max_sentences: int,
    skip_sentences: int,
    output_path: Path,
) -> Dict[str, Any]:
    label_set = set(labels)
    runner = dag_from_ud_baseline if system == "baseline" else dag_from_ud

    predicted: List[RelationTuple] = []
    gold: List[RelationTuple] = []
    by_file_pred: Dict[str, List[RelationTuple]] = defaultdict(list)
    by_file_gold: Dict[str, List[RelationTuple]] = defaultdict(list)

    seen = 0
    evaluated = 0
    token_count = 0

    for conllu_path in conllu_paths:
        resolved_path = (PROJECT_ROOT / conllu_path).resolve() if not conllu_path.is_absolute() else conllu_path
        for sentence in read_conllu(resolved_path):
            if seen < skip_sentences:
                seen += 1
                continue
            if evaluated >= max_sentences:
                break

            example_id = f"{resolved_path.name}:{sentence.sent_id}"
            result = runner(sentence)
            pred_rel = [
                relation
                for relation in _relations_from_edges(example_id, result.dag.edges)
                if relation[3] in label_set
            ]
            gold_rel = [
                relation
                for relation in _relations_from_ud(example_id, sentence)
                if relation[3] in label_set
            ]

            predicted.extend(pred_rel)
            gold.extend(gold_rel)
            by_file_pred[resolved_path.name].extend(pred_rel)
            by_file_gold[resolved_path.name].extend(gold_rel)

            seen += 1
            evaluated += 1
            token_count += len(sentence.tokens)

        if evaluated >= max_sentences:
            break

    report = {
        "inputs": [str(path) for path in conllu_paths],
        "system": system,
        "labels": list(labels),
        "max_sentences": max_sentences,
        "skip_sentences": skip_sentences,
        "metrics": {
            "sentences": evaluated,
            "tokens": token_count,
            "gold_relations": len(gold),
            "predicted_relations": len(predicted),
            "semantic_relation_prf": set_prf(predicted, gold).as_dict(),
            "by_label": _by_label_prf(predicted, gold, labels),
            "by_file": {
                filename: {
                    "gold_relations": len(by_file_gold[filename]),
                    "predicted_relations": len(by_file_pred[filename]),
                    "semantic_relation_prf": set_prf(
                        by_file_pred[filename],
                        by_file_gold[filename],
                    ).as_dict(),
                }
                for filename in sorted(by_file_gold)
            },
        },
        "notes": [
            "Gold relations here are UD-dependency-derived proxy labels, not manual semantic gold.",
            "The baseline system uses token, lemma, POS, morphology, and linear order, but no UD head/deprel edges.",
            "The compiler system uses UD dependency edges and is therefore an upper-bound sanity check for this proxy task.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    _print_summary(report, output_path)
    return report


def _relations_from_edges(example_id: str, edges: Iterable[Dict[str, Any]]) -> List[RelationTuple]:
    relations: List[RelationTuple] = []
    for edge in edges:
        src = str(edge.get("src", ""))
        dst = str(edge.get("dst", ""))
        if src.startswith("n"):
            src = src[1:]
        if dst.startswith("n"):
            dst = dst[1:]
        relations.append((example_id, src, dst, str(edge.get("label", ""))))
    return relations


def _relations_from_ud(example_id: str, sentence: Any) -> List[RelationTuple]:
    relations, _unmapped = relations_from_ud(sentence)
    return [
        (example_id, str(relation.src), str(relation.dst), relation.label)
        for relation in relations
    ]


def _by_label_prf(
    predicted: Sequence[RelationTuple],
    gold: Sequence[RelationTuple],
    labels: Sequence[str],
) -> Dict[str, Dict[str, float | int]]:
    return {
        label: set_prf(
            [relation for relation in predicted if relation[3] == label],
            [relation for relation in gold if relation[3] == label],
        ).as_dict()
        for label in labels
    }


def _print_summary(report: Dict[str, Any], output_path: Path) -> None:
    metrics = report["metrics"]
    relation = metrics["semantic_relation_prf"]
    print("=" * 72)
    print(f"UD proxy semantic DAG evaluation ({report['system']})")
    print("=" * 72)
    print(f"  sentences                  : {metrics['sentences']}")
    print(f"  tokens                     : {metrics['tokens']}")
    print(f"  gold relations             : {metrics['gold_relations']}")
    print(f"  predicted relations        : {metrics['predicted_relations']}")
    print(
        "  relation P/R/F1            : "
        f"{relation['precision']} / {relation['recall']} / {relation['f1']}"
    )
    print("  by label:")
    for label, scores in metrics["by_label"].items():
        print(
            f"    {label:<10} P/R/F1: "
            f"{scores['precision']} / {scores['recall']} / {scores['f1']} "
            f"(gold={scores['support']})"
        )
    print(f"\nFull report written to: {output_path}")


def main() -> None:
    args = parse_args()
    evaluate(
        conllu_paths=args.conllu,
        system=args.system,
        labels=args.labels,
        max_sentences=args.max_sentences,
        skip_sentences=args.skip_sentences,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
