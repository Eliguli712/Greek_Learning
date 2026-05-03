"""
Evaluate UD-backed semantic DAGs against manually labeled historical sentences.

This is the project-level correctness check: the input sentences come from
public Ancient Greek UD test files, but the gold target is this repo's semantic
DAG edge inventory. The gold annotations intentionally focus on content
semantic edges, so copied function words and articles are not automatically
rewarded.

Usage:
    python scripts/evaluate_historical_gold.py
    python scripts/evaluate_historical_gold.py --system baseline
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.metrics import set_prf
from src.ud_adapter import UDSentence, dag_from_ud, read_conllu
from src.ud_baseline import dag_from_ud_baseline


DEFAULT_GOLD = PROJECT_ROOT / "data" / "gold_semantic" / "historical_semantic_dags.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "historical_semantic_gold_eval.json"
BASELINE_OUTPUT = PROJECT_ROOT / "outputs" / "historical_semantic_gold_baseline_eval.json"
DEFAULT_LABELS = ("AGENT", "THEME", "MODIFIER", "COMPLEMENT", "COORD")


RelationTuple = Tuple[str, str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--system",
        choices=("compiler", "baseline"),
        default="compiler",
        help="Which system to evaluate: the full UD compiler or the simple local baseline.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_LABELS),
        help="DAG labels to evaluate. Defaults to the project semantic edge labels.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    return rows


def evaluate(
    gold_path: Path,
    output_path: Path,
    labels: Sequence[str],
    system: str = "compiler",
) -> Dict[str, Any]:
    examples = load_jsonl(gold_path)
    sentence_cache: Dict[Path, Mapping[str, UDSentence]] = {}
    label_set = set(labels)
    system_runner = dag_from_ud if system == "compiler" else dag_from_ud_baseline

    pred_rel_all: List[RelationTuple] = []
    gold_rel_all: List[RelationTuple] = []
    per_example: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    for example in examples:
        conllu_path = (PROJECT_ROOT / example["conllu"]).resolve()
        if conllu_path not in sentence_cache:
            sentence_cache[conllu_path] = {
                sentence.sent_id: sentence for sentence in read_conllu(conllu_path)
            }

        sentence = sentence_cache[conllu_path].get(example["sent_id"])
        if sentence is None:
            raise KeyError(f"Could not find sent_id={example['sent_id']} in {conllu_path}")

        actual_tokens = [token.form for token in sentence.tokens]
        expected_tokens = list(example.get("tokens", []))
        if expected_tokens and expected_tokens != actual_tokens:
            warnings.append(
                {
                    "id": example["id"],
                    "type": "token_mismatch",
                    "expected": expected_tokens,
                    "actual": actual_tokens,
                }
            )

        result = system_runner(sentence)
        pred_rel = [
            item
            for item in _relations_from_dag_edges(example["id"], result.dag.edges)
            if item[3] in label_set
        ]
        gold_rel = [
            item
            for item in _relations_from_gold(example["id"], example.get("relations", []))
            if item[3] in label_set
        ]

        pred_rel_all.extend(pred_rel)
        gold_rel_all.extend(gold_rel)

        per_example.append(
            {
                "id": example["id"],
                "source": example.get("source"),
                "sent_id": sentence.sent_id,
                "text": sentence.text,
                "tokens": actual_tokens,
                "gold_relations": [list(item[1:]) for item in gold_rel],
                "predicted_relations": [list(item[1:]) for item in pred_rel],
                "relation_prf": set_prf(pred_rel, gold_rel).as_dict(),
                "validation": result.dag.validation,
            }
        )

    relation_prf = set_prf(pred_rel_all, gold_rel_all).as_dict()
    report = {
        "gold": str(gold_path),
        "system": system,
        "labels": list(labels),
        "metrics": {
            "examples": len(examples),
            "tokens": sum(len(item["tokens"]) for item in per_example),
            "gold_relations": len(gold_rel_all),
            "predicted_relations": len(pred_rel_all),
            "semantic_relation_prf": relation_prf,
            "by_label": _by_label_prf(pred_rel_all, gold_rel_all, labels),
            "dag_validity_rate": round(
                sum(1 for item in per_example if item["validation"].get("ok")) / len(per_example),
                4,
            )
            if per_example
            else 0.0,
        },
        "per_example": per_example,
        "warnings": warnings,
        "notes": [
            "The gold file contains manually selected semantic edges over real historical Greek sentences.",
            "Gold edges are content-semantic: articles, discourse particles, and purely functional markers are not gold edges.",
            "Precision is meaningful here because predicted extra DAG edges count as false positives.",
            "Recall is meaningful here because missing or differently labeled manual semantic edges count as false negatives.",
            "Scores measure UD-backed DAG compilation, not raw Greek parsing from plain text.",
            "The baseline system ignores UD dependency links and predicts semantic edges from local token order, POS, and case heuristics only.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    _print_summary(report, output_path)
    return report


def _relations_from_dag_edges(example_id: str, edges: Iterable[Dict[str, Any]]) -> List[RelationTuple]:
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


def _relations_from_gold(example_id: str, relations: Sequence[Dict[str, Any]]) -> List[RelationTuple]:
    return [
        (
            example_id,
            str(item.get("src", "")),
            str(item.get("dst", "")),
            str(item.get("label", "")),
        )
        for item in relations
    ]


def _by_label_prf(
    predicted: Sequence[RelationTuple],
    gold: Sequence[RelationTuple],
    labels: Sequence[str],
) -> Dict[str, Dict[str, float | int]]:
    by_pred: Dict[str, List[RelationTuple]] = defaultdict(list)
    by_gold: Dict[str, List[RelationTuple]] = defaultdict(list)
    for item in predicted:
        by_pred[item[3]].append(item)
    for item in gold:
        by_gold[item[3]].append(item)

    return {
        label: set_prf(by_pred[label], by_gold[label]).as_dict()
        for label in labels
    }


def _print_summary(report: Dict[str, Any], output_path: Path) -> None:
    metrics = report["metrics"]
    relation = metrics["semantic_relation_prf"]
    print("=" * 72)
    print(f"Historical manual semantic DAG evaluation ({report['system']})")
    print("=" * 72)
    print(f"  examples                   : {metrics['examples']}")
    print(f"  tokens                     : {metrics['tokens']}")
    print(f"  gold relations             : {metrics['gold_relations']}")
    print(f"  predicted relations        : {metrics['predicted_relations']}")
    print(
        "  relation P/R/F1            : "
        f"{relation['precision']} / {relation['recall']} / {relation['f1']}"
    )
    print(f"  DAG validity rate          : {metrics['dag_validity_rate']}")
    print("  by label:")
    for label, scores in metrics["by_label"].items():
        print(
            f"    {label:<10} P/R/F1: "
            f"{scores['precision']} / {scores['recall']} / {scores['f1']} "
            f"(gold={scores['support']})"
        )
    if report.get("warnings"):
        print(f"  warnings                   : {len(report['warnings'])}")
    print(f"\nFull report written to: {output_path}")


def main() -> None:
    args = parse_args()
    output_path = args.output
    if output_path is None:
        output_path = DEFAULT_OUTPUT if args.system == "compiler" else BASELINE_OUTPUT
    evaluate(args.gold, output_path, args.labels, system=args.system)


if __name__ == "__main__":
    main()
