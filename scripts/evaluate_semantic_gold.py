"""
Evaluate predicted semantic DAG edges against a small manual gold set.

The gold file uses this project's own graph labels, so this measures whether
the produced semantic graph matches a manually specified target graph rather
than merely whether UD dependencies were converted.

Usage:
    python scripts/evaluate_semantic_gold.py
    python scripts/evaluate_semantic_gold.py --gold data/gold_semantic/simple_semantic_dags.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.metrics import accuracy, set_prf
from src.normalize import surface_key
from src.pipeline import SemanticCompiler


DEFAULT_GOLD = PROJECT_ROOT / "data" / "gold_semantic" / "simple_semantic_dags.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "semantic_gold_eval.json",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(gold_path: Path, output_path: Path) -> Dict[str, Any]:
    compiler = SemanticCompiler(project_root=PROJECT_ROOT)
    examples = load_jsonl(gold_path)

    pred_lemma_all: List[str] = []
    gold_lemma_all: List[str] = []
    pred_type_all: List[str] = []
    gold_type_all: List[str] = []
    pred_rel_all: List[Tuple[str, str, str, str]] = []
    gold_rel_all: List[Tuple[str, str, str, str]] = []
    per_example: List[Dict[str, Any]] = []

    for example in examples:
        result = compiler.analyze(example["text"], example.get("language", "ancient_greek"))

        pred_lemmas = [
            surface_key(item.get("lemma", ""))
            for item in result.lexeme_layer
            if "error" not in item
        ]
        gold_lemmas = [surface_key(item) for item in example.get("lemmas", [])]
        pred_types = [
            item.get("semantic_type", "")
            for item in result.semantic_tokens
            if "error" not in item
        ]
        gold_types = list(example.get("semantic_types", []))

        pred_lemma_all.extend(pred_lemmas[: len(gold_lemmas)])
        gold_lemma_all.extend(gold_lemmas[: len(pred_lemmas)])
        pred_type_all.extend(pred_types[: len(gold_types)])
        gold_type_all.extend(gold_types[: len(pred_types)])

        pred_rel = _relations_from_edges(example["id"], result.relations)
        gold_rel = _relations_from_gold(example["id"], example.get("relations", []))
        pred_rel_all.extend(pred_rel)
        gold_rel_all.extend(gold_rel)

        per_example.append(
            {
                "id": example["id"],
                "text": example["text"],
                "predicted_tokens": result.tokens,
                "gold_tokens": example.get("tokens", []),
                "predicted_lemmas": pred_lemmas,
                "gold_lemmas": gold_lemmas,
                "predicted_semantic_types": pred_types,
                "gold_semantic_types": gold_types,
                "predicted_relations": [list(item[1:]) for item in pred_rel],
                "gold_relations": [list(item[1:]) for item in gold_rel],
                "relation_prf": set_prf(pred_rel, gold_rel).as_dict(),
                "validation": result.validation,
            }
        )

    report = {
        "gold": str(gold_path),
        "metrics": {
            "examples": len(examples),
            "tokens": len(gold_lemma_all),
            "gold_relations": len(gold_rel_all),
            "predicted_relations": len(pred_rel_all),
            "lemma_accuracy": round(accuracy(pred_lemma_all, gold_lemma_all), 4),
            "semantic_type_accuracy": round(accuracy(pred_type_all, gold_type_all), 4),
            "semantic_relation_prf": set_prf(pred_rel_all, gold_rel_all).as_dict(),
            "dag_validity_rate": round(
                sum(1 for item in per_example if item["validation"].get("ok")) / len(per_example),
                4,
            )
            if per_example
            else 0.0,
        },
        "per_example": per_example,
        "notes": [
            "This is a manual gold semantic-DAG evaluation using this project's own labels.",
            "The set is intentionally small and transparent; it is a correctness check, not a broad corpus benchmark.",
            "Relations are evaluated as exact token-index edge matches: source index, target index, label.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    _print_summary(report, output_path)
    return report


def _relations_from_edges(example_id: str, edges: Iterable[Dict[str, Any]]) -> List[Tuple[str, str, str, str]]:
    relations: List[Tuple[str, str, str, str]] = []
    for edge in edges:
        src = str(edge.get("src", ""))
        dst = str(edge.get("dst", ""))
        label = str(edge.get("label", ""))
        if src.startswith("n"):
            src = src[1:]
        if dst.startswith("n"):
            dst = dst[1:]
        relations.append((example_id, src, dst, label))
    return relations


def _relations_from_gold(example_id: str, relations: Sequence[Dict[str, Any]]) -> List[Tuple[str, str, str, str]]:
    return [
        (
            example_id,
            str(item.get("src", "")),
            str(item.get("dst", "")),
            str(item.get("label", "")),
        )
        for item in relations
    ]


def _print_summary(report: Dict[str, Any], output_path: Path) -> None:
    metrics = report["metrics"]
    relation = metrics["semantic_relation_prf"]
    print("=" * 72)
    print("Manual semantic DAG evaluation")
    print("=" * 72)
    print(f"  examples                   : {metrics['examples']}")
    print(f"  tokens                     : {metrics['tokens']}")
    print(f"  lemma accuracy             : {metrics['lemma_accuracy']}")
    print(f"  semantic type accuracy     : {metrics['semantic_type_accuracy']}")
    print(
        "  relation P/R/F1            : "
        f"{relation['precision']} / {relation['recall']} / {relation['f1']}"
    )
    print(f"  DAG validity rate          : {metrics['dag_validity_rate']}")
    print(f"\nFull report written to: {output_path}")


def main() -> None:
    args = parse_args()
    evaluate(args.gold, args.output)


if __name__ == "__main__":
    main()
