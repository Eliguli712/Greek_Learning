"""
Run the SemanticCompiler on a gold JSONL set and report per-stage metrics
plus an error breakdown for development analysis.

Usage (PowerShell):
    & .venv/Scripts/python.exe scripts/evaluate.py --split dev
    & .venv/Scripts/python.exe scripts/evaluate.py --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.metrics import accuracy, dag_validity_rate, labelled_prf, set_prf
from src.pipeline import SemanticCompiler


SPLIT_PATHS = {
    "dev": PROJECT_ROOT / "data/gold/dev_examples.jsonl",
    "test": PROJECT_ROOT / "data/gold/test_examples.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLIT_PATHS.keys()), default="dev")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON evaluation report.",
    )
    return parser.parse_args()


def load_gold(path: Path) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def evaluate(split: str, output_path: Path | None = None) -> Dict[str, Any]:
    path = SPLIT_PATHS[split]
    if not path.exists():
        raise FileNotFoundError(f"Gold file missing: {path}")

    examples = load_gold(path)
    compiler = SemanticCompiler(project_root=PROJECT_ROOT)

    pred_lemmas_all: List[str] = []
    gold_lemmas_all: List[str] = []
    pred_types_all: List[str] = []
    gold_types_all: List[str] = []
    relation_pred: List[Tuple[str, str, str]] = []
    relation_gold: List[Tuple[str, str, str]] = []

    per_example: List[Dict[str, Any]] = []
    error_buckets: Dict[str, List[Dict[str, Any]]] = {
        "lemma_errors": [],
        "semantic_type_errors": [],
        "missing_relations": [],
        "spurious_relations": [],
        "invalid_dag": [],
    }
    validations: List[Dict[str, Any]] = []

    for example in examples:
        result = compiler.analyze(example["text"], example.get("language", "ancient_greek"))

        pred_tokens = result.tokens
        pred_lemmas = [tok.get("lemma", "") for tok in result.lexeme_layer if "error" not in tok]
        pred_types = [tok.get("semantic_type", "") for tok in result.semantic_tokens if "error" not in tok]

        gold_tokens = example.get("tokens", [])
        gold_lemmas = example.get("lemmas", [])
        gold_types = example.get("semantic_types", [])

        pred_lemmas_all.extend(pred_lemmas[: len(gold_lemmas)])
        gold_lemmas_all.extend(gold_lemmas[: len(pred_lemmas)])
        pred_types_all.extend(pred_types[: len(gold_types)])
        gold_types_all.extend(gold_types[: len(pred_types)])

        for i in range(min(len(pred_lemmas), len(gold_lemmas))):
            if pred_lemmas[i] != gold_lemmas[i]:
                error_buckets["lemma_errors"].append(
                    {
                        "id": example["id"],
                        "token": gold_tokens[i] if i < len(gold_tokens) else None,
                        "predicted": pred_lemmas[i],
                        "gold": gold_lemmas[i],
                    }
                )
        for i in range(min(len(pred_types), len(gold_types))):
            if pred_types[i] != gold_types[i]:
                error_buckets["semantic_type_errors"].append(
                    {
                        "id": example["id"],
                        "token": gold_tokens[i] if i < len(gold_tokens) else None,
                        "predicted": pred_types[i],
                        "gold": gold_types[i],
                    }
                )

        # Relations are matched on (src_lemma, dst_lemma, label) tuples so we
        # do not depend on node id ordering.
        gold_lemma_for_index = {i: lemma for i, lemma in enumerate(gold_lemmas)}
        pred_lemma_for_index = {i: lemma for i, lemma in enumerate(pred_lemmas)}

        gold_rel_set: List[Tuple[str, str, str]] = []
        for rel in example.get("relations", []):
            gold_rel_set.append(
                (
                    gold_lemma_for_index.get(rel["src"], "?"),
                    gold_lemma_for_index.get(rel["dst"], "?"),
                    rel["label"],
                )
            )

        pred_rel_set: List[Tuple[str, str, str]] = []
        for edge in result.relations:
            src_idx = _node_index(edge["src"])
            dst_idx = _node_index(edge["dst"])
            pred_rel_set.append(
                (
                    pred_lemma_for_index.get(src_idx, "?"),
                    pred_lemma_for_index.get(dst_idx, "?"),
                    edge["label"],
                )
            )

        relation_pred.extend(pred_rel_set)
        relation_gold.extend(gold_rel_set)

        missing = set(gold_rel_set) - set(pred_rel_set)
        spurious = set(pred_rel_set) - set(gold_rel_set)
        for rel in missing:
            error_buckets["missing_relations"].append({"id": example["id"], "relation": list(rel)})
        for rel in spurious:
            error_buckets["spurious_relations"].append({"id": example["id"], "relation": list(rel)})

        if not result.validation.get("ok", False):
            error_buckets["invalid_dag"].append(
                {
                    "id": example["id"],
                    "validation": result.validation,
                }
            )

        validations.append(result.validation)
        per_example.append(
            {
                "id": example["id"],
                "predicted_tokens": pred_tokens,
                "predicted_lemmas": pred_lemmas,
                "predicted_semantic_types": pred_types,
                "predicted_relations": [list(r) for r in pred_rel_set],
                "gold_relations": [list(r) for r in gold_rel_set],
                "validation": result.validation,
            }
        )

    metrics = {
        "split": split,
        "n_examples": len(examples),
        "lemma_accuracy": round(accuracy(pred_lemmas_all, gold_lemmas_all), 4),
        "semantic_type_macro_f1": labelled_prf(pred_types_all, gold_types_all, average="macro").as_dict(),
        "semantic_type_micro_f1": labelled_prf(pred_types_all, gold_types_all, average="micro").as_dict(),
        "relation_prf": set_prf(relation_pred, relation_gold).as_dict(),
        "dag_validity_rate": round(dag_validity_rate(validations), 4),
    }

    report = {
        "metrics": metrics,
        "error_analysis": error_buckets,
        "per_example": per_example,
    }

    target = output_path or (PROJECT_ROOT / "outputs" / f"eval_{split}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    _print_summary(metrics, error_buckets, target)
    return report


def _node_index(node_id: str) -> int:
    if isinstance(node_id, str) and node_id.startswith("n"):
        try:
            return int(node_id[1:])
        except ValueError:
            return -1
    return -1


def _print_summary(metrics: Dict[str, Any], errors: Dict[str, List[Dict[str, Any]]], path: Path) -> None:
    print("=" * 60)
    print(f"Evaluation report ({metrics['split']}, n={metrics['n_examples']})")
    print("=" * 60)
    print(f"  lemma accuracy             : {metrics['lemma_accuracy']}")
    print(f"  semantic type macro F1     : {metrics['semantic_type_macro_f1']['f1']}")
    print(f"  semantic type micro F1     : {metrics['semantic_type_micro_f1']['f1']}")
    print(
        "  relation P / R / F1        : "
        f"{metrics['relation_prf']['precision']} / "
        f"{metrics['relation_prf']['recall']} / "
        f"{metrics['relation_prf']['f1']}"
    )
    print(f"  DAG validity rate          : {metrics['dag_validity_rate']}")
    print("-" * 60)
    print("Error counts:")
    for bucket, items in errors.items():
        print(f"  {bucket:<24}: {len(items)}")
    print(f"\nFull report written to: {path}")


def main() -> None:
    args = parse_args()
    evaluate(args.split, args.output)


if __name__ == "__main__":
    main()
