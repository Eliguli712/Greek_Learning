"""Train and evaluate the raw-pipeline relation classifier on UD splits.

The model is trained lazily from data/ud_treebanks/*-train.conllu by
src.relation_classifier. This script tunes the prediction threshold on dev
sentences and reports strict held-out test performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.evaluate_ud import evaluate_ud
from scripts.download_ud_treebanks import main as download_ud_treebanks
from src.relation_classifier import relation_classifier_from_project


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-sentences", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "relation_classifier_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not list((PROJECT_ROOT / "data" / "ud_treebanks").glob("*-train.conllu")):
        download_ud_treebanks()

    classifier = relation_classifier_from_project(PROJECT_ROOT)
    if classifier is None:
        raise RuntimeError("No UD train files found under data/ud_treebanks.")

    dev_paths = sorted((PROJECT_ROOT / "data" / "ud_treebanks").glob("*-dev.conllu"))
    test_paths = sorted((PROJECT_ROOT / "data" / "ud_treebanks").glob("*-test.conllu"))
    thresholds = [0.15, 0.25, 0.35, 0.45, 0.55]

    dev_runs: List[Dict[str, Any]] = []
    for threshold in thresholds:
        classifier.threshold = threshold
        report = evaluate_ud(
            dev_paths,
            max_sentences=args.max_sentences,
            use_relation_classifier=True,
        )
        dev_runs.append(
            {
                "threshold": threshold,
                "mapped_relation_prf": report["metrics"]["mapped_relation_prf"],
            }
        )

    best = max(dev_runs, key=lambda item: item["mapped_relation_prf"]["f1"])
    classifier.threshold = float(best["threshold"])
    test_report = evaluate_ud(
        test_paths,
        max_sentences=args.max_sentences,
        use_relation_classifier=True,
    )

    output = {
        "train_files": [
            str(path)
            for path in sorted((PROJECT_ROOT / "data" / "ud_treebanks").glob("*-train.conllu"))
        ],
        "dev_files": [str(path) for path in dev_paths],
        "test_files": [str(path) for path in test_paths],
        "max_sentences": args.max_sentences,
        "dev_threshold_sweep": dev_runs,
        "selected_threshold": best["threshold"],
        "test_metrics": test_report["metrics"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    relation = test_report["metrics"]["mapped_relation_prf"]
    print("=" * 72)
    print("Relation classifier train/dev/test report")
    print("=" * 72)
    print(f"  selected threshold          : {best['threshold']}")
    print(
        "  test mapped relation P/R/F1 : "
        f"{relation['precision']} / {relation['recall']} / {relation['f1']}"
    )
    print(f"  full report                 : {args.output}")


if __name__ == "__main__":
    main()
