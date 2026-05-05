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
from time import perf_counter
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
        "--progress-interval",
        type=int,
        default=50,
        help="Print evaluation progress every N sentences.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "relation_classifier_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_started = perf_counter()
    train_dir = PROJECT_ROOT / "data" / "ud_treebanks"
    _log(
        f"starting train/dev/test run; max_sentences={args.max_sentences}, "
        f"progress_interval={args.progress_interval}"
    )

    train_paths = sorted(train_dir.glob("*-train.conllu"))
    if not train_paths:
        _log("no UD train files found; downloading official UD train/dev/test splits")
        download_ud_treebanks()
        train_paths = sorted(train_dir.glob("*-train.conllu"))
    else:
        _log(f"found {len(train_paths)} UD train files: {_path_names(train_paths)}")

    classifier = relation_classifier_from_project(PROJECT_ROOT, progress=_log)
    if classifier is None:
        raise RuntimeError("No UD train files found under data/ud_treebanks.")

    dev_paths = sorted(train_dir.glob("*-dev.conllu"))
    test_paths = sorted(train_dir.glob("*-test.conllu"))
    _log(f"dev files: {_path_names(dev_paths)}")
    _log(f"test files: {_path_names(test_paths)}")
    thresholds = [0.15, 0.25, 0.35, 0.45, 0.55]

    dev_runs: List[Dict[str, Any]] = []
    for threshold_index, threshold in enumerate(thresholds, start=1):
        step_started = perf_counter()
        _log(
            f"dev threshold {threshold_index}/{len(thresholds)} = {threshold}: "
            f"evaluating up to {args.max_sentences} sentences"
        )
        classifier.threshold = threshold
        report = evaluate_ud(
            dev_paths,
            max_sentences=args.max_sentences,
            use_relation_classifier=True,
            progress=_eval_progress(
                label=f"dev threshold {threshold}",
                started=step_started,
            ),
            progress_interval=args.progress_interval,
        )
        relation = report["metrics"]["mapped_relation_prf"]
        _print_score("dev", threshold, report["metrics"])
        _log(
            f"dev threshold {threshold} done in {_elapsed(step_started)}: "
            f"P/R/F1={relation['precision']} / {relation['recall']} / {relation['f1']}"
        )
        dev_runs.append(
            {
                "threshold": threshold,
                "mapped_relation_prf": report["metrics"]["mapped_relation_prf"],
            }
        )

    best = max(dev_runs, key=lambda item: item["mapped_relation_prf"]["f1"])
    classifier.threshold = float(best["threshold"])
    _log(f"selected dev threshold={best['threshold']}")
    test_started = perf_counter()
    _log(f"test evaluation: evaluating up to {args.max_sentences} sentences")
    test_report = evaluate_ud(
        test_paths,
        max_sentences=args.max_sentences,
        use_relation_classifier=True,
        progress=_eval_progress(
            label="test",
            started=test_started,
        ),
        progress_interval=args.progress_interval,
    )
    test_relation = test_report["metrics"]["mapped_relation_prf"]
    _print_score("test", best["threshold"], test_report["metrics"])
    _log(
        f"test evaluation done in {_elapsed(test_started)}: "
        f"P/R/F1={test_relation['precision']} / {test_relation['recall']} / {test_relation['f1']}"
    )

    output = {
        "train_files": [
            str(path)
            for path in sorted(train_dir.glob("*-train.conllu"))
        ],
        "dev_files": [str(path) for path in dev_paths],
        "test_files": [str(path) for path in test_paths],
        "max_sentences": args.max_sentences,
        "dev_threshold_sweep": dev_runs,
        "selected_threshold": best["threshold"],
        "test_metrics": test_report["metrics"],
    }

    _log(f"writing report: {args.output}")
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
    _log(f"finished total runtime {_elapsed(run_started)}")


def _log(message: str) -> None:
    print(f"[relation-classifier] {message}", flush=True)


def _print_score(split: str, threshold: float, metrics: Dict[str, Any]) -> None:
    relation = metrics["mapped_relation_prf"]
    print(
        "SCORE "
        f"split={split} "
        f"threshold={threshold} "
        f"relation_f1={relation['f1']} "
        f"relation_precision={relation['precision']} "
        f"relation_recall={relation['recall']} "
        f"lemma_acc={metrics['lemma_accuracy_normalized']} "
        f"upos_acc={metrics['upos_accuracy']} "
        f"semantic_acc={metrics['semantic_type_proxy_accuracy']} "
        f"aligned={metrics['sentences_aligned']}/{metrics['sentences_seen']}",
        flush=True,
    )


def _path_names(paths: List[Path]) -> str:
    if not paths:
        return "(none)"
    return ", ".join(path.name for path in paths)


def _eval_progress(label: str, started: float):
    def progress(status: Dict[str, Any]) -> None:
        current_file = Path(str(status["current_file"])).name
        seen = status["sentences_seen"]
        max_sentences = status["max_sentences"]
        aligned = status["sentences_aligned"]
        skipped = status["sentences_skipped_alignment"]
        _log(
            f"{label}: {seen}/{max_sentences} sentences seen, "
            f"aligned={aligned}, skipped={skipped}, file={current_file}, "
            f"elapsed={_elapsed(started)}"
        )

    return progress


def _elapsed(started: float) -> str:
    return f"{perf_counter() - started:.1f}s"


if __name__ == "__main__":
    main()
