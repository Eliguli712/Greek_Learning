"""
Evaluate historical semantic DAGs with corpus-aware train/test separation.

This script is meant for the manual historical semantic gold file. It keeps the
compiler rule-based, but it learns which paper-aligned function-like modifier
filters to enable from the training fold only, then reports strict edge
precision/recall/F1 on held-out test examples.

Usage:
    python scripts/evaluate_historical_splits.py
    python scripts/evaluate_historical_splits.py --mode within-corpus --test-ratio 0.35
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
from typing import AbstractSet, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.ud_adapter import (  # noqa: E402
    ARTICLE_LEMMAS,
    DISCOURSE_PARTICLE_LEMMAS,
    UDSentence,
    dag_from_ud,
    read_conllu,
)


DEFAULT_GOLD = PROJECT_ROOT / "data" / "gold_semantic" / "historical_semantic_dags.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "historical_semantic_split_eval.json"
DEFAULT_SPLIT_DIR = PROJECT_ROOT / "outputs" / "historical_semantic_splits"
DEFAULT_LABELS = ("AGENT", "THEME", "MODIFIER", "COMPLEMENT", "COORD")

RelationTuple = Tuple[str, str, str, str]
ModifierFilter = Dict[str, AbstractSet[str]]


FILTER_CANDIDATES: Dict[str, Tuple[str, AbstractSet[str]]] = {
    "det_article": ("det", ARTICLE_LEMMAS),
    "advmod_discourse_particle": ("advmod", DISCOURSE_PARTICLE_LEMMAS),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument(
        "--mode",
        choices=("leave-one-corpus-out", "within-corpus"),
        default="leave-one-corpus-out",
        help="Split protocol. Leave-one-corpus-out trains on all other corpora and tests on one held-out corpus.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.35,
        help="Held-out share per corpus for --mode within-corpus.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--test-corpus",
        action="append",
        default=None,
        help="Limit leave-one-corpus-out to one or more corpus names, e.g. grc_perseus.",
    )
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument(
        "--no-split-files",
        action="store_true",
        help="Do not write train/test JSONL files for each fold.",
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


def evaluate_split(
    gold_path: Path,
    output_path: Path,
    split_dir: Path,
    labels: Sequence[str],
    mode: str,
    test_ratio: float,
    seed: int,
    test_corpora: Optional[Sequence[str]],
    write_split_files: bool,
) -> Dict[str, Any]:
    examples = load_jsonl(gold_path)
    folds = _make_folds(examples, mode=mode, test_ratio=test_ratio, seed=seed, test_corpora=test_corpora)
    label_set = set(labels)
    sentence_cache: Dict[Path, Mapping[str, UDSentence]] = {}

    fold_reports: List[Dict[str, Any]] = []
    for fold in folds:
        selected_filter, filter_stats = _train_modifier_filter(
            fold["train_examples"],
            label_set,
            sentence_cache,
        )
        train_report = _evaluate_examples(
            fold["train_examples"],
            label_set,
            sentence_cache,
            selected_filter,
        )
        test_report = _evaluate_examples(
            fold["test_examples"],
            label_set,
            sentence_cache,
            selected_filter,
        )
        test_by_corpus = {
            corpus: _evaluate_examples(rows, label_set, sentence_cache, selected_filter)["metrics"]
            for corpus, rows in sorted(_group_by_corpus(fold["test_examples"]).items())
        }

        fold_report = {
            "name": fold["name"],
            "train_corpora": fold["train_corpora"],
            "test_corpora": fold["test_corpora"],
            "train_examples": [example["id"] for example in fold["train_examples"]],
            "test_examples": [example["id"] for example in fold["test_examples"]],
            "selected_filters": _filter_names(selected_filter),
            "filter_training_stats": filter_stats,
            "train_metrics": train_report["metrics"],
            "test_metrics": test_report["metrics"],
            "test_by_corpus": test_by_corpus,
        }
        fold_reports.append(fold_report)

        if write_split_files:
            _write_fold_files(split_dir, fold)

    report = {
        "gold": str(gold_path),
        "mode": mode,
        "labels": list(labels),
        "seed": seed,
        "test_ratio": test_ratio if mode == "within-corpus" else None,
        "corpora": {
            corpus: [example["id"] for example in rows]
            for corpus, rows in sorted(_group_by_corpus(examples).items())
        },
        "filter_policy": (
            "Candidate function-like modifier filters are selected from the training fold "
            "only when they remove false positives and remove no true positives on train."
        ),
        "folds": fold_reports,
        "macro_test_f1": _macro_metric(fold_reports, "f1"),
        "macro_test_precision": _macro_metric(fold_reports, "precision"),
        "macro_test_recall": _macro_metric(fold_reports, "recall"),
        "notes": [
            "The manual gold examples are split before filter selection.",
            "No test example is used when selecting function-like modifier filters.",
            "Leave-one-corpus-out is the stricter default because the held-out corpus is unseen during filter selection.",
            "Scores are exact edge precision/recall/F1 over source token, destination token, and semantic label.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    _print_summary(report, output_path, split_dir if write_split_files else None)
    return report


def _make_folds(
    examples: Sequence[Dict[str, Any]],
    mode: str,
    test_ratio: float,
    seed: int,
    test_corpora: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    by_corpus = _group_by_corpus(examples)
    if len(by_corpus) < 2:
        raise ValueError("At least two corpora are required for corpus-aware splitting.")

    if mode == "leave-one-corpus-out":
        requested = set(test_corpora or by_corpus.keys())
        folds: List[Dict[str, Any]] = []
        for corpus in sorted(by_corpus):
            if corpus not in requested:
                continue
            train = [
                example
                for other, rows in by_corpus.items()
                if other != corpus
                for example in rows
            ]
            test = list(by_corpus[corpus])
            if not train or not test:
                continue
            folds.append(
                {
                    "name": f"heldout_{corpus}",
                    "train_corpora": sorted(c for c in by_corpus if c != corpus),
                    "test_corpora": [corpus],
                    "train_examples": sorted(train, key=lambda item: item["id"]),
                    "test_examples": sorted(test, key=lambda item: item["id"]),
                }
            )
        if not folds:
            raise ValueError(f"No folds created for requested test corpora: {sorted(requested)}")
        return folds

    if not (0.0 < test_ratio < 1.0):
        raise ValueError("--test-ratio must be between 0 and 1 for within-corpus splitting.")

    rng = random.Random(seed)
    train_examples: List[Dict[str, Any]] = []
    test_examples: List[Dict[str, Any]] = []
    for corpus, rows in sorted(by_corpus.items()):
        shuffled = sorted(rows, key=lambda item: item["id"])
        rng.shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * test_ratio))
        if test_count >= len(shuffled):
            test_count = len(shuffled) - 1
        test_examples.extend(shuffled[:test_count])
        train_examples.extend(shuffled[test_count:])

    return [
        {
            "name": f"within_corpus_seed_{seed}",
            "train_corpora": sorted(by_corpus),
            "test_corpora": sorted(by_corpus),
            "train_examples": sorted(train_examples, key=lambda item: item["id"]),
            "test_examples": sorted(test_examples, key=lambda item: item["id"]),
        }
    ]


def _group_by_corpus(examples: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for example in examples:
        grouped[_corpus_name(example)].append(example)
    return dict(grouped)


def _corpus_name(example: Mapping[str, Any]) -> str:
    stem = Path(str(example.get("conllu", ""))).stem
    if stem.endswith("-ud-test"):
        stem = stem[: -len("-ud-test")]
    return stem or "unknown"


def _train_modifier_filter(
    examples: Sequence[Dict[str, Any]],
    label_set: AbstractSet[str],
    sentence_cache: Dict[Path, Mapping[str, UDSentence]],
) -> Tuple[ModifierFilter, Dict[str, Any]]:
    predicted, gold, details = _collect_relations(
        examples,
        label_set,
        sentence_cache,
        filtered_modifier_lemmas={},
    )
    gold_set = set(gold)
    stats: Dict[str, Dict[str, int]] = {
        name: {"predicted": 0, "true_positive": 0, "false_positive": 0}
        for name in FILTER_CANDIDATES
    }

    for relation in predicted:
        candidate = _candidate_for_relation(details.get(relation))
        if candidate is None:
            continue
        stats[candidate]["predicted"] += 1
        if relation in gold_set:
            stats[candidate]["true_positive"] += 1
        else:
            stats[candidate]["false_positive"] += 1

    selected: ModifierFilter = {}
    for name, item in stats.items():
        if item["false_positive"] > 0 and item["true_positive"] == 0:
            deprel, lemmas = FILTER_CANDIDATES[name]
            selected[deprel] = set(selected.get(deprel, set())) | set(lemmas)
            item["selected"] = 1
        else:
            item["selected"] = 0

    return selected, stats


def _candidate_for_relation(detail: Optional[Dict[str, Any]]) -> Optional[str]:
    if detail is None or detail.get("label") != "MODIFIER":
        return None
    deprel = str(detail.get("deprel", ""))
    lemma = str(detail.get("lemma", ""))
    for name, (candidate_deprel, lemmas) in FILTER_CANDIDATES.items():
        if deprel == candidate_deprel and lemma in lemmas:
            return name
    return None


def _evaluate_examples(
    examples: Sequence[Dict[str, Any]],
    label_set: AbstractSet[str],
    sentence_cache: Dict[Path, Mapping[str, UDSentence]],
    filtered_modifier_lemmas: Mapping[str, AbstractSet[str]],
) -> Dict[str, Any]:
    predicted, gold, _details = _collect_relations(
        examples,
        label_set,
        sentence_cache,
        filtered_modifier_lemmas=filtered_modifier_lemmas,
    )
    relation = _set_prf(predicted, gold)
    return {
        "metrics": {
            "examples": len(examples),
            "gold_relations": len(gold),
            "predicted_relations": len(predicted),
            "semantic_relation_prf": relation,
            "by_label": _by_label_prf(predicted, gold, sorted(label_set)),
        }
    }


def _collect_relations(
    examples: Sequence[Dict[str, Any]],
    label_set: AbstractSet[str],
    sentence_cache: Dict[Path, Mapping[str, UDSentence]],
    filtered_modifier_lemmas: Mapping[str, AbstractSet[str]],
) -> Tuple[List[RelationTuple], List[RelationTuple], Dict[RelationTuple, Dict[str, Any]]]:
    predicted: List[RelationTuple] = []
    gold: List[RelationTuple] = []
    details: Dict[RelationTuple, Dict[str, Any]] = {}

    for example in examples:
        sentence = _sentence_for_example(example, sentence_cache)
        result = dag_from_ud(
            sentence,
            filtered_modifier_lemmas=filtered_modifier_lemmas,
        )

        for edge in result.dag.edges:
            src = _node_position(str(edge.get("src", "")))
            dst = _node_position(str(edge.get("dst", "")))
            label = str(edge.get("label", ""))
            if src is None or dst is None or label not in label_set:
                continue
            relation = (example["id"], str(src), str(dst), label)
            predicted.append(relation)
            token = sentence.tokens[dst]
            details[relation] = {
                "deprel": token.deprel,
                "lemma": token.lemma,
                "label": label,
            }

        for relation in example.get("relations", []):
            label = str(relation.get("label", ""))
            if label not in label_set:
                continue
            gold.append(
                (
                    example["id"],
                    str(relation.get("src", "")),
                    str(relation.get("dst", "")),
                    label,
                )
            )

    return predicted, gold, details


def _sentence_for_example(
    example: Mapping[str, Any],
    sentence_cache: Dict[Path, Mapping[str, UDSentence]],
) -> UDSentence:
    conllu_path = (PROJECT_ROOT / str(example["conllu"])).resolve()
    if conllu_path not in sentence_cache:
        sentence_cache[conllu_path] = {
            sentence.sent_id: sentence for sentence in read_conllu(conllu_path)
        }
    sentence = sentence_cache[conllu_path].get(str(example["sent_id"]))
    if sentence is None:
        raise KeyError(f"Could not find sent_id={example['sent_id']} in {conllu_path}")
    return sentence


def _node_position(node_id: str) -> Optional[int]:
    if not node_id.startswith("n"):
        return None
    try:
        return int(node_id[1:])
    except ValueError:
        return None


def _set_prf(predicted: Iterable[RelationTuple], gold: Iterable[RelationTuple]) -> Dict[str, float | int]:
    pred_counts = Counter(predicted)
    gold_counts = Counter(gold)
    tp = sum(min(count, gold_counts.get(item, 0)) for item, count in pred_counts.items())
    predicted_total = sum(pred_counts.values())
    gold_total = sum(gold_counts.values())
    precision = tp / predicted_total if predicted_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": gold_total,
    }


def _by_label_prf(
    predicted: Sequence[RelationTuple],
    gold: Sequence[RelationTuple],
    labels: Sequence[str],
) -> Dict[str, Dict[str, float | int]]:
    return {
        label: _set_prf(
            [relation for relation in predicted if relation[3] == label],
            [relation for relation in gold if relation[3] == label],
        )
        for label in labels
    }


def _filter_names(filtered_modifier_lemmas: Mapping[str, AbstractSet[str]]) -> List[str]:
    selected = []
    for name, (deprel, lemmas) in FILTER_CANDIDATES.items():
        if set(lemmas).issubset(set(filtered_modifier_lemmas.get(deprel, set()))):
            selected.append(name)
    return selected


def _macro_metric(fold_reports: Sequence[Dict[str, Any]], metric: str) -> float:
    if not fold_reports:
        return 0.0
    values = [
        float(fold["test_metrics"]["semantic_relation_prf"][metric])
        for fold in fold_reports
    ]
    return round(sum(values) / len(values), 4)


def _write_fold_files(split_dir: Path, fold: Mapping[str, Any]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(split_dir / f"{fold['name']}_train.jsonl", fold["train_examples"])
    _write_jsonl(split_dir / f"{fold['name']}_test.jsonl", fold["test_examples"])


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_summary(report: Dict[str, Any], output_path: Path, split_dir: Optional[Path]) -> None:
    print("=" * 72)
    print("Historical semantic DAG split evaluation")
    print("=" * 72)
    print(f"  mode                       : {report['mode']}")
    print(f"  folds                      : {len(report['folds'])}")
    print(f"  macro test P/R/F1          : {report['macro_test_precision']} / {report['macro_test_recall']} / {report['macro_test_f1']}")
    for fold in report["folds"]:
        relation = fold["test_metrics"]["semantic_relation_prf"]
        print(
            f"  {fold['name']:<26} P/R/F1: "
            f"{relation['precision']} / {relation['recall']} / {relation['f1']} "
            f"(selected={','.join(fold['selected_filters']) or 'none'})"
        )
    print(f"\nFull report written to: {output_path}")
    if split_dir is not None:
        print(f"Split files written to: {split_dir}")


def main() -> None:
    args = parse_args()
    evaluate_split(
        gold_path=args.gold,
        output_path=args.output,
        split_dir=args.split_dir,
        labels=args.labels,
        mode=args.mode,
        test_ratio=args.test_ratio,
        seed=args.seed,
        test_corpora=args.test_corpus,
        write_split_files=not args.no_split_files,
    )


if __name__ == "__main__":
    main()
