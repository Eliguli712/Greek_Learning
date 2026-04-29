"""
Evaluate the semantic compiler against Ancient Greek Universal Dependencies data.

This is a real-corpus evaluator for CoNLL-U files. It does not expect the UD data
to match this project's custom semantic format exactly, so it reports:

  * token alignment rate
  * lemma accuracy against UD lemmas
  * UPOS accuracy against UD coarse POS tags
  * proxy semantic-type accuracy from a deterministic UPOS -> semantic_type map
  * mapped relation precision/recall/F1 for common UD dependencies

Usage:
    python scripts/evaluate_ud.py --conllu data/real_eval/grc_perseus-ud-test.conllu
    python scripts/evaluate_ud.py --conllu path/a.conllu path/b.conllu --max-sentences 200
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.metrics import PRF, accuracy, set_prf
from src.normalize import surface_key
from src.pipeline import SemanticCompiler


UPOS_TO_SEMANTIC_TYPE = {
    "NOUN": "entity",
    "PROPN": "entity",
    "PRON": "indexical",
    "DET": "indexical",
    "NUM": "entity",
    "VERB": "event",
    "AUX": "event",
    "ADJ": "property",
    "ADV": "manner",
    "ADP": "relation",
    "CCONJ": "connector",
    "SCONJ": "connector",
    "PART": "operator",
    "INTJ": "unknown",
    "X": "unknown",
}


UD_RELATION_MAP = {
    "nsubj": "AGENT",
    "nsubj:pass": "THEME",
    "csubj": "AGENT",
    "obj": "THEME",
    "iobj": "THEME",
    "obl:arg": "THEME",
    "amod": "MODIFIER",
    "nummod": "MODIFIER",
    "advmod": "MODIFIER",
    "conj": "COORD",
}


@dataclass
class UDToken:
    index: int
    form: str
    lemma: str
    upos: str
    head: int
    deprel: str


@dataclass
class UDSentence:
    sent_id: str
    text: str
    tokens: List[UDToken]


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
        help="Maximum number of sentences to evaluate across all input files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "real_eval_ud.json",
        help="Path for the JSON report.",
    )
    parser.add_argument(
        "--include-sentences",
        action="store_true",
        help="Include per-sentence details in the JSON report.",
    )
    return parser.parse_args()


def read_conllu(path: Path) -> List[UDSentence]:
    sentences: List[UDSentence] = []
    metadata: Dict[str, str] = {}
    tokens: List[UDToken] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                if tokens:
                    sentences.append(_sentence_from_parts(metadata, tokens))
                metadata = {}
                tokens = []
                continue

            if line.startswith("#"):
                if "=" in line:
                    key, value = line[1:].split("=", 1)
                    metadata[key.strip()] = value.strip()
                continue

            fields = line.split("\t")
            if len(fields) != 10:
                continue

            token_id = fields[0]
            if "-" in token_id or "." in token_id:
                continue

            upos = fields[3]
            if upos == "PUNCT":
                continue

            form = fields[1]
            if not surface_key(form):
                continue

            try:
                index = int(token_id)
                head = int(fields[6]) if fields[6].isdigit() else 0
            except ValueError:
                continue

            tokens.append(
                UDToken(
                    index=index,
                    form=form,
                    lemma=fields[2],
                    upos=upos,
                    head=head,
                    deprel=fields[7],
                )
            )

    if tokens:
        sentences.append(_sentence_from_parts(metadata, tokens))

    return sentences


def _sentence_from_parts(metadata: Dict[str, str], tokens: List[UDToken]) -> UDSentence:
    text = metadata.get("text") or " ".join(token.form for token in tokens)
    sent_id = metadata.get("sent_id") or metadata.get("newdoc id") or f"sent_{len(tokens)}"
    return UDSentence(sent_id=sent_id, text=text, tokens=list(tokens))


def evaluate_ud(
    conllu_paths: Sequence[Path],
    max_sentences: int,
    include_sentences: bool = False,
) -> Dict[str, Any]:
    compiler = SemanticCompiler(project_root=PROJECT_ROOT)

    pred_lemmas: List[str] = []
    gold_lemmas: List[str] = []
    pred_upos: List[str] = []
    gold_upos: List[str] = []
    pred_semantic: List[str] = []
    gold_semantic: List[str] = []
    pred_relations: List[Tuple[str, str, str]] = []
    gold_relations: List[Tuple[str, str, str]] = []

    total_sentences = 0
    aligned_sentences = 0
    skipped_sentences = 0
    total_gold_tokens = 0
    aligned_tokens = 0
    per_sentence: List[Dict[str, Any]] = []

    for path in conllu_paths:
        for sentence in read_conllu(path):
            if total_sentences >= max_sentences:
                break

            total_sentences += 1
            gold_tokens_for_eval = list(sentence.tokens)
            total_gold_tokens += len(gold_tokens_for_eval)

            # Feed space-joined UD tokens to reduce punctuation and spacing drift.
            input_text = " ".join(token.form for token in gold_tokens_for_eval)
            result = compiler.analyze(input_text, language="ancient_greek")

            alignment = _align_tokens(result.tokens, gold_tokens_for_eval)
            if alignment is None:
                skipped_sentences += 1
                if include_sentences:
                    per_sentence.append(
                        {
                            "id": sentence.sent_id,
                            "status": "skipped_alignment",
                            "gold_tokens": [token.form for token in gold_tokens_for_eval],
                            "pred_tokens": result.tokens,
                        }
                    )
                continue

            aligned_sentences += 1
            aligned_tokens += len(alignment)

            gold_by_pred_index: Dict[int, UDToken] = {}
            ud_to_pred_index: Dict[int, int] = {}
            for pred_index, gold_token in alignment:
                gold_by_pred_index[pred_index] = gold_token
                ud_to_pred_index[gold_token.index] = pred_index

            sentence_pred_lemmas: List[str] = []
            sentence_gold_lemmas: List[str] = []

            for pred_index, gold_token in alignment:
                lexeme = _safe_index(result.lexeme_layer, pred_index)
                semantic = _safe_index(result.semantic_tokens, pred_index)

                pred_lemma = str(lexeme.get("lemma", "")) if "error" not in lexeme else ""
                pred_pos = str(semantic.get("pos", "")) if "error" not in semantic else ""
                pred_type = str(semantic.get("semantic_type", "")) if "error" not in semantic else ""
                gold_type = UPOS_TO_SEMANTIC_TYPE.get(gold_token.upos, "unknown")

                pred_lemmas.append(surface_key(pred_lemma))
                gold_lemmas.append(surface_key(gold_token.lemma))
                pred_upos.append(pred_pos)
                gold_upos.append(gold_token.upos)
                pred_semantic.append(pred_type)
                gold_semantic.append(gold_type)

                sentence_pred_lemmas.append(pred_lemma)
                sentence_gold_lemmas.append(gold_token.lemma)

            sent_pred_relations = _mapped_predicted_relations(result.relations)
            sent_gold_relations = _mapped_gold_relations(gold_tokens_for_eval, ud_to_pred_index)

            pred_relations.extend(
                _relation_with_sentence_id(sentence.sent_id, relation)
                for relation in sent_pred_relations
            )
            gold_relations.extend(
                _relation_with_sentence_id(sentence.sent_id, relation)
                for relation in sent_gold_relations
            )

            if include_sentences:
                per_sentence.append(
                    {
                        "id": sentence.sent_id,
                        "status": "aligned",
                        "text": input_text,
                        "gold_tokens": [token.form for token in gold_tokens_for_eval],
                        "pred_tokens": result.tokens,
                        "gold_lemmas": sentence_gold_lemmas,
                        "pred_lemmas": sentence_pred_lemmas,
                        "gold_relations": [list(item) for item in sent_gold_relations],
                        "pred_relations": [list(item) for item in sent_pred_relations],
                    }
                )

        if total_sentences >= max_sentences:
            break

    report: Dict[str, Any] = {
        "inputs": [str(path) for path in conllu_paths],
        "metrics": {
            "sentences_seen": total_sentences,
            "sentences_aligned": aligned_sentences,
            "sentences_skipped_alignment": skipped_sentences,
            "sentence_alignment_rate": _safe_div(aligned_sentences, total_sentences),
            "gold_tokens_seen": total_gold_tokens,
            "tokens_aligned": aligned_tokens,
            "token_alignment_rate": _safe_div(aligned_tokens, total_gold_tokens),
            "lemma_accuracy_normalized": round(accuracy(pred_lemmas, gold_lemmas), 4),
            "upos_accuracy": round(accuracy(pred_upos, gold_upos), 4),
            "semantic_type_proxy_accuracy": round(accuracy(pred_semantic, gold_semantic), 4),
            "mapped_relation_prf": set_prf(pred_relations, gold_relations).as_dict(),
        },
        "notes": [
            "Lemma accuracy is normalized with accent/case stripping.",
            "Semantic type is a proxy gold label derived from UD UPOS, not manually annotated semantics.",
            "Relation evaluation maps only common UD deprels into this repo's AGENT/THEME/MODIFIER/COORD inventory.",
            "Sentences with tokenization drift are skipped for token-level metrics.",
        ],
    }

    if include_sentences:
        report["sentences"] = per_sentence

    return report


def _align_tokens(
    predicted_tokens: Sequence[str],
    gold_tokens: Sequence[UDToken],
) -> Optional[List[Tuple[int, UDToken]]]:
    if len(predicted_tokens) != len(gold_tokens):
        return None

    alignment: List[Tuple[int, UDToken]] = []
    for pred_index, (pred, gold) in enumerate(zip(predicted_tokens, gold_tokens)):
        if surface_key(pred) != surface_key(gold.form):
            return None
        alignment.append((pred_index, gold))
    return alignment


def _safe_index(items: Sequence[Dict[str, Any]], index: int) -> Dict[str, Any]:
    if 0 <= index < len(items):
        return items[index]
    return {}


def _mapped_predicted_relations(edges: Iterable[Dict[str, Any]]) -> List[Tuple[int, int, str]]:
    relations: List[Tuple[int, int, str]] = []
    for edge in edges:
        src = _node_index(str(edge.get("src", "")))
        dst = _node_index(str(edge.get("dst", "")))
        label = str(edge.get("label", ""))
        if src >= 0 and dst >= 0 and label in set(UD_RELATION_MAP.values()):
            relations.append((src, dst, label))
    return relations


def _mapped_gold_relations(
    tokens: Sequence[UDToken],
    ud_to_pred_index: Dict[int, int],
) -> List[Tuple[int, int, str]]:
    relations: List[Tuple[int, int, str]] = []
    for token in tokens:
        label = UD_RELATION_MAP.get(token.deprel)
        if label is None:
            continue
        if token.head not in ud_to_pred_index or token.index not in ud_to_pred_index:
            continue

        head = ud_to_pred_index[token.head]
        dep = ud_to_pred_index[token.index]
        if head == dep:
            continue

        relations.append((head, dep, label))
    return relations


def _relation_with_sentence_id(
    sent_id: str,
    relation: Tuple[int, int, str],
) -> Tuple[str, int, int, str]:
    return (sent_id, relation[0], relation[1], relation[2])


def _node_index(node_id: str) -> int:
    if node_id.startswith("n"):
        try:
            return int(node_id[1:])
        except ValueError:
            return -1
    return -1


def _safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def main() -> None:
    args = parse_args()
    report = evaluate_ud(
        conllu_paths=args.conllu,
        max_sentences=args.max_sentences,
        include_sentences=args.include_sentences,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    metrics = report["metrics"]
    print("=" * 72)
    print("Real UD evaluation")
    print("=" * 72)
    print(f"  sentences seen/aligned      : {metrics['sentences_seen']} / {metrics['sentences_aligned']}")
    print(f"  token alignment rate        : {metrics['token_alignment_rate']}")
    print(f"  lemma accuracy normalized   : {metrics['lemma_accuracy_normalized']}")
    print(f"  UPOS accuracy               : {metrics['upos_accuracy']}")
    print(f"  semantic proxy accuracy     : {metrics['semantic_type_proxy_accuracy']}")
    rel = metrics["mapped_relation_prf"]
    print(f"  mapped relation P/R/F1      : {rel['precision']} / {rel['recall']} / {rel['f1']}")
    print(f"\nFull report written to: {args.output}")


if __name__ == "__main__":
    main()
