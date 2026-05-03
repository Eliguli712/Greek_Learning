"""Lightweight supervised relation classifier for raw-pipeline DAG edges."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.normalize import surface_key


NEGATIVE_LABEL = "NONE"

UD_RELATION_MAP: Dict[str, str] = {
    "nsubj": "AGENT",
    "csubj": "AGENT",
    "nsubj:pass": "THEME",
    "csubj:pass": "THEME",
    "obj": "THEME",
    "iobj": "THEME",
    "obl:arg": "THEME",
    "amod": "MODIFIER",
    "nummod": "MODIFIER",
    "advmod": "MODIFIER",
    "conj": "COORD",
}


@dataclass(frozen=True)
class UDToken:
    index: int
    form: str
    lemma: str
    upos: str
    feats: Dict[str, str]
    head: int
    deprel: str


@dataclass(frozen=True)
class UDSentence:
    tokens: List[UDToken]


@dataclass(frozen=True)
class ClassifiedRelation:
    src: int
    dst: int
    label: str
    confidence: float


@dataclass
class RelationClassifier:
    model: Pipeline
    threshold: float = 0.15
    max_distance: int = 12

    def predict(self, semantic_tokens: Sequence[Dict[str, Any]]) -> List[ClassifiedRelation]:
        if len(semantic_tokens) < 2:
            return []

        candidates: List[Tuple[int, int]] = []
        features: List[Dict[str, Any]] = []
        for src, dst in _candidate_pairs(len(semantic_tokens), self.max_distance):
            candidates.append((src, dst))
            features.append(_features_from_semantic_tokens(semantic_tokens, src, dst))

        if not features:
            return []

        probabilities = self.model.predict_proba(features)
        classes = list(self.model.classes_)
        none_index = classes.index(NEGATIVE_LABEL) if NEGATIVE_LABEL in classes else -1

        best_by_dependent: Dict[int, Tuple[float, int, str]] = {}
        for (src, dst), row in zip(candidates, probabilities):
            label_index, score = _best_non_none(row, none_index)
            if label_index is None or score < self.threshold:
                continue
            label = str(classes[label_index])
            current = best_by_dependent.get(dst)
            if current is None or score > current[0]:
                best_by_dependent[dst] = (float(score), src, label)

        relations: List[ClassifiedRelation] = []
        for dst, (score, src, label) in sorted(best_by_dependent.items()):
            relations.append(
                ClassifiedRelation(
                    src=src,
                    dst=dst,
                    label=label,
                    confidence=score,
                )
            )
        return relations


def relation_classifier_from_project(project_root: Path) -> Optional[RelationClassifier]:
    return _cached_relation_classifier(str(project_root.resolve()))


@lru_cache(maxsize=4)
def _cached_relation_classifier(project_root: str) -> Optional[RelationClassifier]:
    root = Path(project_root)
    train_paths = sorted((root / "data" / "ud_treebanks").glob("*-train.conllu"))
    if not train_paths:
        return None

    train_sentences: List[UDSentence] = []
    for path in train_paths:
        train_sentences.extend(_read_conllu(path))

    if not train_sentences:
        return None

    features, labels = _training_rows(train_sentences)
    if not features or len(set(labels)) < 2:
        return None

    model = Pipeline(
        [
            ("vec", DictVectorizer(sparse=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=300,
                    solver="lbfgs",
                    random_state=13,
                ),
            ),
        ]
    )
    model.fit(features, labels)
    return RelationClassifier(model=model, threshold=0.15)


def _training_rows(sentences: Sequence[UDSentence]) -> Tuple[List[Dict[str, Any]], List[str]]:
    features: List[Dict[str, Any]] = []
    labels: List[str] = []

    for sentence in sentences:
        index_to_position = {token.index: pos for pos, token in enumerate(sentence.tokens)}
        positive: Dict[Tuple[int, int], str] = {}
        for dep_pos, token in enumerate(sentence.tokens):
            label = UD_RELATION_MAP.get(token.deprel)
            head_pos = index_to_position.get(token.head)
            if label is None or head_pos is None or head_pos == dep_pos:
                continue
            positive[(head_pos, dep_pos)] = label

        candidate_pairs = set(_candidate_pairs(len(sentence.tokens), max_distance=12))
        candidate_pairs.update(positive.keys())

        negatives_added = 0
        negative_limit = max(40, 10 * max(1, len(positive)))
        for src, dst in sorted(candidate_pairs):
            label = positive.get((src, dst), NEGATIVE_LABEL)
            if label == NEGATIVE_LABEL:
                if negatives_added >= negative_limit:
                    continue
                negatives_added += 1
            features.append(_features_from_ud_tokens(sentence.tokens, src, dst))
            labels.append(label)

    return features, labels


def _read_conllu(path: Path) -> List[UDSentence]:
    sentences: List[UDSentence] = []
    tokens: List[UDToken] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                if tokens:
                    sentences.append(UDSentence(tokens=list(tokens)))
                tokens = []
                continue

            if line.startswith("#"):
                continue

            fields = line.split("\t")
            if len(fields) != 10:
                continue
            token_id = fields[0]
            if "-" in token_id or "." in token_id:
                continue
            if fields[3] == "PUNCT":
                continue

            try:
                index = int(token_id)
                head = int(fields[6]) if fields[6].isdigit() else 0
            except ValueError:
                continue

            tokens.append(
                UDToken(
                    index=index,
                    form=fields[1],
                    lemma=fields[2],
                    upos=fields[3],
                    feats=_parse_feats(fields[5]),
                    head=head,
                    deprel=fields[7],
                )
            )

    if tokens:
        sentences.append(UDSentence(tokens=list(tokens)))
    return sentences


def _parse_feats(raw: str) -> Dict[str, str]:
    if not raw or raw == "_":
        return {}
    feats: Dict[str, str] = {}
    for item in raw.split("|"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        feats[key] = value
    return feats


def _candidate_pairs(length: int, max_distance: int) -> Iterable[Tuple[int, int]]:
    for src in range(length):
        for dst in range(length):
            if src == dst:
                continue
            if abs(src - dst) <= max_distance:
                yield src, dst


def _features_from_ud_tokens(tokens: Sequence[UDToken], src: int, dst: int) -> Dict[str, Any]:
    src_token = tokens[src]
    dst_token = tokens[dst]
    return _pair_features(
        src_pos=src_token.upos,
        dst_pos=dst_token.upos,
        src_form=src_token.form,
        dst_form=dst_token.form,
        src_lemma=src_token.lemma,
        dst_lemma=dst_token.lemma,
        src_case=src_token.feats.get("Case", ""),
        dst_case=dst_token.feats.get("Case", ""),
        src_number=src_token.feats.get("Number", ""),
        dst_number=dst_token.feats.get("Number", ""),
        src_gender=src_token.feats.get("Gender", ""),
        dst_gender=dst_token.feats.get("Gender", ""),
        src_voice=src_token.feats.get("Voice", ""),
        dst_voice=dst_token.feats.get("Voice", ""),
        src=src,
        dst=dst,
    )


def _features_from_semantic_tokens(
    tokens: Sequence[Dict[str, Any]],
    src: int,
    dst: int,
) -> Dict[str, Any]:
    src_token = tokens[src]
    dst_token = tokens[dst]
    src_features = src_token.get("features") or {}
    dst_features = dst_token.get("features") or {}
    src_ud = src_features.get("ud_features") or {}
    dst_ud = dst_features.get("ud_features") or {}

    return _pair_features(
        src_pos=str(src_token.get("pos", "")),
        dst_pos=str(dst_token.get("pos", "")),
        src_form=str(src_token.get("token", "")),
        dst_form=str(dst_token.get("token", "")),
        src_lemma=str(src_token.get("lemma", "")),
        dst_lemma=str(dst_token.get("lemma", "")),
        src_case=str(src_ud.get("Case", "") or src_features.get("case", "")),
        dst_case=str(dst_ud.get("Case", "") or dst_features.get("case", "")),
        src_number=str(src_ud.get("Number", "")),
        dst_number=str(dst_ud.get("Number", "")),
        src_gender=str(src_ud.get("Gender", "")),
        dst_gender=str(dst_ud.get("Gender", "")),
        src_voice=str(src_ud.get("Voice", "")),
        dst_voice=str(dst_ud.get("Voice", "")),
        src=src,
        dst=dst,
    )


def _pair_features(
    *,
    src_pos: str,
    dst_pos: str,
    src_form: str,
    dst_form: str,
    src_lemma: str,
    dst_lemma: str,
    src_case: str,
    dst_case: str,
    src_number: str,
    dst_number: str,
    src_gender: str,
    dst_gender: str,
    src_voice: str,
    dst_voice: str,
    src: int,
    dst: int,
) -> Dict[str, Any]:
    distance = dst - src
    src_key = surface_key(src_form)
    dst_key = surface_key(dst_form)
    return {
        "src_pos": src_pos,
        "dst_pos": dst_pos,
        "pos_pair": f"{src_pos}->{dst_pos}",
        "src_suffix_1": src_key[-1:],
        "src_suffix_2": src_key[-2:],
        "src_suffix_3": src_key[-3:],
        "dst_suffix_1": dst_key[-1:],
        "dst_suffix_2": dst_key[-2:],
        "dst_suffix_3": dst_key[-3:],
        "src_lemma": src_lemma,
        "dst_lemma": dst_lemma,
        "lemma_pair": f"{src_lemma}->{dst_lemma}",
        "src_case": src_case,
        "dst_case": dst_case,
        "case_pair": f"{src_case}->{dst_case}",
        "src_number": src_number,
        "dst_number": dst_number,
        "same_number": bool(src_number and src_number == dst_number),
        "src_gender": src_gender,
        "dst_gender": dst_gender,
        "same_gender": bool(src_gender and src_gender == dst_gender),
        "src_voice": src_voice,
        "dst_voice": dst_voice,
        "direction": "right" if distance > 0 else "left",
        "distance": abs(distance),
        "distance_bucket": _distance_bucket(abs(distance)),
        "adjacent": abs(distance) == 1,
        "src_is_predicate": src_pos in {"VERB", "AUX"},
        "dst_is_nominal": dst_pos in {"NOUN", "PROPN", "PRON", "DET", "NUM", "ADJ"},
    }


def _distance_bucket(distance: int) -> str:
    if distance <= 1:
        return "1"
    if distance <= 3:
        return "2-3"
    if distance <= 6:
        return "4-6"
    return "7+"


def _best_non_none(row: Sequence[float], none_index: int) -> Tuple[Optional[int], float]:
    best_index: Optional[int] = None
    best_score = 0.0
    for index, score in enumerate(row):
        if index == none_index:
            continue
        if score > best_score:
            best_index = index
            best_score = float(score)
    return best_index, best_score


__all__ = ["ClassifiedRelation", "RelationClassifier", "relation_classifier_from_project"]
