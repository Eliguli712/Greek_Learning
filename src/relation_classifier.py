"""Lightweight supervised relation classifier for raw-pipeline DAG edges."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.normalize import surface_key


NEGATIVE_LABEL = "NONE"
ProgressCallback = Callable[[str], None]
_CLASSIFIER_CACHE: Dict[str, Optional["RelationClassifier"]] = {}

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


def relation_classifier_from_project(
    project_root: Path,
    progress: Optional[ProgressCallback] = None,
) -> Optional[RelationClassifier]:
    root = project_root.resolve()
    cache_key = str(root)
    if cache_key in _CLASSIFIER_CACHE:
        _progress(progress, "using cached relation classifier")
        return _CLASSIFIER_CACHE[cache_key]

    classifier = _build_relation_classifier(root, progress)
    _CLASSIFIER_CACHE[cache_key] = classifier
    return classifier


def _build_relation_classifier(
    root: Path,
    progress: Optional[ProgressCallback] = None,
) -> Optional[RelationClassifier]:
    train_paths = sorted((root / "data" / "ud_treebanks").glob("*-train.conllu"))
    if not train_paths:
        _progress(progress, "no UD train files found under data/ud_treebanks")
        return None

    _progress(
        progress,
        "loading UD train files: " + ", ".join(path.name for path in train_paths),
    )
    train_sentences: List[UDSentence] = []
    for index, path in enumerate(train_paths, start=1):
        sentences = _read_conllu(path)
        train_sentences.extend(sentences)
        token_count = sum(len(sentence.tokens) for sentence in sentences)
        _progress(
            progress,
            f"loaded train file {index}/{len(train_paths)} {path.name}: "
            f"{len(sentences)} sentences, {token_count} tokens",
        )

    if not train_sentences:
        _progress(progress, "no usable UD train sentences found")
        return None

    _progress(progress, f"building training rows from {len(train_sentences)} train sentences")
    features, labels = _training_rows(train_sentences, progress=progress)
    if not features or len(set(labels)) < 2:
        _progress(progress, "not enough label variety to train relation classifier")
        return None

    label_counts = Counter(labels)
    label_summary = ", ".join(
        f"{label}={count}" for label, count in sorted(label_counts.items())
    )
    _progress(progress, f"built {len(features)} candidate rows ({label_summary})")

    model = Pipeline(
        [
            ("vec", DictVectorizer(sparse=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=300,
                    solver="lbfgs",
                    random_state=13,
                    verbose=1 if progress is not None else 0,
                ),
            ),
        ]
    )
    _progress(progress, "fitting sklearn LogisticRegression(max_iter=300, solver=lbfgs)")
    fit_started = perf_counter()
    model.fit(features, labels)
    _progress(progress, f"fit complete in {_elapsed(fit_started)}")
    return RelationClassifier(model=model, threshold=0.15)


def _training_rows(
    sentences: Sequence[UDSentence],
    progress: Optional[ProgressCallback] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    features: List[Dict[str, Any]] = []
    labels: List[str] = []
    total = len(sentences)
    progress_interval = max(1, total // 10)

    for sentence_index, sentence in enumerate(sentences, start=1):
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

        if sentence_index == total or sentence_index % progress_interval == 0:
            _progress(
                progress,
                f"feature extraction: {sentence_index}/{total} train sentences, "
                f"{len(features)} rows so far",
            )

    return features, labels


def _progress(callback: Optional[ProgressCallback], message: str) -> None:
    if callback is not None:
        callback(message)


def _elapsed(started: float) -> str:
    return f"{perf_counter() - started:.1f}s"


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
    features = _pair_features(
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
    features.update(
        _context_features(
            tokens=tokens,
            src=src,
            dst=dst,
            pos_getter=lambda token: token.upos,
            lemma_getter=lambda token: token.lemma,
            form_getter=lambda token: token.form,
            case_getter=lambda token: token.feats.get("Case", ""),
            number_getter=lambda token: token.feats.get("Number", ""),
            gender_getter=lambda token: token.feats.get("Gender", ""),
            function_getter=lambda token: token.upos in {"DET", "ADP", "CCONJ", "SCONJ", "PART"},
        )
    )
    return features


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

    features = _pair_features(
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
    features.update(
        _context_features(
            tokens=tokens,
            src=src,
            dst=dst,
            pos_getter=lambda token: str(token.get("pos", "")),
            lemma_getter=lambda token: str(token.get("lemma", "")),
            form_getter=lambda token: str(token.get("token", "")),
            case_getter=lambda token: str(
                ((token.get("features") or {}).get("ud_features") or {}).get("Case", "")
                or (token.get("features") or {}).get("case", "")
            ),
            number_getter=lambda token: str(
                ((token.get("features") or {}).get("ud_features") or {}).get("Number", "")
            ),
            gender_getter=lambda token: str(
                ((token.get("features") or {}).get("ud_features") or {}).get("Gender", "")
            ),
            function_getter=_semantic_token_is_function,
        )
    )
    return features


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


def _context_features(
    *,
    tokens: Sequence[Any],
    src: int,
    dst: int,
    pos_getter: Any,
    lemma_getter: Any,
    form_getter: Any,
    case_getter: Any,
    number_getter: Any,
    gender_getter: Any,
    function_getter: Any,
) -> Dict[str, Any]:
    src_pos = pos_getter(tokens[src])
    dst_pos = pos_getter(tokens[dst])
    src_case = case_getter(tokens[src])
    dst_case = case_getter(tokens[dst])
    src_number = number_getter(tokens[src])
    dst_number = number_getter(tokens[dst])
    src_gender = gender_getter(tokens[src])
    dst_gender = gender_getter(tokens[dst])
    lo, hi = sorted((src, dst))
    between = list(range(lo + 1, hi))

    def pos_at(index: int) -> str:
        if 0 <= index < len(tokens):
            return pos_getter(tokens[index])
        return "BOUNDARY"

    def lemma_at(index: int) -> str:
        if 0 <= index < len(tokens):
            return surface_key(lemma_getter(tokens[index]))
        return "BOUNDARY"

    def form_at(index: int) -> str:
        if 0 <= index < len(tokens):
            return surface_key(form_getter(tokens[index]))
        return "BOUNDARY"

    between_pos = [pos_getter(tokens[index]) for index in between]
    between_lemmas = {surface_key(lemma_getter(tokens[index])) for index in between}
    cconj_keys = {"\u03ba\u03b1\u03b9", "\u03c4\u03b5", "\u03b7", "\u03b7\u03b4\u03b5", "\u03bf\u03c5\u03b4\u03b5"}

    nearest_event_left = _nearest_index_with_pos(tokens, dst, -1, {"VERB", "AUX"}, pos_getter)
    nearest_event_right = _nearest_index_with_pos(tokens, dst, 1, {"VERB", "AUX"}, pos_getter)
    nearest_nominal_left = _nearest_index_with_pos(
        tokens,
        dst,
        -1,
        {"NOUN", "PROPN", "PRON", "DET", "NUM", "ADJ"},
        pos_getter,
    )
    nearest_nominal_right = _nearest_index_with_pos(
        tokens,
        dst,
        1,
        {"NOUN", "PROPN", "PRON", "DET", "NUM", "ADJ"},
        pos_getter,
    )

    return {
        "src_prev_pos": pos_at(src - 1),
        "src_next_pos": pos_at(src + 1),
        "dst_prev_pos": pos_at(dst - 1),
        "dst_next_pos": pos_at(dst + 1),
        "src_prev_lemma": lemma_at(src - 1),
        "src_next_lemma": lemma_at(src + 1),
        "dst_prev_lemma": lemma_at(dst - 1),
        "dst_next_lemma": lemma_at(dst + 1),
        "src_prev_suffix_2": form_at(src - 1)[-2:],
        "dst_prev_suffix_2": form_at(dst - 1)[-2:],
        "between_len": len(between),
        "between_bucket": _distance_bucket(len(between)),
        "between_has_cconj": any(pos == "CCONJ" for pos in between_pos)
        or bool(between_lemmas & cconj_keys),
        "between_has_adp": any(pos == "ADP" for pos in between_pos),
        "between_has_det": any(pos == "DET" for pos in between_pos),
        "between_has_verb": any(pos in {"VERB", "AUX"} for pos in between_pos),
        "between_has_particle": any(pos == "PART" for pos in between_pos),
        "src_is_function": function_getter(tokens[src]),
        "dst_is_function": function_getter(tokens[dst]),
        "same_pos": bool(src_pos and src_pos == dst_pos),
        "same_case": bool(src_case and src_case == dst_case),
        "same_case_number": bool(src_case and src_case == dst_case and src_number and src_number == dst_number),
        "same_case_gender_number": bool(
            src_case
            and src_case == dst_case
            and src_number
            and src_number == dst_number
            and (not src_gender or not dst_gender or src_gender == dst_gender)
        ),
        "dst_prev_is_adp": pos_at(dst - 1) == "ADP",
        "dst_prev_is_det": pos_at(dst - 1) == "DET",
        "dst_prev_is_cconj": pos_at(dst - 1) == "CCONJ",
        "dst_next_is_cconj": pos_at(dst + 1) == "CCONJ",
        "src_is_nearest_event_left_of_dst": nearest_event_left == src,
        "src_is_nearest_event_right_of_dst": nearest_event_right == src,
        "src_is_nearest_nominal_left_of_dst": nearest_nominal_left == src,
        "src_is_nearest_nominal_right_of_dst": nearest_nominal_right == src,
        "src_pos_dst_prev_pos": f"{src_pos}->{pos_at(dst - 1)}",
        "dst_pos_dst_prev_pos": f"{dst_pos}<-{pos_at(dst - 1)}",
        "src_dst_prev_lemma": f"{surface_key(lemma_getter(tokens[src]))}->{lemma_at(dst - 1)}",
    }


def _nearest_index_with_pos(
    tokens: Sequence[Any],
    origin: int,
    direction: int,
    allowed_pos: set[str],
    pos_getter: Any,
) -> Optional[int]:
    index = origin + direction
    while 0 <= index < len(tokens):
        if pos_getter(tokens[index]) in allowed_pos:
            return index
        index += direction
    return None


def _semantic_token_is_function(token: Dict[str, Any]) -> bool:
    features = token.get("features") or {}
    if features.get("function_word") is True:
        return True
    return token.get("pos") in {"DET", "ADP", "CCONJ", "SCONJ", "PART"}


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
