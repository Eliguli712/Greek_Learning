from dataclasses import dataclass, field, asdict
from enum import Enum


class EvaluationLevel(Enum):
    PARSING = "parsing"
    INTERPRETATION = "interpretation"
    DATA_COLLECTION = "data_collection"
    TRANSCRIPTION = "transcription"
    PASSAGE_GENERATION = "passage_generation"


@dataclass
class EvaluationResult:
    level: EvaluationLevel
    precision: float
    recall: float
    f_measure: float
    true_positives: int
    false_positives: int
    false_negatives: int
    details: dict = field(default_factory=dict)

    def to_dict(self):
        d = asdict(self)
        d['level'] = self.level.value
        return d


class Evaluator:
    def compute_metrics(self, tp: int, fp: int, fn: int):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f_measure

    def evaluate_parsing(self, predicted: list, gold: list) -> EvaluationResult:
        """Compare morpheme segments between predicted and gold analyses."""
        tp = fp = fn = 0
        details = {}

        for i, (pred, gld) in enumerate(zip(predicted, gold)):
            pred_segments = frozenset(
                (s['form'], s['role'])
                for t in pred.get('tokens', [])
                for s in t.get('morpheme_record', {}).get('segments', [])
            )
            gold_segments = frozenset(
                (s['form'], s['role'])
                for t in gld.get('tokens', [])
                for s in t.get('morpheme_record', {}).get('segments', [])
            )
            tp_i = len(pred_segments & gold_segments)
            fp_i = len(pred_segments - gold_segments)
            fn_i = len(gold_segments - pred_segments)
            tp += tp_i
            fp += fp_i
            fn += fn_i
            details[f'item_{i}'] = {'tp': tp_i, 'fp': fp_i, 'fn': fn_i}

        precision, recall, f_measure = self.compute_metrics(tp, fp, fn)
        return EvaluationResult(
            level=EvaluationLevel.PARSING,
            precision=precision,
            recall=recall,
            f_measure=f_measure,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            details=details,
        )

    def evaluate_interpretation(self, predicted: list, gold: list) -> EvaluationResult:
        """Compare semantic tokens: lemma, pos, semantic_type."""
        tp = fp = fn = 0
        details = {}

        for i, (pred, gld) in enumerate(zip(predicted, gold)):
            pred_tokens = pred.get('tokens', [])
            gold_tokens = gld.get('tokens', [])

            pred_set = frozenset(
                (t.get('semantic_token', {}).get('lemma', ''),
                 t.get('semantic_token', {}).get('pos', ''),
                 t.get('semantic_token', {}).get('semantic_type', ''))
                for t in pred_tokens
            )
            gold_set = frozenset(
                (t.get('semantic_token', {}).get('lemma', ''),
                 t.get('semantic_token', {}).get('pos', ''),
                 t.get('semantic_token', {}).get('semantic_type', ''))
                for t in gold_tokens
            )
            tp_i = len(pred_set & gold_set)
            fp_i = len(pred_set - gold_set)
            fn_i = len(gold_set - pred_set)
            tp += tp_i
            fp += fp_i
            fn += fn_i
            details[f'item_{i}'] = {'tp': tp_i, 'fp': fp_i, 'fn': fn_i}

        precision, recall, f_measure = self.compute_metrics(tp, fp, fn)
        return EvaluationResult(
            level=EvaluationLevel.INTERPRETATION,
            precision=precision,
            recall=recall,
            f_measure=f_measure,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            details=details,
        )

    def evaluate_data_collection(self, predicted: list, gold: list) -> EvaluationResult:
        """Compare lexeme records: in_lexicon, lemma accuracy."""
        tp = fp = fn = 0
        details = {}

        for i, (pred, gld) in enumerate(zip(predicted, gold)):
            pred_tokens = pred.get('tokens', [])
            gold_tokens = gld.get('tokens', [])

            pred_set = frozenset(
                (t.get('lexeme_record', {}).get('lemma', ''),
                 t.get('lexeme_record', {}).get('in_lexicon', False))
                for t in pred_tokens
            )
            gold_set = frozenset(
                (t.get('lexeme_record', {}).get('lemma', ''),
                 t.get('lexeme_record', {}).get('in_lexicon', False))
                for t in gold_tokens
            )
            tp_i = len(pred_set & gold_set)
            fp_i = len(pred_set - gold_set)
            fn_i = len(gold_set - pred_set)
            tp += tp_i
            fp += fp_i
            fn += fn_i
            details[f'item_{i}'] = {'tp': tp_i, 'fp': fp_i, 'fn': fn_i}

        precision, recall, f_measure = self.compute_metrics(tp, fp, fn)
        return EvaluationResult(
            level=EvaluationLevel.DATA_COLLECTION,
            precision=precision,
            recall=recall,
            f_measure=f_measure,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            details=details,
        )

    def evaluate_transcription(self, predicted: list, gold: list) -> EvaluationResult:
        """Character-level comparison of normalized forms."""
        tp = fp = fn = 0
        details = {}

        for i, (pred_str, gold_str) in enumerate(zip(predicted, gold)):
            pred_chars = set(pred_str)
            gold_chars = set(gold_str)
            tp_i = len(pred_chars & gold_chars)
            fp_i = len(pred_chars - gold_chars)
            fn_i = len(gold_chars - pred_chars)
            tp += tp_i
            fp += fp_i
            fn += fn_i
            details[f'item_{i}'] = {'tp': tp_i, 'fp': fp_i, 'fn': fn_i}

        precision, recall, f_measure = self.compute_metrics(tp, fp, fn)
        return EvaluationResult(
            level=EvaluationLevel.TRANSCRIPTION,
            precision=precision,
            recall=recall,
            f_measure=f_measure,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            details=details,
        )

    def evaluate_passage_generation(self, predicted: list, gold: list) -> EvaluationResult:
        """Compare DAG structure: node counts, edge type distributions."""
        tp = fp = fn = 0
        details = {}

        for i, (pred, gld) in enumerate(zip(predicted, gold)):
            pred_dag = pred.get('dag', {})
            gold_dag = gld.get('dag', {})

            pred_node_count = len(pred_dag.get('nodes', {}))
            gold_node_count = len(gold_dag.get('nodes', {}))

            pred_edge_types = frozenset(e.get('edge_type', '') for e in pred_dag.get('edges', []))
            gold_edge_types = frozenset(e.get('edge_type', '') for e in gold_dag.get('edges', []))

            tp_i = len(pred_edge_types & gold_edge_types)
            fp_i = len(pred_edge_types - gold_edge_types)
            fn_i = len(gold_edge_types - pred_edge_types)
            tp += tp_i
            fp += fp_i
            fn += fn_i
            details[f'item_{i}'] = {
                'tp': tp_i, 'fp': fp_i, 'fn': fn_i,
                'pred_nodes': pred_node_count,
                'gold_nodes': gold_node_count,
            }

        precision, recall, f_measure = self.compute_metrics(tp, fp, fn)
        return EvaluationResult(
            level=EvaluationLevel.PASSAGE_GENERATION,
            precision=precision,
            recall=recall,
            f_measure=f_measure,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            details=details,
        )

    def evaluate_all(self, predicted_analyses: list, gold_analyses: list) -> dict:
        return {
            EvaluationLevel.PARSING.value: self.evaluate_parsing(predicted_analyses, gold_analyses),
            EvaluationLevel.INTERPRETATION.value: self.evaluate_interpretation(predicted_analyses, gold_analyses),
            EvaluationLevel.DATA_COLLECTION.value: self.evaluate_data_collection(predicted_analyses, gold_analyses),
            EvaluationLevel.TRANSCRIPTION.value: self.evaluate_transcription(
                [a.get('input', '') for a in predicted_analyses],
                [a.get('input', '') for a in gold_analyses],
            ),
            EvaluationLevel.PASSAGE_GENERATION.value: self.evaluate_passage_generation(predicted_analyses, gold_analyses),
        }
