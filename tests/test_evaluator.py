import unittest
from greek_learning.evaluation.evaluator import Evaluator, EvaluationLevel


def make_analysis(token_str, lemma, pos, sem_type, in_lexicon=True):
    return {
        'input': token_str,
        'tokens': [{
            'token': token_str,
            'morpheme_record': {'segments': [{'form': token_str, 'role': 'root'}]},
            'lexeme_record': {'lemma': lemma, 'pos': pos, 'in_lexicon': in_lexicon, 'surface': token_str},
            'semantic_token': {'lemma': lemma, 'pos': pos, 'semantic_type': sem_type, 'surface': token_str},
        }],
        'dag': {
            'nodes': {'ROOT': {}, 'LEX_0': {}},
            'edges': [{'edge_type': 'PREDICATE', 'source': 'ROOT', 'target': 'LEX_0', 'weight': 1.0}],
        },
    }


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator()

    def test_compute_metrics_perfect(self):
        p, r, f = self.evaluator.compute_metrics(10, 0, 0)
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(f, 1.0)

    def test_compute_metrics_zero(self):
        p, r, f = self.evaluator.compute_metrics(0, 5, 5)
        self.assertAlmostEqual(p, 0.0)
        self.assertAlmostEqual(r, 0.0)
        self.assertAlmostEqual(f, 0.0)

    def test_compute_metrics_partial(self):
        p, r, f = self.evaluator.compute_metrics(5, 5, 0)
        self.assertAlmostEqual(p, 0.5)
        self.assertAlmostEqual(r, 1.0)
        self.assertAlmostEqual(f, 2 * 0.5 * 1.0 / 1.5)

    def test_compute_metrics_all_zero(self):
        p, r, f = self.evaluator.compute_metrics(0, 0, 0)
        self.assertAlmostEqual(p, 0.0)
        self.assertAlmostEqual(r, 0.0)
        self.assertAlmostEqual(f, 0.0)

    def test_evaluate_parsing_perfect(self):
        pred = [make_analysis('λόγος', 'λόγος', 'noun', 'ENTITY')]
        gold = [make_analysis('λόγος', 'λόγος', 'noun', 'ENTITY')]
        result = self.evaluator.evaluate_parsing(pred, gold)
        self.assertAlmostEqual(result.precision, 1.0)
        self.assertAlmostEqual(result.recall, 1.0)
        self.assertAlmostEqual(result.f_measure, 1.0)

    def test_evaluate_parsing_no_match(self):
        pred = [{
            'input': 'xxx',
            'tokens': [{'morpheme_record': {'segments': [{'form': 'yyy', 'role': 'root'}]}}],
            'dag': {'nodes': {}, 'edges': []},
        }]
        gold = [{
            'input': 'xxx',
            'tokens': [{'morpheme_record': {'segments': [{'form': 'zzz', 'role': 'root'}]}}],
            'dag': {'nodes': {}, 'edges': []},
        }]
        result = self.evaluator.evaluate_parsing(pred, gold)
        self.assertAlmostEqual(result.precision, 0.0)
        self.assertAlmostEqual(result.recall, 0.0)
        self.assertAlmostEqual(result.f_measure, 0.0)

    def test_evaluate_interpretation(self):
        pred = [make_analysis('λόγος', 'λόγος', 'noun', 'ENTITY')]
        gold = [make_analysis('λόγος', 'λόγος', 'noun', 'ENTITY')]
        result = self.evaluator.evaluate_interpretation(pred, gold)
        self.assertEqual(result.level, EvaluationLevel.INTERPRETATION)
        self.assertAlmostEqual(result.precision, 1.0)

    def test_evaluate_data_collection(self):
        pred = [make_analysis('λόγος', 'λόγος', 'noun', 'ENTITY')]
        gold = [make_analysis('λόγος', 'λόγος', 'noun', 'ENTITY')]
        result = self.evaluator.evaluate_data_collection(pred, gold)
        self.assertEqual(result.level, EvaluationLevel.DATA_COLLECTION)
        self.assertAlmostEqual(result.precision, 1.0)

    def test_evaluate_transcription(self):
        pred = ['λογος']
        gold = ['λογος']
        result = self.evaluator.evaluate_transcription(pred, gold)
        self.assertEqual(result.level, EvaluationLevel.TRANSCRIPTION)
        self.assertAlmostEqual(result.precision, 1.0)

    def test_evaluate_passage_generation(self):
        pred = [make_analysis('λόγος', 'λόγος', 'noun', 'ENTITY')]
        gold = [make_analysis('λόγος', 'λόγος', 'noun', 'ENTITY')]
        result = self.evaluator.evaluate_passage_generation(pred, gold)
        self.assertEqual(result.level, EvaluationLevel.PASSAGE_GENERATION)

    def test_evaluate_all_returns_five_levels(self):
        pred = [make_analysis('λόγος', 'λόγος', 'noun', 'ENTITY')]
        gold = [make_analysis('λόγος', 'λόγος', 'noun', 'ENTITY')]
        results = self.evaluator.evaluate_all(pred, gold)
        self.assertEqual(len(results), 5)
        for level in EvaluationLevel:
            self.assertIn(level.value, results)

    def test_evaluation_result_level(self):
        pred = [make_analysis('λόγος', 'λόγος', 'noun', 'ENTITY')]
        gold = [make_analysis('λόγος', 'λόγος', 'noun', 'ENTITY')]
        result = self.evaluator.evaluate_parsing(pred, gold)
        self.assertEqual(result.level, EvaluationLevel.PARSING)


if __name__ == '__main__':
    unittest.main()
