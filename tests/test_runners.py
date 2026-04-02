import unittest
from greek_learning.runners.single_runner import SingleRunner
from greek_learning.runners.batch_runner import BatchRunner
from greek_learning.runners.corpus_runner import CorpusRunner


class TestSingleRunner(unittest.TestCase):
    def setUp(self):
        self.runner = SingleRunner(language='greek')

    def test_run_returns_dict(self):
        result = self.runner.run('ὁ ἄνθρωπος λέγει')
        self.assertIsInstance(result, dict)

    def test_run_has_required_keys(self):
        result = self.runner.run('ὁ ἄνθρωπος λέγει')
        for key in ('input', 'language', 'tokens', 'relations', 'dag'):
            self.assertIn(key, result)

    def test_run_token_count(self):
        result = self.runner.run('ὁ ἄνθρωπος λέγει')
        self.assertEqual(len(result['tokens']), 3)

    def test_run_token_fields(self):
        result = self.runner.run('ὁ ἄνθρωπος λέγει')
        for tok in result['tokens']:
            for key in ('token', 'phoneme_record', 'morpheme_record', 'lexeme_record', 'semantic_token'):
                self.assertIn(key, tok)

    def test_run_dag_serializable(self):
        import json
        result = self.runner.run('ὁ ἄνθρωπος λέγει')
        json_str = json.dumps(result['dag'])
        self.assertIsInstance(json_str, str)

    def test_run_known_lexicon_entry(self):
        result = self.runner.run('λέγει')
        tok = result['tokens'][0]
        self.assertEqual(tok['lexeme_record']['lemma'], 'λέγω')

    def test_run_language_preserved(self):
        result = self.runner.run('λέγει')
        self.assertEqual(result['language'], 'greek')

    def test_run_single_token(self):
        result = self.runner.run('λόγος')
        self.assertEqual(len(result['tokens']), 1)


class TestBatchRunner(unittest.TestCase):
    def setUp(self):
        self.runner = BatchRunner(language='greek')

    def test_run_list(self):
        items = ['ὁ ἄνθρωπος λέγει', 'λόγος καλός']
        results = self.runner.run(items)
        self.assertEqual(len(results), 2)

    def test_run_empty_list(self):
        results = self.runner.run([])
        self.assertEqual(results, [])

    def test_max_items(self):
        runner = BatchRunner(language='greek', max_items=2)
        items = ['λόγος', 'θεός', 'ἄνθρωπος', 'ψυχή']
        results = runner.run(items)
        self.assertEqual(len(results), 2)

    def test_run_controlled_min_tokens(self):
        items = ['λόγος', 'ὁ ἄνθρωπος λέγει', 'θεός καλός']
        results = self.runner.run_controlled(items, filters={'min_tokens': 3})
        # Only 'ὁ ἄνθρωπος λέγει' has 3 tokens
        self.assertEqual(len(results), 1)

    def test_run_controlled_max_tokens(self):
        items = ['λόγος', 'ὁ ἄνθρωπος λέγει', 'θεός καλός']
        results = self.runner.run_controlled(items, filters={'max_tokens': 1})
        # Only 'λόγος' has 1 token
        self.assertEqual(len(results), 1)

    def test_run_controlled_language_override(self):
        items = ['amor amat']
        results = self.runner.run_controlled(items, filters={'language': 'latin'})
        self.assertEqual(len(results), 1)
        # Latin lexicon should recognize 'amat'
        tokens = results[0]['tokens']
        amat_tok = next((t for t in tokens if t['token'] == 'amat'), None)
        self.assertIsNotNone(amat_tok)
        self.assertEqual(amat_tok['lexeme_record']['lemma'], 'amo')


class TestCorpusRunner(unittest.TestCase):
    def setUp(self):
        self.runner = CorpusRunner(language='greek')

    def test_run_stream_list(self):
        lines = ['ὁ ἄνθρωπος λέγει', 'λόγος καλός', '']
        results = list(self.runner.run_stream(lines))
        # Empty line should be skipped
        self.assertEqual(len(results), 2)

    def test_run_stream_generator(self):
        def gen():
            yield 'λόγος'
            yield 'θεός'
        results = list(self.runner.run_stream(gen()))
        self.assertEqual(len(results), 2)

    def test_run_stream_result_structure(self):
        lines = ['ὁ ἄνθρωπος λέγει']
        result = next(self.runner.run_stream(lines))
        for key in ('input', 'language', 'tokens', 'relations', 'dag'):
            self.assertIn(key, result)

    def test_run_stream_skips_empty(self):
        lines = ['', '   ', 'λόγος']
        results = list(self.runner.run_stream(lines))
        # Only 'λόγος' should produce a result (spaces-only stripped to empty)
        self.assertEqual(len(results), 1)


if __name__ == '__main__':
    unittest.main()
