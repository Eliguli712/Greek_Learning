import unittest
from greek_learning.layers.lexeme_normalizer import LexemeNormalizer


class TestLexemeNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = LexemeNormalizer()

    def test_known_greek_noun(self):
        result = self.normalizer.normalize('λόγος', [], language='greek')
        self.assertEqual(result['lemma'], 'λόγος')
        self.assertEqual(result['pos'], 'noun')
        self.assertTrue(result['in_lexicon'])

    def test_known_greek_noun_inflected(self):
        result = self.normalizer.normalize('λόγου', [], language='greek')
        self.assertEqual(result['lemma'], 'λόγος')
        self.assertEqual(result['pos'], 'noun')
        self.assertTrue(result['in_lexicon'])

    def test_known_greek_verb(self):
        result = self.normalizer.normalize('λέγει', [], language='greek')
        self.assertEqual(result['lemma'], 'λέγω')
        self.assertEqual(result['pos'], 'verb')
        self.assertTrue(result['in_lexicon'])

    def test_known_greek_article(self):
        result = self.normalizer.normalize('ὁ', [], language='greek')
        self.assertEqual(result['pos'], 'article')
        self.assertTrue(result['in_lexicon'])

    def test_unknown_word_uses_root(self):
        segments = [{'form': 'ξξξ', 'role': 'root'}, {'form': 'ος', 'role': 'ending'}]
        result = self.normalizer.normalize('ξξξος', segments, language='greek')
        self.assertEqual(result['lemma'], 'ξξξ')
        self.assertEqual(result['pos'], 'unknown')
        self.assertFalse(result['in_lexicon'])

    def test_unknown_word_no_segments(self):
        result = self.normalizer.normalize('ξξξ', [], language='greek')
        self.assertFalse(result['in_lexicon'])
        self.assertEqual(result['pos'], 'unknown')

    def test_known_latin_noun(self):
        result = self.normalizer.normalize('amor', [], language='latin')
        self.assertEqual(result['lemma'], 'amor')
        self.assertEqual(result['pos'], 'noun')
        self.assertTrue(result['in_lexicon'])

    def test_known_latin_verb(self):
        result = self.normalizer.normalize('amat', [], language='latin')
        self.assertEqual(result['lemma'], 'amo')
        self.assertEqual(result['pos'], 'verb')
        self.assertTrue(result['in_lexicon'])

    def test_surface_preserved(self):
        result = self.normalizer.normalize('λόγος', [], language='greek')
        self.assertEqual(result['surface'], 'λόγος')

    def test_known_greek_adjective(self):
        result = self.normalizer.normalize('καλός', [], language='greek')
        self.assertEqual(result['lemma'], 'καλός')
        self.assertEqual(result['pos'], 'adjective')


if __name__ == '__main__':
    unittest.main()
