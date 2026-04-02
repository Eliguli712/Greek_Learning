import unittest
from greek_learning.layers.semantic_tokenizer import SemanticTokenizer, POS_TO_SEMANTIC_TYPE, POS_TO_DEFAULT_ROLE


class TestSemanticTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SemanticTokenizer()

    def _make_lexeme(self, surface, lemma, pos, in_lexicon=True):
        return {'surface': surface, 'lemma': lemma, 'pos': pos, 'in_lexicon': in_lexicon}

    def test_noun_semantic_type(self):
        lex = self._make_lexeme('λόγος', 'λόγος', 'noun')
        result = self.tokenizer.tokenize(lex)
        self.assertEqual(result['semantic_type'], 'ENTITY')

    def test_verb_semantic_type(self):
        lex = self._make_lexeme('λέγει', 'λέγω', 'verb')
        result = self.tokenizer.tokenize(lex)
        self.assertEqual(result['semantic_type'], 'EVENT')

    def test_adjective_semantic_type(self):
        lex = self._make_lexeme('καλός', 'καλός', 'adjective')
        result = self.tokenizer.tokenize(lex)
        self.assertEqual(result['semantic_type'], 'PROPERTY')

    def test_preposition_semantic_type(self):
        lex = self._make_lexeme('ἐν', 'ἐν', 'preposition')
        result = self.tokenizer.tokenize(lex)
        self.assertEqual(result['semantic_type'], 'RELATION')

    def test_conjunction_semantic_type(self):
        lex = self._make_lexeme('καί', 'καί', 'conjunction')
        result = self.tokenizer.tokenize(lex)
        self.assertEqual(result['semantic_type'], 'OPERATOR')

    def test_article_semantic_type(self):
        lex = self._make_lexeme('ὁ', 'ὁ', 'article')
        result = self.tokenizer.tokenize(lex)
        self.assertEqual(result['semantic_type'], 'FUNCTION')

    def test_particle_semantic_type(self):
        lex = self._make_lexeme('δέ', 'δέ', 'particle')
        result = self.tokenizer.tokenize(lex)
        self.assertEqual(result['semantic_type'], 'DISCOURSE_MARKER')

    def test_unknown_pos(self):
        lex = self._make_lexeme('xxx', 'xxx', 'unknown', in_lexicon=False)
        result = self.tokenizer.tokenize(lex)
        self.assertEqual(result['semantic_type'], 'UNKNOWN')
        self.assertEqual(result['semantic_role'], 'UNKNOWN')

    def test_noun_default_role(self):
        lex = self._make_lexeme('λόγος', 'λόγος', 'noun')
        result = self.tokenizer.tokenize(lex)
        self.assertEqual(result['semantic_role'], 'THEME')

    def test_verb_default_role(self):
        lex = self._make_lexeme('λέγει', 'λέγω', 'verb')
        result = self.tokenizer.tokenize(lex)
        self.assertEqual(result['semantic_role'], 'PREDICATE')

    def test_adjective_default_role(self):
        lex = self._make_lexeme('καλός', 'καλός', 'adjective')
        result = self.tokenizer.tokenize(lex)
        self.assertEqual(result['semantic_role'], 'MODIFIER')

    def test_features_include_in_lexicon(self):
        lex = self._make_lexeme('λόγος', 'λόγος', 'noun')
        result = self.tokenizer.tokenize(lex)
        self.assertIn('in_lexicon', result['features'])

    def test_surface_and_lemma_preserved(self):
        lex = self._make_lexeme('λόγου', 'λόγος', 'noun')
        result = self.tokenizer.tokenize(lex)
        self.assertEqual(result['surface'], 'λόγου')
        self.assertEqual(result['lemma'], 'λόγος')

    def test_pos_to_semantic_type_mapping_completeness(self):
        for pos in ['noun', 'verb', 'adjective', 'adverb', 'preposition',
                    'conjunction', 'article', 'particle', 'pronoun', 'unknown']:
            self.assertIn(pos, POS_TO_SEMANTIC_TYPE)
            self.assertIn(pos, POS_TO_DEFAULT_ROLE)


if __name__ == '__main__':
    unittest.main()
