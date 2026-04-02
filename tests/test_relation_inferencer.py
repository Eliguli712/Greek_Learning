import unittest
from greek_learning.layers.relation_inferencer import RelationInferencer


def make_token(surface, sem_type, sem_role='UNKNOWN'):
    return {
        'surface': surface,
        'lemma': surface,
        'pos': 'unknown',
        'semantic_type': sem_type,
        'semantic_role': sem_role,
        'features': {},
    }


class TestRelationInferencer(unittest.TestCase):
    def setUp(self):
        self.inferencer = RelationInferencer()

    def test_simple_subject_predicate(self):
        # article + noun + verb
        tokens = [
            make_token('ὁ', 'FUNCTION', 'DETERMINER'),
            make_token('ἄνθρωπος', 'ENTITY', 'THEME'),
            make_token('λέγει', 'EVENT', 'PREDICATE'),
        ]
        result = self.inferencer.infer(tokens)
        self.assertIn('relations', result)
        self.assertEqual(len(result['relations']), 1)
        rel = result['relations'][0]
        self.assertEqual(rel['predicate'], 2)
        self.assertEqual(rel['subject'], 1)

    def test_subject_predicate_object(self):
        # noun + verb + noun
        tokens = [
            make_token('ἄνθρωπος', 'ENTITY', 'THEME'),
            make_token('φιλεῖ', 'EVENT', 'PREDICATE'),
            make_token('θεόν', 'ENTITY', 'THEME'),
        ]
        result = self.inferencer.infer(tokens)
        rel = result['relations'][0]
        self.assertEqual(rel['subject'], 0)
        self.assertEqual(rel['predicate'], 1)
        self.assertEqual(rel['object'], 2)

    def test_modifier_attached(self):
        # adjective + noun + verb
        tokens = [
            make_token('καλός', 'PROPERTY', 'MODIFIER'),
            make_token('ἄνθρωπος', 'ENTITY', 'THEME'),
            make_token('λέγει', 'EVENT', 'PREDICATE'),
        ]
        result = self.inferencer.infer(tokens)
        rel = result['relations'][0]
        self.assertIn(0, rel['modifiers'])

    def test_no_verb_produces_clause(self):
        tokens = [
            make_token('ἄνθρωπος', 'ENTITY', 'THEME'),
            make_token('καλός', 'PROPERTY', 'MODIFIER'),
        ]
        result = self.inferencer.infer(tokens)
        self.assertEqual(len(result['relations']), 0)
        self.assertEqual(len(result['clauses']), 1)

    def test_operator_in_relation(self):
        tokens = [
            make_token('ἄνθρωπος', 'ENTITY', 'THEME'),
            make_token('καί', 'OPERATOR', 'CONNECTOR'),
            make_token('θεός', 'ENTITY', 'THEME'),
            make_token('λέγει', 'EVENT', 'PREDICATE'),
        ]
        result = self.inferencer.infer(tokens)
        rel = result['relations'][0]
        self.assertIn(1, rel['operators'])

    def test_clauses_non_empty(self):
        tokens = [
            make_token('ἄνθρωπος', 'ENTITY', 'THEME'),
            make_token('λέγει', 'EVENT', 'PREDICATE'),
        ]
        result = self.inferencer.infer(tokens)
        self.assertGreater(len(result['clauses']), 0)

    def test_tokens_returned(self):
        tokens = [make_token('λέγει', 'EVENT', 'PREDICATE')]
        result = self.inferencer.infer(tokens)
        self.assertEqual(len(result['tokens']), 1)

    def test_multiple_verbs(self):
        tokens = [
            make_token('ἄνθρωπος', 'ENTITY', 'THEME'),
            make_token('λέγει', 'EVENT', 'PREDICATE'),
            make_token('θεός', 'ENTITY', 'THEME'),
            make_token('φιλεῖ', 'EVENT', 'PREDICATE'),
        ]
        result = self.inferencer.infer(tokens)
        self.assertEqual(len(result['relations']), 2)


if __name__ == '__main__':
    unittest.main()
