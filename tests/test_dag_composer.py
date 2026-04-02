import unittest
from greek_learning.layers.dag_composer import DAGComposer, SemanticDAG, DAGNode, DAGEdge


def make_token(surface, sem_type, sem_role='UNKNOWN'):
    return {
        'surface': surface,
        'lemma': surface,
        'pos': 'unknown',
        'semantic_type': sem_type,
        'semantic_role': sem_role,
        'features': {},
    }


def make_relation_record(tokens):
    from greek_learning.layers.relation_inferencer import RelationInferencer
    inferencer = RelationInferencer()
    return inferencer.infer(tokens)


class TestDAGComposer(unittest.TestCase):
    def setUp(self):
        self.composer = DAGComposer()

    def test_dag_has_root_node(self):
        tokens = [make_token('λέγει', 'EVENT', 'PREDICATE')]
        rr = make_relation_record(tokens)
        dag = self.composer.compose(rr, tokens)
        self.assertIn('ROOT', dag.nodes)

    def test_dag_has_lexeme_nodes(self):
        tokens = [
            make_token('ἄνθρωπος', 'ENTITY', 'THEME'),
            make_token('λέγει', 'EVENT', 'PREDICATE'),
        ]
        rr = make_relation_record(tokens)
        dag = self.composer.compose(rr, tokens)
        self.assertIn('LEX_0', dag.nodes)
        self.assertIn('LEX_1', dag.nodes)

    def test_dag_has_clause_node(self):
        tokens = [
            make_token('ἄνθρωπος', 'ENTITY', 'THEME'),
            make_token('λέγει', 'EVENT', 'PREDICATE'),
        ]
        rr = make_relation_record(tokens)
        dag = self.composer.compose(rr, tokens)
        self.assertIn('CLAUSE_0', dag.nodes)

    def test_cycle_detection_raises(self):
        dag = SemanticDAG()
        n1 = DAGNode('A', 'LEXEME', 'A', 'ENTITY', 'THEME')
        n2 = DAGNode('B', 'LEXEME', 'B', 'EVENT', 'PREDICATE')
        dag.add_node(n1)
        dag.add_node(n2)
        dag.add_edge(DAGEdge('A', 'B', 'SUBJECT'))
        with self.assertRaises(ValueError):
            dag.add_edge(DAGEdge('B', 'A', 'OBJECT'))

    def test_topological_sort_returns_all_nodes(self):
        tokens = [
            make_token('ἄνθρωπος', 'ENTITY', 'THEME'),
            make_token('λέγει', 'EVENT', 'PREDICATE'),
        ]
        rr = make_relation_record(tokens)
        dag = self.composer.compose(rr, tokens)
        order = dag.topological_sort()
        self.assertEqual(set(order), set(dag.nodes.keys()))

    def test_to_dict_is_serializable(self):
        import json
        tokens = [make_token('λέγει', 'EVENT', 'PREDICATE')]
        rr = make_relation_record(tokens)
        dag = self.composer.compose(rr, tokens)
        d = dag.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)

    def test_to_dict_has_nodes_and_edges(self):
        tokens = [make_token('λέγει', 'EVENT', 'PREDICATE')]
        rr = make_relation_record(tokens)
        dag = self.composer.compose(rr, tokens)
        d = dag.to_dict()
        self.assertIn('nodes', d)
        self.assertIn('edges', d)

    def test_get_children(self):
        dag = SemanticDAG()
        n1 = DAGNode('A', 'ROOT', 'A', 'ROOT', 'ROOT')
        n2 = DAGNode('B', 'LEXEME', 'B', 'ENTITY', 'THEME')
        dag.add_node(n1)
        dag.add_node(n2)
        dag.add_edge(DAGEdge('A', 'B', 'SUBJECT'))
        children = dag.get_children('A')
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].node_id, 'B')

    def test_get_parents(self):
        dag = SemanticDAG()
        n1 = DAGNode('A', 'ROOT', 'A', 'ROOT', 'ROOT')
        n2 = DAGNode('B', 'LEXEME', 'B', 'ENTITY', 'THEME')
        dag.add_node(n1)
        dag.add_node(n2)
        dag.add_edge(DAGEdge('A', 'B', 'SUBJECT'))
        parents = dag.get_parents('B')
        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0].node_id, 'A')

    def test_node_count_includes_root_and_clause_and_lexemes(self):
        tokens = [
            make_token('ἄνθρωπος', 'ENTITY', 'THEME'),
            make_token('λέγει', 'EVENT', 'PREDICATE'),
        ]
        rr = make_relation_record(tokens)
        dag = self.composer.compose(rr, tokens)
        # ROOT + CLAUSE_0 + LEX_0 + LEX_1 = 4
        self.assertEqual(len(dag.nodes), 4)

    def test_edge_missing_node_raises(self):
        dag = SemanticDAG()
        n1 = DAGNode('A', 'ROOT', 'A', 'ROOT', 'ROOT')
        dag.add_node(n1)
        with self.assertRaises(ValueError):
            dag.add_edge(DAGEdge('A', 'NONEXISTENT', 'SUBJECT'))


if __name__ == '__main__':
    unittest.main()
