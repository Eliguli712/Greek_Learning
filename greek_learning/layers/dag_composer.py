from dataclasses import dataclass, field, asdict
from collections import deque


@dataclass
class DAGNode:
    node_id: str
    node_type: str
    label: str
    semantic_type: str
    semantic_role: str
    properties: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class DAGEdge:
    source: str
    target: str
    edge_type: str
    weight: float = 1.0

    def to_dict(self):
        return asdict(self)


class SemanticDAG:
    def __init__(self):
        self.nodes: dict = {}
        self.edges: list = []

    def add_node(self, node: DAGNode):
        self.nodes[node.node_id] = node

    def _is_reachable(self, start: str, target: str) -> bool:
        """Return True if target is reachable from start via existing edges."""
        visited = set()
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            for edge in self.edges:
                if edge.source == current and edge.target not in visited:
                    queue.append(edge.target)
        return False

    def add_edge(self, edge: DAGEdge):
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError(
                f"Both source '{edge.source}' and target '{edge.target}' must exist as nodes."
            )
        # Cycle detection: would adding source→target create a cycle?
        # A cycle would exist if target can already reach source.
        if self._is_reachable(edge.target, edge.source):
            raise ValueError("Adding this edge would create a cycle")
        self.edges.append(edge)

    def get_children(self, node_id: str) -> list:
        child_ids = [e.target for e in self.edges if e.source == node_id]
        return [self.nodes[cid] for cid in child_ids if cid in self.nodes]

    def get_parents(self, node_id: str) -> list:
        parent_ids = [e.source for e in self.edges if e.target == node_id]
        return [self.nodes[pid] for pid in parent_ids if pid in self.nodes]

    def topological_sort(self) -> list:
        """Kahn's algorithm."""
        in_degree = {nid: 0 for nid in self.nodes}
        for edge in self.edges:
            in_degree[edge.target] = in_degree.get(edge.target, 0) + 1

        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        result = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for edge in self.edges:
                if edge.source == node:
                    in_degree[edge.target] -= 1
                    if in_degree[edge.target] == 0:
                        queue.append(edge.target)

        if len(result) != len(self.nodes):
            raise ValueError("Graph has a cycle; topological sort not possible.")
        return result

    def to_dict(self) -> dict:
        return {
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
            'edges': [e.to_dict() for e in self.edges],
        }


class DAGComposer:
    def compose(self, relation_record: dict, semantic_tokens: list) -> SemanticDAG:
        dag = SemanticDAG()

        # Create ROOT node
        root = DAGNode(
            node_id='ROOT',
            node_type='ROOT',
            label='ROOT',
            semantic_type='ROOT',
            semantic_role='ROOT',
        )
        dag.add_node(root)

        # Create CLAUSE nodes and LEXEME nodes
        clauses = relation_record.get('clauses', [])
        relations = relation_record.get('relations', [])

        clause_node_ids = []
        for ci, clause in enumerate(clauses):
            clause_id = f'CLAUSE_{ci}'
            clause_node = DAGNode(
                node_id=clause_id,
                node_type='CLAUSE',
                label=f'Clause {ci}',
                semantic_type='CLAUSE',
                semantic_role='CLAUSE',
                properties={'predicate_index': clause.get('predicate')},
            )
            dag.add_node(clause_node)
            clause_node_ids.append(clause_id)
            dag.add_edge(DAGEdge(source='ROOT', target=clause_id, edge_type='SCOPE'))

        # Create LEXEME nodes for each token
        for ti, token in enumerate(semantic_tokens):
            token_id = f'LEX_{ti}'
            lex_node = DAGNode(
                node_id=token_id,
                node_type='LEXEME',
                label=token.get('surface', ''),
                semantic_type=token.get('semantic_type', 'UNKNOWN'),
                semantic_role=token.get('semantic_role', 'UNKNOWN'),
                properties={
                    'lemma': token.get('lemma', ''),
                    'pos': token.get('pos', 'unknown'),
                },
            )
            dag.add_node(lex_node)

        # Attach lexeme nodes to their clause nodes
        for ci, clause in enumerate(clauses):
            clause_id = f'CLAUSE_{ci}'
            for member_idx in clause.get('members', []):
                token_id = f'LEX_{member_idx}'
                if token_id in dag.nodes:
                    # Determine edge type from semantic role
                    token = semantic_tokens[member_idx] if member_idx < len(semantic_tokens) else {}
                    sem_type = token.get('semantic_type', 'UNKNOWN')
                    edge_type = 'PREDICATE'
                    if member_idx < len(relations) or True:
                        # Check relation for this clause
                        if ci < len(relations):
                            rel = relations[ci]
                            if rel.get('subject') == member_idx:
                                edge_type = 'SUBJECT'
                            elif rel.get('object') == member_idx:
                                edge_type = 'OBJECT'
                            elif rel.get('predicate') == member_idx:
                                edge_type = 'PREDICATE'
                            elif member_idx in rel.get('modifiers', []):
                                edge_type = 'MODIFIER'
                            elif member_idx in rel.get('operators', []):
                                edge_type = 'CONNECTOR'
                        elif sem_type == 'FUNCTION':
                            edge_type = 'DETERMINER'
                        elif sem_type == 'PROPERTY':
                            edge_type = 'MODIFIER'
                        elif sem_type == 'OPERATOR':
                            edge_type = 'LOGICAL_OP'
                        elif sem_type == 'DISCOURSE_MARKER':
                            edge_type = 'CONNECTOR'

                    # Avoid duplicate edges
                    existing = any(
                        e.source == clause_id and e.target == token_id
                        for e in dag.edges
                    )
                    if not existing:
                        try:
                            dag.add_edge(DAGEdge(
                                source=clause_id,
                                target=token_id,
                                edge_type=edge_type,
                            ))
                        except ValueError:
                            pass

        # If no clauses, attach all lexeme nodes directly to ROOT
        if not clause_node_ids:
            for ti in range(len(semantic_tokens)):
                token_id = f'LEX_{ti}'
                if token_id in dag.nodes:
                    try:
                        dag.add_edge(DAGEdge(source='ROOT', target=token_id, edge_type='SCOPE'))
                    except ValueError:
                        pass

        return dag
