from dataclasses import dataclass, field, asdict


@dataclass
class RelationRecord:
    tokens: list
    relations: list
    clauses: list

    def to_dict(self):
        return asdict(self)


class RelationInferencer:
    def infer(self, semantic_tokens: list) -> dict:
        tokens = semantic_tokens
        relations = []
        clauses = []

        # Find predicate indices (EVENT tokens)
        predicate_indices = [i for i, t in enumerate(tokens) if t.get('semantic_type') == 'EVENT']

        for pred_idx in predicate_indices:
            relation = {
                'subject': None,
                'predicate': pred_idx,
                'object': None,
                'modifiers': [],
                'operators': [],
            }

            # Find nearest ENTITY to the left → subject
            for i in range(pred_idx - 1, -1, -1):
                t = tokens[i]
                if t.get('semantic_type') == 'ENTITY':
                    relation['subject'] = i
                    break

            # Find nearest ENTITY to the right → object
            for i in range(pred_idx + 1, len(tokens)):
                t = tokens[i]
                if t.get('semantic_type') == 'ENTITY':
                    relation['object'] = i
                    break

            # PROPERTY tokens → modifiers
            for i, t in enumerate(tokens):
                if t.get('semantic_type') == 'PROPERTY':
                    relation['modifiers'].append(i)

            # OPERATOR tokens → operators
            for i, t in enumerate(tokens):
                if t.get('semantic_type') == 'OPERATOR':
                    relation['operators'].append(i)

            relations.append(relation)

        # Build clause groupings
        if relations:
            for rel in relations:
                members = []
                for key in ('subject', 'predicate', 'object'):
                    if rel[key] is not None:
                        members.append(rel[key])
                members.extend(rel['modifiers'])
                members.extend(rel['operators'])
                # add FUNCTION (articles) and DISCOURSE_MARKER
                for i, t in enumerate(tokens):
                    if t.get('semantic_type') in ('FUNCTION', 'DISCOURSE_MARKER'):
                        if i not in members:
                            members.append(i)
                clause = {
                    'predicate': rel['predicate'],
                    'members': sorted(set(members)),
                }
                clauses.append(clause)
        else:
            # No verbs: single clause with all tokens
            clauses.append({
                'predicate': None,
                'members': list(range(len(tokens))),
            })

        record = RelationRecord(tokens=tokens, relations=relations, clauses=clauses)
        return asdict(record)
