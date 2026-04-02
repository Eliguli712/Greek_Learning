from dataclasses import dataclass, field, asdict


POS_TO_SEMANTIC_TYPE = {
    'noun': 'ENTITY',
    'verb': 'EVENT',
    'adjective': 'PROPERTY',
    'adverb': 'PROPERTY',
    'preposition': 'RELATION',
    'conjunction': 'OPERATOR',
    'article': 'FUNCTION',
    'particle': 'DISCOURSE_MARKER',
    'pronoun': 'ENTITY',
    'unknown': 'UNKNOWN',
}

POS_TO_DEFAULT_ROLE = {
    'noun': 'THEME',
    'verb': 'PREDICATE',
    'adjective': 'MODIFIER',
    'adverb': 'MODIFIER',
    'preposition': 'RELATION',
    'conjunction': 'CONNECTOR',
    'article': 'DETERMINER',
    'particle': 'DISCOURSE_MARKER',
    'pronoun': 'AGENT',
    'unknown': 'UNKNOWN',
}


@dataclass
class SemanticToken:
    surface: str
    lemma: str
    pos: str
    semantic_type: str
    semantic_role: str
    features: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class SemanticTokenizer:
    def tokenize(self, lexeme_record: dict) -> dict:
        surface = lexeme_record.get('surface', '')
        lemma = lexeme_record.get('lemma', '')
        pos = lexeme_record.get('pos', 'unknown')

        semantic_type = POS_TO_SEMANTIC_TYPE.get(pos, 'UNKNOWN')
        semantic_role = POS_TO_DEFAULT_ROLE.get(pos, 'UNKNOWN')

        features = {
            'in_lexicon': lexeme_record.get('in_lexicon', False),
        }

        token = SemanticToken(
            surface=surface,
            lemma=lemma,
            pos=pos,
            semantic_type=semantic_type,
            semantic_role=semantic_role,
            features=features,
        )
        return asdict(token)
