from greek_learning.layers.phoneme_normalizer import PhonemeNormalizer
from greek_learning.layers.morpheme_segmenter import MorphemeSegmenter
from greek_learning.layers.lexeme_normalizer import LexemeNormalizer
from greek_learning.layers.semantic_tokenizer import SemanticTokenizer
from greek_learning.layers.relation_inferencer import RelationInferencer
from greek_learning.layers.dag_composer import DAGComposer


class SingleRunner:
    def __init__(self, language: str = 'greek'):
        self.language = language
        self.phoneme_normalizer = PhonemeNormalizer()
        self.morpheme_segmenter = MorphemeSegmenter()
        self.lexeme_normalizer = LexemeNormalizer()
        self.semantic_tokenizer = SemanticTokenizer()
        self.relation_inferencer = RelationInferencer()
        self.dag_composer = DAGComposer()

    def run(self, text: str) -> dict:
        raw_tokens = text.split()
        token_records = []
        semantic_tokens = []

        for raw_token in raw_tokens:
            phoneme_record = self.phoneme_normalizer.normalize(raw_token)
            normalized = phoneme_record['normalized']
            lang = phoneme_record.get('language', self.language)

            morpheme_record = self.morpheme_segmenter.segment(normalized, language=lang)
            lexeme_record = self.lexeme_normalizer.normalize(
                raw_token, morpheme_record['segments'], language=lang
            )
            semantic_token = self.semantic_tokenizer.tokenize(lexeme_record)
            semantic_tokens.append(semantic_token)

            token_records.append({
                'token': raw_token,
                'phoneme_record': phoneme_record,
                'morpheme_record': morpheme_record,
                'lexeme_record': lexeme_record,
                'semantic_token': semantic_token,
            })

        relation_record = self.relation_inferencer.infer(semantic_tokens)
        dag = self.dag_composer.compose(relation_record, semantic_tokens)

        return {
            'input': text,
            'language': self.language,
            'tokens': token_records,
            'relations': relation_record,
            'dag': dag.to_dict(),
        }
