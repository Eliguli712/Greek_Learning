from dataclasses import dataclass, field, asdict


@dataclass
class LexemeRecord:
    surface: str
    lemma: str
    pos: str
    in_lexicon: bool

    def to_dict(self):
        return asdict(self)


GREEK_LEXICON = {
    'λόγος': ('λόγος', 'noun'),
    'λόγου': ('λόγος', 'noun'),
    'λόγῳ': ('λόγος', 'noun'),
    'λόγον': ('λόγος', 'noun'),
    'λόγοι': ('λόγος', 'noun'),
    'ἄνθρωπος': ('ἄνθρωπος', 'noun'),
    'ἀνθρώπου': ('ἄνθρωπος', 'noun'),
    'ἀνθρώπῳ': ('ἄνθρωπος', 'noun'),
    'ἄνθρωπον': ('ἄνθρωπος', 'noun'),
    'ἀνθρώπων': ('ἄνθρωπος', 'noun'),
    'θεός': ('θεός', 'noun'),
    'θεοῦ': ('θεός', 'noun'),
    'θεῷ': ('θεός', 'noun'),
    'θεόν': ('θεός', 'noun'),
    'θεοί': ('θεός', 'noun'),
    'φιλεῖ': ('φιλέω', 'verb'),
    'φιλεῖς': ('φιλέω', 'verb'),
    'φιλῶ': ('φιλέω', 'verb'),
    'φιλοῦμεν': ('φιλέω', 'verb'),
    'λέγει': ('λέγω', 'verb'),
    'λέγεις': ('λέγω', 'verb'),
    'λέγω': ('λέγω', 'verb'),
    'λέγομεν': ('λέγω', 'verb'),
    'ἔλεγεν': ('λέγω', 'verb'),
    'καλός': ('καλός', 'adjective'),
    'καλή': ('καλός', 'adjective'),
    'καλόν': ('καλός', 'adjective'),
    'καλοῦ': ('καλός', 'adjective'),
    'σοφός': ('σοφός', 'adjective'),
    'σοφή': ('σοφός', 'adjective'),
    'σοφόν': ('σοφός', 'adjective'),
    'ὁ': ('ὁ', 'article'),
    'ἡ': ('ὁ', 'article'),
    'τό': ('ὁ', 'article'),
    'τοῦ': ('ὁ', 'article'),
    'τῆς': ('ὁ', 'article'),
    'τῷ': ('ὁ', 'article'),
    'τήν': ('ὁ', 'article'),
    'τόν': ('ὁ', 'article'),
    'καί': ('καί', 'conjunction'),
    'ἤ': ('ἤ', 'conjunction'),
    'ἀλλά': ('ἀλλά', 'conjunction'),
    'δέ': ('δέ', 'particle'),
    'γάρ': ('γάρ', 'particle'),
    'μέν': ('μέν', 'particle'),
    'οὖν': ('οὖν', 'particle'),
    'ἐν': ('ἐν', 'preposition'),
    'ἐκ': ('ἐκ', 'preposition'),
    'ἀπό': ('ἀπό', 'preposition'),
    'πρός': ('πρός', 'preposition'),
    'εἰς': ('εἰς', 'preposition'),
    'ἐπί': ('ἐπί', 'preposition'),
    'διά': ('διά', 'preposition'),
    'κατά': ('κατά', 'preposition'),
    'ψυχή': ('ψυχή', 'noun'),
    'ψυχῆς': ('ψυχή', 'noun'),
    'ψυχῇ': ('ψυχή', 'noun'),
    'ψυχήν': ('ψυχή', 'noun'),
    'νοῦς': ('νοῦς', 'noun'),
    'νοῦ': ('νοῦς', 'noun'),
    'νῷ': ('νοῦς', 'noun'),
    'νοῦν': ('νοῦς', 'noun'),
    'ἀγαθός': ('ἀγαθός', 'adjective'),
    'ἀγαθή': ('ἀγαθός', 'adjective'),
    'ἀγαθόν': ('ἀγαθός', 'adjective'),
}

LATIN_LEXICON = {
    'amor': ('amor', 'noun'),
    'amoris': ('amor', 'noun'),
    'amori': ('amor', 'noun'),
    'amorem': ('amor', 'noun'),
    'homo': ('homo', 'noun'),
    'hominis': ('homo', 'noun'),
    'homini': ('homo', 'noun'),
    'hominem': ('homo', 'noun'),
    'deus': ('deus', 'noun'),
    'dei': ('deus', 'noun'),
    'deo': ('deus', 'noun'),
    'deum': ('deus', 'noun'),
    'amat': ('amo', 'verb'),
    'amas': ('amo', 'verb'),
    'amo': ('amo', 'verb'),
    'amamus': ('amo', 'verb'),
    'dicit': ('dico', 'verb'),
    'dicis': ('dico', 'verb'),
    'dico': ('dico', 'verb'),
    'bonus': ('bonus', 'adjective'),
    'bona': ('bonus', 'adjective'),
    'bonum': ('bonus', 'adjective'),
    'et': ('et', 'conjunction'),
    'sed': ('sed', 'conjunction'),
    'in': ('in', 'preposition'),
    'ex': ('ex', 'preposition'),
    'ad': ('ad', 'preposition'),
    'de': ('de', 'preposition'),
}


class LexemeNormalizer:
    def __init__(self):
        self.greek_lexicon = GREEK_LEXICON
        self.latin_lexicon = LATIN_LEXICON

    def normalize(self, token: str, morpheme_segments: list, language: str = 'greek') -> dict:
        lexicon = self.greek_lexicon if language == 'greek' else self.latin_lexicon

        if token in lexicon:
            lemma, pos = lexicon[token]
            in_lexicon = True
        else:
            in_lexicon = False
            # Use root morpheme as best-guess lemma
            root_seg = next((s for s in morpheme_segments if s.get('role') == 'root'), None)
            lemma = root_seg['form'] if root_seg else token
            pos = 'unknown'

        record = LexemeRecord(surface=token, lemma=lemma, pos=pos, in_lexicon=in_lexicon)
        return asdict(record)
