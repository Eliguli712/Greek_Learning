import unicodedata
from dataclasses import dataclass, field, asdict


@dataclass
class PhonemeRecord:
    original: str
    normalized: str
    phonemes: list
    syllables: list
    language: str

    def to_dict(self):
        return asdict(self)


class PhonemeNormalizer:
    GREEK_VOWELS = {'α', 'ε', 'η', 'ι', 'ο', 'υ', 'ω'}
    GREEK_DIPHTHONGS = {'αι', 'αυ', 'ει', 'ευ', 'οι', 'ου', 'υι', 'ηυ'}
    LATIN_VOWELS = {'a', 'e', 'i', 'o', 'u', 'y'}
    LATIN_DIPHTHONGS = {'ae', 'au', 'ei', 'eu', 'oe', 'ui'}

    GREEK_CHARS = set('αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ')

    def _detect_language(self, text: str) -> str:
        nfd = unicodedata.normalize('NFD', text)
        for ch in nfd:
            if ch in self.GREEK_CHARS:
                return 'greek'
        return 'latin'

    def _strip_diacritics(self, text: str) -> str:
        nfd = unicodedata.normalize('NFD', text)
        return ''.join(ch for ch in nfd if unicodedata.category(ch)[0] != 'M')

    def _syllabify_greek(self, normalized: str) -> list:
        vowels = self.GREEK_VOWELS
        diphthongs = self.GREEK_DIPHTHONGS
        syllables = []
        current = ''
        i = 0
        while i < len(normalized):
            ch = normalized[i]
            # Check for diphthong
            if i + 1 < len(normalized) and normalized[i:i+2] in diphthongs:
                current += normalized[i:i+2]
                i += 2
                syllables.append(current)
                current = ''
            elif ch in vowels:
                current += ch
                i += 1
                syllables.append(current)
                current = ''
            else:
                current += ch
                i += 1
        if current:
            if syllables:
                syllables[-1] += current
            else:
                syllables.append(current)
        return syllables if syllables else [normalized]

    def _syllabify_latin(self, normalized: str) -> list:
        vowels = self.LATIN_VOWELS
        diphthongs = self.LATIN_DIPHTHONGS
        syllables = []
        current = ''
        i = 0
        while i < len(normalized):
            ch = normalized[i].lower()
            if i + 1 < len(normalized) and normalized[i:i+2].lower() in diphthongs:
                current += normalized[i:i+2]
                i += 2
                syllables.append(current)
                current = ''
            elif ch in vowels:
                current += normalized[i]
                i += 1
                syllables.append(current)
                current = ''
            else:
                current += normalized[i]
                i += 1
        if current:
            if syllables:
                syllables[-1] += current
            else:
                syllables.append(current)
        return syllables if syllables else [normalized]

    def normalize(self, text: str) -> dict:
        language = self._detect_language(text)
        normalized = self._strip_diacritics(text)
        phonemes = list(normalized)

        if language == 'greek':
            syllables = self._syllabify_greek(normalized)
        else:
            syllables = self._syllabify_latin(normalized)

        record = PhonemeRecord(
            original=text,
            normalized=normalized,
            phonemes=phonemes,
            syllables=syllables,
            language=language,
        )
        return asdict(record)
