import unicodedata
from dataclasses import dataclass, field, asdict


@dataclass
class LatinPhonemeRecord:
    original: str
    normalized: str
    phonemes: list
    syllables: list
    language: str

    def to_dict(self):
        return asdict(self)


class LatinPhonemeNormalizer:
    VOWELS = {'a', 'e', 'i', 'o', 'u', 'y'}
    DIPHTHONGS = {'ae', 'au', 'ei', 'eu', 'oe', 'ui'}

    # Macron mappings
    MACRON_MAP = str.maketrans('āēīōūĀĒĪŌŪ', 'aeiouAEIOU')

    def _strip_diacritics(self, text: str) -> str:
        # First replace macrons
        text = text.translate(self.MACRON_MAP)
        nfd = unicodedata.normalize('NFD', text)
        return ''.join(ch for ch in nfd if unicodedata.category(ch)[0] != 'M')

    def _syllabify(self, normalized: str) -> list:
        vowels = self.VOWELS
        diphthongs = self.DIPHTHONGS
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
        normalized = self._strip_diacritics(text)
        phonemes = list(normalized)
        syllables = self._syllabify(normalized)

        record = LatinPhonemeRecord(
            original=text,
            normalized=normalized,
            phonemes=phonemes,
            syllables=syllables,
            language='latin',
        )
        return asdict(record)
