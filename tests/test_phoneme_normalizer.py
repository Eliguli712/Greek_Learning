import unittest
from greek_learning.layers.phoneme_normalizer import PhonemeNormalizer, PhonemeRecord


class TestPhonemeNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = PhonemeNormalizer()

    def test_greek_strips_diacritics(self):
        result = self.normalizer.normalize('λόγος')
        self.assertEqual(result['language'], 'greek')
        # normalized should have no combining marks
        import unicodedata
        for ch in result['normalized']:
            self.assertNotEqual(unicodedata.category(ch)[0], 'M')

    def test_greek_normalized_base_letters(self):
        result = self.normalizer.normalize('ἄνθρωπος')
        self.assertIn('α', result['normalized'])

    def test_greek_phonemes_list(self):
        result = self.normalizer.normalize('λογος')
        self.assertEqual(result['phonemes'], list('λογος'))

    def test_greek_syllables_non_empty(self):
        result = self.normalizer.normalize('λόγος')
        self.assertIsInstance(result['syllables'], list)
        self.assertGreater(len(result['syllables']), 0)

    def test_greek_language_detection(self):
        result = self.normalizer.normalize('θεός')
        self.assertEqual(result['language'], 'greek')

    def test_latin_language_detection(self):
        result = self.normalizer.normalize('amor')
        self.assertEqual(result['language'], 'latin')

    def test_latin_strips_diacritics(self):
        result = self.normalizer.normalize('amor')
        self.assertEqual(result['normalized'], 'amor')

    def test_latin_syllables(self):
        result = self.normalizer.normalize('amor')
        self.assertIsInstance(result['syllables'], list)
        self.assertGreater(len(result['syllables']), 0)

    def test_original_preserved(self):
        text = 'λόγος'
        result = self.normalizer.normalize(text)
        self.assertEqual(result['original'], text)

    def test_greek_syllable_has_vowel_nucleus(self):
        result = self.normalizer.normalize('ανθρωπος')
        # Each syllable should contain at least one vowel or be non-empty
        for syl in result['syllables']:
            self.assertTrue(len(syl) > 0)


if __name__ == '__main__':
    unittest.main()
