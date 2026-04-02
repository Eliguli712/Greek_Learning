import unittest
from greek_learning.layers.morpheme_segmenter import MorphemeSegmenter


class TestMorphemeSegmenter(unittest.TestCase):
    def setUp(self):
        self.segmenter = MorphemeSegmenter()

    def _get_roles(self, segments):
        return [s['role'] for s in segments]

    def _get_forms(self, segments):
        return [s['form'] for s in segments]

    def test_greek_simple_word(self):
        result = self.segmenter.segment('λογος', language='greek')
        self.assertEqual(result['token'], 'λογος')
        self.assertIsInstance(result['segments'], list)
        self.assertGreater(len(result['segments']), 0)

    def test_greek_prefix_detection_syn(self):
        # συν + some word
        result = self.segmenter.segment('συνλογος', language='greek')
        forms = self._get_forms(result['segments'])
        roles = self._get_roles(result['segments'])
        self.assertIn('prefix', roles)
        self.assertIn('συν', forms)

    def test_greek_prefix_detection_kata(self):
        result = self.segmenter.segment('καταλογος', language='greek')
        roles = self._get_roles(result['segments'])
        self.assertIn('prefix', roles)

    def test_greek_verb_ending_stripped(self):
        # λεγ + ει
        result = self.segmenter.segment('λεγει', language='greek')
        roles = self._get_roles(result['segments'])
        self.assertIn('ending', roles)
        self.assertIn('root', roles)

    def test_greek_noun_ending_stripped(self):
        # λογ + ος
        result = self.segmenter.segment('λογος', language='greek')
        roles = self._get_roles(result['segments'])
        self.assertIn('root', roles)

    def test_greek_unknown_word_is_root(self):
        result = self.segmenter.segment('χχχχχ', language='greek')
        roles = self._get_roles(result['segments'])
        # Should have at least a root
        self.assertIn('root', roles)

    def test_latin_prefix_detection(self):
        result = self.segmenter.segment('contradicere', language='latin')
        roles = self._get_roles(result['segments'])
        self.assertIn('prefix', roles)

    def test_latin_verb_ending(self):
        result = self.segmenter.segment('amamus', language='latin')
        roles = self._get_roles(result['segments'])
        self.assertIn('root', roles)

    def test_latin_noun_ending(self):
        result = self.segmenter.segment('dominus', language='latin')
        roles = self._get_roles(result['segments'])
        self.assertIn('root', roles)

    def test_language_in_result(self):
        result = self.segmenter.segment('λογος', language='greek')
        self.assertEqual(result['language'], 'greek')

    def test_latin_language_in_result(self):
        result = self.segmenter.segment('amamus', language='latin')
        self.assertEqual(result['language'], 'latin')


if __name__ == '__main__':
    unittest.main()
