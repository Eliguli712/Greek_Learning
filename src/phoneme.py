import unicodedata
import nltk
from pathlib import Path
import json

# Greek vowels for syllabification and phoneme splitting
GREEK_VOWELS = set("αεηιουω")

def syllabify_word(word):
	"""
	Simple Greek syllabification: split before each vowel (not perfect, but works for demo).
	"""
	syllables = []
	current = ""
	for ch in word:
		if ch in GREEK_VOWELS and current:
			syllables.append(current)
			current = ch
		else:
			current += ch
	if current:
		syllables.append(current)
	return syllables

def split_to_phonemes(word):
	"""
	Split word into characters as phonemes (for demo; real Greek needs digraph handling).
	"""
	return list(word)

def normalize_text(text: str) -> str:
	"""
	Orthographic normalization using Unicode and NLTK (for demonstration).
	Removes diacritics, lowercases, and strips non-letter characters.
	"""
	decomposed = unicodedata.normalize("NFD", text)
	no_diacritics = ''.join(ch for ch in decomposed if unicodedata.category(ch) != 'Mn')
	normalized = ''.join(ch for ch in no_diacritics.lower() if ch.isalpha() or ch.isspace())
	tokens = nltk.word_tokenize(normalized)
	return ' '.join(tokens)

def process_text_to_json(text):
	"""
	For each word, output {"word":..., "syllables":..., "phonemes":...}
	"""
	nltk.download('punkt', quiet=True)
	tokens = nltk.word_tokenize(text)
	results = []
	for word in tokens:
		if not word.isalpha():
			continue
		syllables = syllabify_word(word)
		phonemes = split_to_phonemes(word)
		results.append({
			"word": word,
			"syllables": syllables,
			"phonemes": phonemes
		})
	return results

if __name__ == "__main__":
	nltk.download('punkt', quiet=True)
	aristotle_path = Path("Text/Aristotle.txt")
	if not aristotle_path.exists():
		aristotle_path = Path("../Text/Aristotle.txt")
	output_json = Path("outputs/phoneme_aristotle.json")
	output_words = Path("outputs/phoneme_aristotle.words.txt")
	output_json.parent.mkdir(parents=True, exist_ok=True)
	if aristotle_path.exists():
		with aristotle_path.open("r", encoding="utf-8") as f:
			text = f.read()
		normalized = normalize_text(text)
		results = process_text_to_json(normalized)
		# Write JSON output
		with output_json.open("w", encoding="utf-8") as fout:
			json.dump(results, fout, ensure_ascii=False, indent=2)
		# Write word sequence (one word per line)
		with output_words.open("w", encoding="utf-8") as fout:
			for item in results:
				fout.write(item["word"] + "\n")
		print(f"Phoneme JSON output written to {output_json}")
		print(f"Word sequence output written to {output_words}")
	else:
		print("Aristotle.txt not found.")
