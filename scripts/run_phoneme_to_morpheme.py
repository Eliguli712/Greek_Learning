
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from morpheme import MorphemeSegmenter

# Load word list from phoneme layer output
words_path = Path("outputs/phoneme_aristotle.words.txt")
output_path = Path("outputs/morpheme_aristotle.json")

with words_path.open("r", encoding="utf-8") as fin:
    words = [line.strip() for line in fin if line.strip()]

segmenter = MorphemeSegmenter()
results = []
for word in words:
    try:
        analysis = segmenter.analyze(word)
        results.append(analysis.as_dict())
    except Exception as e:
        results.append({"token": word, "error": str(e)})

with output_path.open("w", encoding="utf-8") as fout:
    json.dump(results, fout, ensure_ascii=False, indent=2)

print(f"Morpheme analysis written to {output_path}")
