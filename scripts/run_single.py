import json
from src.pipeline import SemanticCompiler

# single pipeline test
resources = {}
compiler = SemanticCompiler(resources)

text = "μεγαλόδους"
lang = "ancient_greek"

result = compiler.analyze(text, lang)
print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))

# mutltiple pipeline test
resources = {}
compiler = SemanticCompiler(resources)

with open("data/gold/dev_examples.jsonl", "r", encoding="utf-8") as f:
    items = [json.loads(line) for line in f]

results = []
for item in items:
    result = compiler.analyze(item["text"], item["language"])
    results.append(result.__dict__)

with open("outputs/dev_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)