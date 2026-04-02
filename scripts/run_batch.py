import json
from pathlib import Path
from src.pipeline import SemanticCompiler

resources = {}
compiler = SemanticCompiler(resources)

input_path = Path("data/corpus/passages.jsonl")
output_path = Path("outputs/mass_results.jsonl")

with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line)
        result = compiler.analyze(item["text"], item["language"])
        fout.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")