# Output Format Options

The pipeline supports two flexible output formats for compilation results:

## 1. Combined Format (Default)
Single JSON file containing all 6 pipeline stages

### Single Example
```bash
& python scripts/run_single.py --format combined
# Output: outputs/single_result.json (all stages in one file)
```

### Batch Processing
```bash
& python scripts/run_batch.py --format combined --max-records 5
# Output: outputs/batch_results.jsonl (one JSON line per example)
```

**Use case:** Easier to consume with standard JSON tools, single file per analysis

---

## 2. Staged Format
Separate JSON file for each of the 6 pipeline stages, plus metadata

### Single Example
```bash
& python scripts/run_single.py --format staged
# Output directory: outputs/single_result/
#   ├─ single_metadata.json      (text, language, tokens, validation notes)
#   ├─ single_phoneme.json       (syllables + phoneme splitting)
#   ├─ single_morpheme.json      (morpheme segmentation results)
#   ├─ single_lexeme.json        (lemmatization + transliteration)
#   ├─ single_semantic.json      (semantic type/role/class assignment)
#   ├─ single_relations.json     (extracted semantic relations)
#   └─ single_dag.json           (DAG nodes + edges + validation)
```

### Batch Processing
```bash
& python scripts/run_batch.py --format staged --max-records 5
# Output directory structure:
# outputs/batch_results/
#   ├─ summary.json              (index of all examples)
#   ├─ <example_id_1>/
#   │  ├─ output_metadata.json
#   │  ├─ output_phoneme.json
#   │  ├─ output_morpheme.json
#   │  ├─ output_lexeme.json
#   │  ├─ output_semantic.json
#   │  ├─ output_relations.json
#   │  └─ output_dag.json
#   └─ <example_id_2>/
#      └─ (7 files per example)
```

**Use case:** Detailed analysis, debugging specific pipeline stages, downstream stage-specific processing

---

## File Schemas

### staged: metadata.json
```json
{
  "text": "string (original input text)",
  "language": "string (ancient_greek | ...)",
  "sentences": ["array of segmented sentences"],
  "tokens": ["array of word tokens"],
  "validation": {"object (DAG validity report)"},
  "notes": ["array of processing notes"]
}
```

### staged: phoneme.json
```json
[
  {
    "word": "string",
    "syllables": ["array"],
    "phonemes": ["array"]
  }
]
```

### staged: morpheme.json
```json
[
  {
    "token": "string",
    "normalized_token": "string",
    "morphemes": [
      {
        "form": "string",
        "type": "string (root|combining_form|...)",
        "gloss": "string",
        "lemma": "string or null",
        ...
      }
    ],
    "strategy": "string",
    "candidate_score": "float",
    ...
  }
]
```

### staged: lexeme.json
```json
[
  {
    "token": "string",
    "lemma": "string",
    "lexeme": "string",
    "canonical_form": "string (transliteration)",
    "pos": "string",
    "transliteration": "string",
    "strategy": "component_lookup | whole_word_lexicon | similarity_fallback",
    "confidence": "float (0-1)"
  }
]
```

### staged: semantic.json
```json
[
  {
    "token": "string",
    "lemma": "string",
    "semantic_type": "string (entity|event|property|...)",
    "semantic_role": "string (referent|predicate|modifier|...)",
    "semantic_class": "string or null (deity|human|communication|...)",
    "pos": "string",
    "features": {"object"},
    "confidence": "float"
  }
]
```

### staged: relations.json
```json
[
  {
    "src": "integer (token index)",
    "dst": "integer (token index)",
    "label": "string (AGENT|THEME|MODIFIER|COORD|SCOPE)",
    "confidence": "float",
    "rule": "string (rule name that produced this relation)"
  }
]
```

### staged: dag.json
```json
{
  "nodes": [
    {
      "id": "string (n0, n1, ...)",
      "index": "integer",
      "token": "string",
      "lemma": "string",
      "semantic_type": "string",
      "semantic_role": "string",
      ...
    }
  ],
  "edges": [
    {
      "src": "string (node id)",
      "dst": "string (node id)",
      "label": "string (relation type)",
      "confidence": "float",
      "rule": "string"
    }
  ],
  "validation": {
    "ok": "boolean",
    "acyclic": "boolean",
    "illegal_edges": ["array"],
    "dangling_edges": ["array"],
    "missing_required_args": ["array"]
  }
}
```

### combined: full_result.json
Combines all above into a single JSON:
```json
{
  "text": "string",
  "language": "string",
  "sentences": ["array"],
  "tokens": ["array"],
  "phoneme_layer": [... phoneme.json content ...],
  "morpheme_layer": [... morpheme.json content ...],
  "lexeme_layer": [... lexeme.json content ...],
  "semantic_tokens": [... semantic.json content ...],
  "relations": [... relations.json content ...],
  "dag": {... dag.json content ...},
  "validation": {...},
  "notes": [...]
}
```

---

## Comparing Formats

| Aspect | Combined | Staged |
|--------|----------|--------|
| **File count** | 1 file per result | 7 files per result |
| **Ease of use** | Single JSON.load() | Load individual stages |
| **Stage replay** | Extract from single JSON | Direct file access |
| **Pipeline debugging** | Full context integrated | Focus on specific stage |
| **Storage** | Compact (especially JSONL) | More flexible (stage caching) |
| **Downstream use** | Easier streaming | Better for modular processing |

---

## Examples

### Load combined result
```python
import json
with open("outputs/single_result.json") as f:
    result = json.load(f)
    # Access all stages: result["phoneme_layer"], result["semantic_tokens"], etc.
```

### Load staged phoneme only
```python
import json
with open("outputs/single_result/single_phoneme.json") as f:
    phonemes = json.load(f)
    # Process just phoneme data
```

### Iterate JSONL batch (combined)
```python
with open("outputs/batch_results.jsonl") as f:
    for line in f:
        result = json.loads(line)
        # Process each example
```

### Access all examples from batch (staged)
```python
import json
from pathlib import Path

batch_dir = Path("outputs/batch_results")
summary = json.load(open(batch_dir / "summary.json"))

for item in summary:
    example_id = item["id"]
    example_dir = batch_dir / example_id
    
    # Load metadata
    metadata = json.load(open(example_dir / "output_metadata.json"))
    
    # Load specific stages as needed
    dag = json.load(open(example_dir / "output_dag.json"))
    relations = json.load(open(example_dir / "output_relations.json"))
```

---

## Default Behavior

- `scripts/run_single.py`: **combined** format (console output + file save)
- `scripts/run_batch.py`: **combined** format (JSONL output)
- Both support `--format combined` or `--format staged` override

To change defaults, modify the `default="combined"` argument in `parse_args()`.
