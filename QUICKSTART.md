# Quick Start Guide

## Activate the virtual environment

```powershell
& .\.venv\Scripts\Activate.ps1
```

## Run end-to-end on a single example

```powershell
& python scripts/run_single.py
```

Output: A single example ("θεός γράφει λόγον") compiled through all 6 stages (phoneme → morpheme → lexeme → semantic token → relation → DAG), plus the dev-set batch results.

## Run on the Aristotle excerpt (or any JSONL corpus)

```powershell
# Default: reads Text/Aristotle.txt, segments into sentences, processes first 20
& python scripts/run_batch.py

# Custom corpus with {text, language} JSONL records
& python scripts/run_batch.py --input data/corpus/myfile.jsonl --output outputs/myresults.jsonl
```

Output: JSONL, one compiled result per line.

## Evaluate against gold test sets

```powershell
# Evaluate dev set
& python scripts/evaluate.py --split dev

# Evaluate held-out test set  
& python scripts/evaluate.py --split test
```

Output: JSON report with per-stage metrics (lemma accuracy, semantic type F1, relation F1, DAG validity). Current gold metrics: **100% on both sets** (6 dev + 4 test examples).

## Build DAGs from real UD Ancient Greek

```bash
python scripts/run_ud_dag.py --conllu data/real_eval/grc_perseus-ud-test.conllu --max-sentences 100000 --output outputs/ud_dag_perseus_full.jsonl

python scripts/run_ud_dag.py --conllu data/real_eval/grc_perseus-ud-test.conllu data/real_eval/grc_proiel-ud-test.conllu --max-sentences 100000 --output outputs/ud_dag_combined_full.jsonl
```

This path uses Universal Dependencies as the upstream annotation source, then maps UD lemmas/POS/features/dependencies into the project's semantic DAG format. It evaluates the DAG abstraction on real annotated Greek rather than relying on the toy raw-text analyzer.

## Evaluate Historical Semantic Gold

```bash
python scripts/evaluate_historical_gold.py
python scripts/evaluate_historical_gold.py --labels AGENT THEME COMPLEMENT COORD --output /tmp/historical_core_roles.json
```

The default gold file is `data/gold_semantic/historical_semantic_dags.jsonl`: 21 real Greek sentences from the public PROIEL and Perseus UD test files, manually labeled with this project's semantic DAG edges. The full semantic-graph score is lower than the core-role score because extra predicted edges count as false positives.

## Export graphs for visualization

```powershell
# Export dev-set graphs as JSON and Graphviz DOT
& python scripts/export_graph.py --split dev

# Export test-set graphs
& python scripts/export_graph.py --split test
```

Output: `outputs/graphs/dev_*` and `outputs/graphs/test_*` directories with `.json` and `.dot` files (one per example) plus a summary.

## Visualize a Graphviz DOT file

```bash
# If Graphviz is installed
dot -Tpng outputs/graphs/dev_d1.dot -o dev_d1.png
```

## Add your own gold examples

Edit `data/gold/dev_examples.jsonl` or `data/gold/test_examples.jsonl` with JSON lines of this shape:

```json
{
  "id": "my_example",
  "language": "ancient_greek",
  "text": "θεός λόγον γράφει",
  "tokens": ["θεός", "λόγον", "γράφει"],
  "lemmas": ["θεός", "λόγος", "γράφω"],
  "semantic_types": ["entity", "entity", "event"],
  "relations": [
    {"src": 2, "dst": 0, "label": "AGENT"},
    {"src": 2, "dst": 1, "label": "THEME"}
  ]
}
```

Then re-run `scripts/evaluate.py --split dev` to measure performance.

## Project structure

- **src/**: Core pipeline modules (phoneme, morpheme, lexeme, semantic_tokenizer, syntax_logic, dag, metrics, validate, normalize, similarity_comparator)
- **scripts/**: Entry points (run_single, run_batch, evaluate, export_graph)
- **data/**: Resources (gold examples, lexica, semantic rules, transliteration, phoneme features)
- **outputs/**: Results (phoneme/morpheme/lexeme/batch outputs, evaluation reports, graph visualizations)

## Standard library and academic dependencies

- **NLTK 3.9+**: Text tokenization and sentence splitting  
- **NetworkX 3.2+**: DAG validation and cycle detection  
- **scikit-learn 1.4+**: TF-IDF vectorization and metrics (for similarity fallback and evaluation)

All installed in `.venv/` via `pip install -r requirements.txt`.
