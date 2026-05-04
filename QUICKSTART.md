# Quick Start Guide

## Activate the virtual environment

```powershell
& .\.venv\Scripts\Activate.ps1
```

You can also run commands without activation by prefixing them with
`.\.venv\Scripts\python.exe`.

## Experiment Paths

The project now has three separate evaluation paths. Keep them distinct when
reporting scores:

| Path | Input | Uses UD dependency edges? | Main command | Current not-raw-pipeline score |
| --- | --- | --- | --- | --- |
| Raw toy compiler | Raw text from `data/gold/*.jsonl` | No | `python scripts/evaluate.py --split test` | relation F1 `0.8889` |
| Historical compiler | UD tokens/POS/morph/dependencies + manual semantic gold | Yes | `python scripts/evaluate_historical_gold.py --system compiler` | full F1 `0.9398` |
| Historical clean baseline | UD tokens/POS/morph only + manual semantic gold | No | `python scripts/evaluate_historical_gold.py --system baseline` | full F1 `0.6241`, core F1 `0.6400` |
| Large-scale UD proxy baseline | UD tokens/POS/morph only + UD-derived proxy gold | No | `python scripts/evaluate_ud_baseline.py --system baseline ...` | about `0.40` on held-out/proxy slices |
| Parser-backed UD proxy | UD tokens/POS/morph/dependencies + UD-derived proxy gold | Yes | `python scripts/evaluate_ud_baseline.py --system compiler ...` | `1.0` sanity check |
| Raw pipeline UD proxy | Raw UD sentence text through project raw compiler | No UD edges at inference | `python scripts/evaluate_ud.py ...` | around `0.2992` on 200 sentences |
| Raw pipeline + relation classifier | Raw UD sentence text + classifier trained from UD splits | No dependency edges at inference | `python scripts/train_relation_classifier.py ...` | around `0.3236` on 500 test sentences |

## Reproduce Current Paper-Facing Scores

Historical manual semantic gold, full label set:

```powershell
python scripts/evaluate_historical_gold.py `
  --system compiler `
  --output outputs/historical_semantic_gold_compiler_eval.json

python scripts/evaluate_historical_gold.py `
  --system baseline `
  --output outputs/historical_semantic_gold_baseline_eval.json
```

Historical manual semantic gold, core roles only:

```powershell
python scripts/evaluate_historical_gold.py `
  --system compiler `
  --labels AGENT THEME COMPLEMENT COORD `
  --output outputs/historical_semantic_gold_compiler_core_eval.json

python scripts/evaluate_historical_gold.py `
  --system baseline `
  --labels AGENT THEME COMPLEMENT COORD `
  --output outputs/historical_semantic_gold_baseline_core_eval.json
```

Current expected relation F1 on `not-raw-pipeline`:

- compiler full: `0.9398`
- compiler core: `0.9319`
- clean baseline full: `0.6241`
- clean baseline core: `0.6400`

Interpretation:

- `compiler` uses UD dependency edges and is the main UD-backed semantic DAG compiler.
- `baseline` ignores UD dependency edges. It uses token, lemma, POS, morphology, and linear order only. These are raw-text-derivable from an upstream tagger, but the current historical experiment reads gold UD annotations from CoNLL-U.
- Full labels are `AGENT THEME MODIFIER COMPLEMENT COORD`.
- Core labels are `AGENT THEME COMPLEMENT COORD`.

## Large-Scale UD Proxy Checks

Download official UD train/dev/test splits when you need the relation-classifier
path. The files are ignored by git under `data/ud_treebanks/`.

```powershell
python scripts/download_ud_treebanks.py
```

Dependency-free clean baseline against UD-derived proxy gold:

```powershell
python scripts/evaluate_ud_baseline.py `
  --system baseline `
  --conllu data/real_eval/grc_perseus-ud-test.conllu data/real_eval/grc_proiel-ud-test.conllu `
  --max-sentences 500 `
  --output outputs/ud_proxy_baseline_500.json
```

Parser-backed upper-bound sanity check on the same proxy task:

```powershell
python scripts/evaluate_ud_baseline.py `
  --system compiler `
  --conllu data/real_eval/grc_perseus-ud-test.conllu data/real_eval/grc_proiel-ud-test.conllu `
  --max-sentences 500 `
  --output outputs/ud_proxy_compiler_500.json
```

Raw-text pipeline against UD proxy labels:

```powershell
python scripts/evaluate_ud.py `
  --conllu data/real_eval/grc_perseus-ud-test.conllu data/real_eval/grc_proiel-ud-test.conllu `
  --max-sentences 200 `
  --output outputs/raw_pipeline_ud_eval_200.json
```

Raw-text pipeline with a supervised relation classifier trained from official
UD train splits:

```powershell
python scripts/train_relation_classifier.py `
  --max-sentences 500 `
  --output outputs/relation_classifier_report_500.json
```

The large-scale proxy checks use UD dependency mapping as silver/proxy gold.
They are useful for scale and sanity, but they are not the same as the manual
historical semantic gold benchmark.

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

## Evaluate Historical Gold with Train/Test Splits

```bash
# Default: leave one corpus out, train filter choices on the other corpus
python scripts/evaluate_historical_splits.py

# Alternative: make a deterministic train/test split inside each corpus
python scripts/evaluate_historical_splits.py --mode within-corpus --test-ratio 0.35
```

The split evaluator writes one JSON report plus separate train/test JSONL files for each fold under `outputs/historical_semantic_splits/`. Modifier-filter choices are selected from the training fold only, then applied to the held-out test fold.

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
