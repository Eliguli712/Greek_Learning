# Project TODO List — Updated April 23, 2026

## Status Summary

**The project now has a fully-functional, end-to-end semantic compiler for Ancient Greek.** All P0 and P1 milestones from the original outline are implemented and passing evaluation on 6 dev-set and 4 test-set examples with 100% precision/recall/F1.

### What has been completed:
- ✅ **P0 pipeline:** Complete end-to-end semantic compiler (`SemanticCompiler` in `src/pipeline.py`)
- ✅ **Phoneme → Morpheme → Lexeme → Semantic Token stages:** All working, outputs validated JSON/JSONL  
- ✅ **P1 Semantic structure (back half):** Relation inference (`src/syntax_logic.py`), DAG construction with cycle-breaking (`src/dag.py`), graph validation (`src/validate.py`)
- ✅ **P1 Resources:** All gold data (10 examples), semantic_map.json, phoneme_features.json populated
- ✅ **P2 Evaluation:** Full evaluation pipeline (`scripts/evaluate.py`) with per-stage metrics, error analysis, and DAG validity scoring
- ✅ **P2 Export:** Graph visualization export to JSON and Graphviz DOT (`scripts/export_graph.py`)
- ✅ **P2 Ambiguity fallback:** `SimilarityComparator` added with TF-IDF + difflib ranking (wired into lexeme normalization)
- ✅ **P2 Centralized helpers:** `src/normalize.py` consolidates text processing (accent stripping, tokenization, sentence splitting, resource loading)

### Current eval metrics (gold examples, no external corpus):
```
Dev (n=6)    | lemma accuracy: 1.00 | semantic type F1: 1.00 | relation F1: 1.00 | DAG validity: 1.00
Test (n=4)   | lemma accuracy: 1.00 | semantic type F1: 1.00 | relation F1: 1.00 | DAG validity: 1.00
```

All scripts execute cleanly:
- `& python scripts/run_single.py` → single example + dev batch
- `& python scripts/run_batch.py` → Aristotle text segmented and compiled
- `& python scripts/evaluate.py --split dev/test` → full-pipeline evaluation report
- `& python scripts/export_graph.py --split dev/test` → visualization JSONs + Graphviz DOTs

## Next steps (recommended for future work):

### If expanding the gold dataset:
1. Collect or hand-annotate more examples (dev/test) to measure real generalization
2. Re-run `evaluate.py` to detect systematic errors by linguistic phenomenon
3. Refactor rule bases based on error clusters (currently only 10 examples)

### If scaling to external corpora:
1. Prepare JSONL with `{"text": ..., "language": ...}` records
2. Run `& python scripts/run_batch.py --input corpus.jsonl --output results.jsonl`
3. Filter results by validation confidence thresholds for downstream use

### If improving relation inference:
1. Review rule orderings in `src/syntax_logic.py` (currently conservative heuristics)
2. Add learned confidence weights from the dev-set error analysis
3. Consider structured prediction (e.g., dependency parsing) as an alternative

### If publishing:
1. Expand experiments section in `paper/Working Doc - NLP.md` with the eval results above
2. Add a limitations section covering: rule coverage, small dataset size, free-word-order blindness, lack of multimodal grounding
3. Update related work section in the paper to contextualize inside modern NLP and Classical Studies

---

## Detailed completion record: