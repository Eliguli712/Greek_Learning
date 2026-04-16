## Context Reasoning Pipeline: TODO List

### 1. Implemented Components
- Morpheme segmentation (`morpheme.py`): Rule-based, lexicon-first segmentation.
- Lexeme normalization (`lexeme.py`): Lemma lookup, canonical form, POS assignment.

### 2. Missing / To Be Implemented
- **Phoneme Layer (`phoneme.py`)**
	- Orthographic normalization, syllabification, phoneme feature extraction.
- **Semantic Token Layer (`semantic_tokenizer.py`)**
	- Assign semantic types, roles, and allowed graph relations to tokens.
- **Syntax/Logic Layer (`syntax_logic.py`)**
	- Infer syntactic/logical relations, context scope, and attachment templates.
- **DAG Composition Layer (`dag.py`)**
	- Assemble typed nodes/edges, check acyclicity and closure, build final graph.
- **Pipeline Integration (`pipeline.py`)**
	- Connect all layers into a working pipeline; currently missing main workflow.
- **Validation, Scoring, Export**
	- Implement structure validation, per-layer evaluation, and export (CSV/DAG).
- **Context Reasoning Logic**
	- Implement context-aware reasoning: token disambiguation, scope modeling, relation inference (mainly in `semantic_tokenizer.py`, `syntax_logic.py`, `dag.py`).

### 3. Upstream/Downstream Interfaces
- Ensure each layer outputs structured JSON for the next layer.
- Design clear interfaces for data handoff and error handling.

### 4. Suggestions / Next Steps
- Prioritize implementation of phoneme, semantic token, syntax/logic, and DAG layers.
- Develop the main pipeline workflow to connect all modules.
- Add context reasoning logic for semantic scope, ambiguity, and relation inference.
- Implement validation and scoring for each layer and the overall pipeline.
