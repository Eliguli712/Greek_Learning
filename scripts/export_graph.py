"""
Export semantic DAGs produced by the pipeline to JSON and Graphviz DOT.

Usage:
    & .venv/Scripts/python.exe scripts/export_graph.py --split dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import networkx as nx

from src.dag import SemanticDAG
from src.pipeline import SemanticCompiler


SPLIT_PATHS = {
    "dev": PROJECT_ROOT / "data/gold/dev_examples.jsonl",
    "test": PROJECT_ROOT / "data/gold/test_examples.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLIT_PATHS.keys()), default="dev")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "graphs",
    )
    return parser.parse_args()


def load_gold(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def to_dot(example_id: str, dag_dict: Dict[str, Any]) -> str:
    lines = [f'digraph "{example_id}" {{', '  rankdir=LR;', '  node [shape=box, fontname="Helvetica"];']
    for node in dag_dict["nodes"]:
        label_parts = [
            node.get("token", ""),
            f"lemma={node.get('lemma','')}",
            f"{node.get('semantic_type','?')}/{node.get('semantic_role','?')}",
        ]
        label = "\\n".join(label_parts).replace('"', "'")
        lines.append(f'  "{node["id"]}" [label="{label}"];')
    for edge in dag_dict["edges"]:
        lines.append(
            f'  "{edge["src"]}" -> "{edge["dst"]}" [label="{edge["label"]}"];'
        )
    lines.append("}")
    return "\n".join(lines)


def export(split: str, out_dir: Path) -> None:
    path = SPLIT_PATHS[split]
    if not path.exists():
        raise FileNotFoundError(f"Gold file missing: {path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    compiler = SemanticCompiler(project_root=PROJECT_ROOT)

    summary: List[Dict[str, Any]] = []
    for example in load_gold(path):
        result = compiler.analyze(example["text"], example.get("language", "ancient_greek"))
        ex_id = example["id"]

        json_path = out_dir / f"{split}_{ex_id}.json"
        dot_path = out_dir / f"{split}_{ex_id}.dot"

        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, ensure_ascii=False, indent=2)
        with dot_path.open("w", encoding="utf-8") as handle:
            handle.write(to_dot(ex_id, result.dag))

        summary.append(
            {
                "id": ex_id,
                "text": example["text"],
                "n_nodes": len(result.dag.get("nodes", [])),
                "n_edges": len(result.dag.get("edges", [])),
                "valid": result.validation.get("ok", False),
                "json": str(json_path.relative_to(PROJECT_ROOT).as_posix()),
                "dot": str(dot_path.relative_to(PROJECT_ROOT).as_posix()),
            }
        )

    summary_path = out_dir / f"{split}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Exported {len(summary)} graphs to {out_dir}")
    print(f"Summary written to {summary_path}")


def main() -> None:
    args = parse_args()
    export(args.split, args.out_dir)


if __name__ == "__main__":
    main()
