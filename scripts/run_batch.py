"""
Run the SemanticCompiler on a JSONL passages file (one passage per line).

The default input is the Aristotle excerpt under Text/, segmented into
sentence-level inputs. A custom JSONL file with {"text": ..., "language": ...}
records may be provided via --input.

Usage:
    & python scripts/run_batch.py
    & python scripts/run_batch.py --input data/corpus/passages.jsonl
    & python scripts/run_batch.py --format staged          # 6 JSONs per example
    & python scripts/run_batch.py --format combined        # single JSONL (default)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.normalize import sentence_split
from src.pipeline import SemanticCompiler
from src.output_format import save_batch_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional JSONL file with {text, language} records.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "batch_results",
        help="Output path (file for combined mode, directory for staged mode).",
    )
    parser.add_argument(
        "--format",
        choices=["combined", "staged"],
        default="combined",
        help="Output format: 'combined' (JSONL) or 'staged' (separate JSON files)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=20,
        help="Cap the number of records processed (useful for first runs).",
    )
    return parser.parse_args()


def iter_records(input_path: Path | None, max_records: int) -> Iterable[dict]:
    if input_path is not None:
        with input_path.open("r", encoding="utf-8") as handle:
            for i, line in enumerate(handle):
                if i >= max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    aristotle = PROJECT_ROOT / "Text" / "Aristotle.txt"
    if not aristotle.exists():
        return
    text = aristotle.read_text(encoding="utf-8")
    for i, sentence in enumerate(sentence_split(text)):
        if i >= max_records:
            break
        yield {"text": sentence, "language": "ancient_greek"}


def main() -> None:
    args = parse_args()
    compiler = SemanticCompiler(project_root=PROJECT_ROOT)

    # Append format-specific suffix if needed
    output_path = args.output
    if args.format == "combined" and not str(output_path).endswith(".jsonl"):
        output_path = Path(str(output_path) + ".jsonl")

    count = 0
    results = []
    for record in iter_records(args.input, args.max_records):
        result = compiler.analyze(record["text"], record.get("language", "ancient_greek"))
        example_id = record.get("id")
        results.append((example_id, result))
        count += 1

    save_batch_results(results, output_path, format_mode=args.format)

    if args.format == "combined":
        print(f"✓ Wrote {count} results to {output_path}")
    else:
        print(f"✓ Wrote {count} examples to {output_path}/")
        print(f"  Each example has 7 files (metadata + 6 stages)")


if __name__ == "__main__":
    main()